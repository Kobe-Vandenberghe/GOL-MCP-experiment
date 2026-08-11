"""Authoritative owner of the shared world.

Everything that mutates the board goes through WorldService: the MCP tools and
the browser both call the same methods, so there is one implementation of
"create a world", "advance N generations", "rewind", and so on.

The service owns the world, the lock, the rewind history and the autorun loop.
It knows nothing about WebSockets, MCP or HTTP — when state changes it calls
the `on_change` coroutine it was constructed with, and the transport layer
decides what to broadcast. Operations return dataclasses rather than prose so
that callers can format, log or serialize them independently.
"""
from __future__ import annotations

import asyncio
import bisect
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

import numpy as np

from gol_world import (
    SimulationError,
    World,
    next_grid_of,
    parse_rle,
    simulate,
    validate_sim_params,
)

MAX_ADVANCE = 100          # cap on generations per animated advance() call
MAX_OBSERVE_CELLS = 20000  # cap ASCII output size for observe()
MAX_FPS = 1000.0
MIN_FPS = 1.0
HISTORY_SNAPSHOT_INTERVAL = 10
HISTORY_MAX_SNAPSHOTS = 2000

Coords = Sequence[Sequence[int]]
BBox = tuple[int, int, int, int] | None


def pack_grid(grid: np.ndarray) -> bytes:
    return np.packbits(grid, axis=None).tobytes()


def unpack_grid(packed: bytes, width: int, height: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=width * height)
    return bits.reshape((height, width)).astype(np.uint8, copy=False)


# ------------------------------------------------------------------ results


@dataclass(frozen=True)
class WorldStatus:
    width: int
    height: int
    edge: str
    generation: int
    population: int
    bbox: BBox
    dynamics: str
    running: bool
    fps: float


@dataclass(frozen=True)
class CreateResult:
    width: int
    height: int
    edge: str
    random_fill: float
    population: int


@dataclass(frozen=True)
class ClearResult:
    width: int
    height: int


@dataclass(frozen=True)
class SetCellsResult:
    set_alive: int
    set_dead: int
    skipped: list[tuple[int, int]]
    population: int


@dataclass(frozen=True)
class PlacementResult:
    pattern_width: int
    pattern_height: int
    x: int
    y: int
    placed_cells: int
    skipped: list[tuple[int, int]]
    population: int
    generation: int
    bbox: BBox


@dataclass(frozen=True)
class AdvanceResult:
    generations: int
    generation: int
    population_before: int
    population: int
    dynamics: str

    @property
    def population_delta(self) -> int:
        return self.population - self.population_before


@dataclass(frozen=True)
class AutorunResult:
    running: bool
    fps: float
    generation: int
    population: int


@dataclass(frozen=True)
class RewindResult:
    from_generation: int
    to_generation: int
    population: int
    dropped_snapshots: int


@dataclass(frozen=True)
class TimelineStatus:
    current_generation: int
    earliest_snapshot: int
    latest_snapshot: int
    max_reachable_generation: int
    snapshot_interval: int
    snapshot_count: int
    snapshot_cap: int


@dataclass(frozen=True)
class SimulationRun:
    """Result of advance_generations / preview_generations.

    `error` is set instead of samples/summary when the requested parameters
    exceed a cap — callers surface it as a structured object rather than
    raising, so an agent can read the suggested limit and retry.
    """
    samples: list[dict] = field(default_factory=list)
    summary: dict | None = None
    error: dict | None = None
    committed: bool = False

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class ObserveResult:
    """The rendered window plus the window actually used, so callers can
    report what was shown when the caller passed 0 to mean "whole board"."""
    text: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class GenerationPreview:
    generation: int
    population: int
    packed_cells: bytes


@dataclass(frozen=True)
class StateSnapshot:
    """Everything the browser needs for one frame. Packed, not encoded — the
    transport layer owns the wire format."""
    width: int
    height: int
    edge: str
    generation: int
    population: int
    running: bool
    fps: float
    timeline_min_generation: int
    timeline_latest_snapshot: int
    timeline_latest_generation: int
    timeline_snapshot_interval: int
    timeline_snapshot_count: int
    packed_cells: bytes


# ------------------------------------------------------------------ service


class WorldService:
    def __init__(
        self,
        world: World | None = None,
        *,
        on_change: Callable[[], Awaitable[None]] | None = None,
        snapshot_interval: int = HISTORY_SNAPSHOT_INTERVAL,
        snapshot_cap: int = HISTORY_MAX_SNAPSHOTS,
    ):
        self.world = world if world is not None else World(160, 100, edge="wrap")
        self.lock = asyncio.Lock()
        self.running = False
        self.fps = 10.0
        self.run_task: asyncio.Task | None = None
        self.snapshot_interval = snapshot_interval
        self.snapshot_cap = snapshot_cap
        self.history_generations: list[int] = []
        self.history_snapshots: dict[int, bytes] = {}
        self.max_reachable_generation = 0
        self._on_change = on_change
        self._reset_history_locked()

    def set_change_listener(self, on_change: Callable[[], Awaitable[None]] | None) -> None:
        self._on_change = on_change

    async def _changed(self) -> None:
        if self._on_change is not None:
            await self._on_change()

    # -------------------------------------------------------------- history
    # The _locked suffix means "caller already holds self.lock".

    def _reset_history_locked(self) -> None:
        self.history_generations.clear()
        self.history_snapshots.clear()
        self.max_reachable_generation = self.world.generation
        self._record_snapshot_locked(force=True)

    def _record_snapshot_locked(self, force: bool = False) -> bool:
        gen = self.world.generation
        if not force and (gen % self.snapshot_interval != 0):
            return False

        if gen not in self.history_snapshots:
            bisect.insort(self.history_generations, gen)
        self.history_snapshots[gen] = pack_grid(self.world.grid)
        if gen > self.max_reachable_generation:
            self.max_reachable_generation = gen

        while len(self.history_generations) > self.snapshot_cap:
            drop_gen = self.history_generations.pop(0)
            if drop_gen == gen and self.history_generations:
                drop_gen = self.history_generations.pop(0)
            self.history_snapshots.pop(drop_gen, None)
        return True

    def _truncate_future_locked(self) -> int:
        idx = bisect.bisect_right(self.history_generations, self.world.generation)
        to_drop = self.history_generations[idx:]
        if not to_drop:
            return 0
        del self.history_generations[idx:]
        for gen in to_drop:
            self.history_snapshots.pop(gen, None)
        self.max_reachable_generation = self.world.generation
        return len(to_drop)

    def _timeline_bounds_locked(self) -> tuple[int, int]:
        if not self.history_generations:
            return self.world.generation, self.world.generation
        return self.history_generations[0], self.history_generations[-1]

    def _reconstruct_generation_locked(self, target_generation: int) -> np.ndarray:
        idx = bisect.bisect_right(self.history_generations, target_generation) - 1
        if idx < 0:
            raise ValueError("target generation is older than retained history")

        start_gen = self.history_generations[idx]
        g = unpack_grid(self.history_snapshots[start_gen], self.world.width, self.world.height)
        for _ in range(target_generation - start_gen):
            g = next_grid_of(g, self.world.edge)
        return g

    # ------------------------------------------------------------ observing

    def _state_snapshot_locked(self) -> StateSnapshot:
        w = self.world
        min_gen, latest_snapshot = self._timeline_bounds_locked()
        return StateSnapshot(
            width=w.width,
            height=w.height,
            edge=w.edge,
            generation=w.generation,
            population=w.population(),
            running=self.running,
            fps=self.fps,
            timeline_min_generation=min_gen,
            timeline_latest_snapshot=latest_snapshot,
            timeline_latest_generation=self.max_reachable_generation,
            timeline_snapshot_interval=self.snapshot_interval,
            timeline_snapshot_count=len(self.history_generations),
            packed_cells=pack_grid(w.grid),
        )

    def state_snapshot(self) -> StateSnapshot:
        """Synchronous on purpose: broadcasts happen from inside operations
        that already hold the lock, and re-acquiring it would deadlock."""
        return self._state_snapshot_locked()

    async def status(self) -> WorldStatus:
        async with self.lock:
            w = self.world
            return WorldStatus(
                width=w.width, height=w.height, edge=w.edge,
                generation=w.generation, population=w.population(),
                bbox=w.bounding_box(), dynamics=w.dynamics(),
                running=self.running, fps=self.fps,
            )

    async def timeline_status(self) -> TimelineStatus:
        async with self.lock:
            min_gen, latest_snapshot = self._timeline_bounds_locked()
            return TimelineStatus(
                current_generation=self.world.generation,
                earliest_snapshot=min_gen,
                latest_snapshot=latest_snapshot,
                max_reachable_generation=self.max_reachable_generation,
                snapshot_interval=self.snapshot_interval,
                snapshot_count=len(self.history_generations),
                snapshot_cap=self.snapshot_cap,
            )

    async def observe(
        self, x: int = 0, y: int = 0, width: int = 0, height: int = 0
    ) -> ObserveResult:
        async with self.lock:
            w = self.world
            vw = width if width > 0 else w.width
            vh = height if height > 0 else w.height
            if vw * vh > MAX_OBSERVE_CELLS:
                raise ValueError(
                    f"requested window is {vw}x{vh} = {vw * vh} cells; max is "
                    f"{MAX_OBSERVE_CELLS}. Pass a smaller width/height window."
                )
            return ObserveResult(text=w.ascii_view(x, y, vw, vh), x=x, y=y,
                                 width=vw, height=vh)

    async def preview_generation(self, target_generation: int) -> GenerationPreview | None:
        """Reconstruct an earlier generation without committing to it.
        Returns None if the generation is outside the retained timeline."""
        async with self.lock:
            if target_generation < 0 or target_generation > self.max_reachable_generation:
                return None
            g = self._reconstruct_generation_locked(target_generation)
            return GenerationPreview(
                generation=target_generation,
                population=int(g.sum()),
                packed_cells=pack_grid(g),
            )

    # ------------------------------------------------------------- mutating

    async def create_world(
        self, width: int, height: int, edge: str = "wrap", random_fill: float = 0.0
    ) -> CreateResult:
        # `edge` is deliberately left to World to validate, so the rule lives
        # in one place and both callers get its fuller message.
        if not 0.0 <= random_fill <= 1.0:
            raise ValueError("random_fill must be between 0 and 1")
        async with self.lock:
            self.world = World(width, height, edge=edge, random_fill=random_fill)
            self._reset_history_locked()
            result = CreateResult(
                width=self.world.width, height=self.world.height, edge=self.world.edge,
                random_fill=random_fill, population=self.world.population(),
            )
        await self._changed()
        return result

    async def clear(self) -> ClearResult:
        async with self.lock:
            self.world.clear()
            self._reset_history_locked()
            result = ClearResult(width=self.world.width, height=self.world.height)
        await self._changed()
        return result

    async def set_cells(
        self, alive: Coords | None = None, dead: Coords | None = None
    ) -> SetCellsResult:
        alive = alive or []
        dead = dead or []
        if not alive and not dead:
            raise ValueError("pass at least one of alive/dead as a list of [x, y] pairs")
        async with self.lock:
            self._truncate_future_locked()
            n_alive, skip_a = self.world.set_cells(alive, 1)
            n_dead, skip_d = self.world.set_cells(dead, 0)
            self._record_snapshot_locked(force=True)
            result = SetCellsResult(
                set_alive=n_alive, set_dead=n_dead, skipped=skip_a + skip_d,
                population=self.world.population(),
            )
        await self._changed()
        return result

    async def place_pattern(
        self, rle: str, x: int, y: int, clear_rect: bool = False
    ) -> PlacementResult:
        # Parsed outside the lock: a malformed pattern must not touch the world.
        cells, pw, ph = parse_rle(rle)
        async with self.lock:
            self._truncate_future_locked()
            w = self.world
            if clear_rect:
                w.set_cells(
                    [(x + dx, y + dy) for dy in range(ph) for dx in range(pw)], 0
                )
            placed, skipped = w.set_cells([(x + dx, y + dy) for dx, dy in cells], 1)
            self._record_snapshot_locked(force=True)
            result = PlacementResult(
                pattern_width=pw, pattern_height=ph, x=x, y=y,
                placed_cells=placed, skipped=skipped,
                population=w.population(), generation=w.generation,
                bbox=w.bounding_box(),
            )
        await self._changed()
        return result

    async def advance(self, generations: int = 1) -> AdvanceResult:
        """Step 1..MAX_ADVANCE generations, broadcasting each one so the
        browser animates the run."""
        n = int(generations)
        if not 1 <= n <= MAX_ADVANCE:
            raise ValueError(f"generations must be between 1 and {MAX_ADVANCE}")

        async with self.lock:
            self._truncate_future_locked()
            pop_before = self.world.population()

        for i in range(n):
            async with self.lock:
                self.world.step(1)
                self._record_snapshot_locked(force=(i == n - 1))
            await self._changed()
            if n > 1:
                await asyncio.sleep(0.01)  # let viewers see the animation

        async with self.lock:
            return AdvanceResult(
                generations=n, generation=self.world.generation,
                population_before=pop_before, population=self.world.population(),
                dynamics=self.world.dynamics(),
            )

    async def rewind(self, target_generation: int) -> RewindResult:
        target = int(target_generation)
        await self.set_autorun(False)
        async with self.lock:
            current = self.world.generation
            if target < 0:
                raise ValueError("generation must be >= 0")
            if target > self.max_reachable_generation:
                raise ValueError(
                    f"generation {target} is outside retained history "
                    f"(max reachable is {self.max_reachable_generation})"
                )

            self.world.grid = self._reconstruct_generation_locked(target)
            self.world.generation = target
            # Committing a rewind establishes a new present on a linear
            # timeline; anything ahead of the target is discarded immediately.
            dropped = self._truncate_future_locked()
            self._record_snapshot_locked(force=True)
            result = RewindResult(
                from_generation=current, to_generation=target,
                population=self.world.population(), dropped_snapshots=dropped,
            )
        await self._changed()
        return result

    # -------------------------------------------------------------- autorun

    async def _run_loop(self) -> None:
        try:
            while self.running:
                t0 = time.monotonic()
                async with self.lock:
                    self._truncate_future_locked()
                    self.world.step(1)
                    self._record_snapshot_locked()
                await self._changed()
                delay = 1.0 / self.fps - (time.monotonic() - t0)
                await asyncio.sleep(delay if delay > 0 else 0.001)
        finally:
            self.running = False
            await self._changed()

    async def set_autorun(self, enabled: bool, fps: float | None = None) -> AutorunResult:
        if fps is not None:
            self.fps = min(MAX_FPS, max(MIN_FPS, float(fps)))
        if enabled and not self.running:
            self.running = True
            self.run_task = asyncio.create_task(self._run_loop())
        elif not enabled:
            self.running = False  # the loop exits on its own and broadcasts
        await self._changed()
        async with self.lock:
            return AutorunResult(
                running=self.running, fps=self.fps,
                generation=self.world.generation, population=self.world.population(),
            )

    async def shutdown(self) -> None:
        self.running = False
        if self.run_task is not None:
            self.run_task.cancel()
            self.run_task = None

    # ------------------------------------------- multi-generation simulation

    def _sim_setup_locked(self) -> tuple[np.ndarray, str, int, int]:
        w = self.world
        return w.grid.copy(), w.edge, w.generation, w.width * w.height

    async def advance_generations(self, count: int, sample_every: int = 1) -> SimulationRun:
        """Simulate and PERMANENTLY commit `count` generations, broadcasting
        once per sample so the browser keeps animating."""
        async with self.lock:
            self._truncate_future_locked()
            start_grid, edge, start_gen, cell_count = self._sim_setup_locked()

        try:
            validate_sim_params(count, sample_every, cell_count)
        except SimulationError as e:
            return SimulationRun(error=e.to_dict())

        samples: list[dict] = []
        it = simulate(start_grid, edge, count, sample_every, start_gen)
        summary = None
        while True:
            try:
                sample, g = next(it)
            except StopIteration as stop:
                summary = stop.value["summary"]
                break
            samples.append(sample)
            async with self.lock:
                # .copy() is load-bearing: simulate() yields its own live
                # working array, so assigning it directly would alias the
                # world's grid onto simulator internals. World.set_cells/clear
                # write in place, so a browser paint or Clear mid-run would
                # reach into the running simulation and silently corrupt every
                # generation after it — and the summary reported to the agent.
                self.world.grid = g.copy()
                self.world.generation = sample["generation"]
                self._record_snapshot_locked()
            await self._changed()
            await asyncio.sleep(0)

        async with self.lock:
            self._record_snapshot_locked(force=True)

        return SimulationRun(samples=samples, summary=summary, committed=True)

    async def preview_generations(self, count: int, sample_every: int = 1) -> SimulationRun:
        """Same simulation as advance_generations, but nothing is committed and
        no state is broadcast."""
        async with self.lock:
            start_grid, edge, start_gen, cell_count = self._sim_setup_locked()

        try:
            validate_sim_params(count, sample_every, cell_count)
        except SimulationError as e:
            return SimulationRun(error=e.to_dict())

        samples: list[dict] = []
        it = simulate(start_grid, edge, count, sample_every, start_gen)
        summary = None
        n = 0
        while True:
            try:
                sample, _g = next(it)
            except StopIteration as stop:
                summary = stop.value["summary"]
                break
            samples.append(sample)
            n += 1
            if n % 25 == 0:
                await asyncio.sleep(0)  # cooperative yield; the world is untouched

        return SimulationRun(samples=samples, summary=summary, committed=False)
