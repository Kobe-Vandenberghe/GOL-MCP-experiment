"""Conway's Game of Life world: NumPy board, RLE parsing, ASCII rendering.

Coordinate convention everywhere: (x, y) with (0, 0) at the top-left,
x growing rightward (column), y growing downward (row). The grid array
is indexed grid[y, x].
"""
from __future__ import annotations

import hashlib
import re
from collections import deque

import numpy as np

MIN_DIM = 8
MAX_DIM = 1024

# Caps for advance_generations / preview_generations (see "multi-generation
# simulation" section below). The real cost of a run is board_area * count,
# not count alone, so MAX_SIM_CELL_STEPS is the primary limit - it scales
# generation headroom down for huge boards and up for small ones, keeping
# worst-case wall time (a chaotic board that never goes extinct/static) to a
# few seconds regardless of board size. MAX_SIM_GENERATIONS is just a flat
# backstop so a tiny board can't be asked for an absurd generation count.
MAX_SIM_GENERATIONS = 20000
MAX_SIM_CELL_STEPS = 300_000_000
MAX_SIM_SAMPLES = 500
PERIOD_DETECT_WINDOW = 64  # how far back to look for an exact repeated state


def _neighbor_counts_of(g: np.ndarray, edge: str) -> np.ndarray:
    if edge == "wrap":
        n = np.zeros_like(g)
        # Separable sum: 4 rolls (2 horizontal + 2 vertical) instead of the
        # naive 8, then subtract the center cell back out. Cuts wall time on
        # the biggest boards by roughly a third - worth it since this runs
        # in a tight loop for advance_generations/preview_generations.
        colsum = np.roll(g, -1, axis=1) + g + np.roll(g, 1, axis=1)
        return np.roll(colsum, -1, axis=0) + np.roll(colsum, 1, axis=0) + colsum - g
    p = np.pad(g, 1)
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )


def next_grid_of(g: np.ndarray, edge: str) -> np.ndarray:
    n = _neighbor_counts_of(g, edge)
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def bounding_box_of(g: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) inclusive box around live cells, or None if empty."""
    ys, xs = np.nonzero(g)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class World:
    def __init__(
        self,
        width: int = 160,
        height: int = 100,
        edge: str = "wrap",
        random_fill: float = 0.0,
    ):
        width, height = int(width), int(height)
        if not (MIN_DIM <= width <= MAX_DIM and MIN_DIM <= height <= MAX_DIM):
            raise ValueError(
                f"width and height must be between {MIN_DIM} and {MAX_DIM} (got {width}x{height})"
            )
        if edge not in ("wrap", "dead"):
            raise ValueError('edge must be "wrap" (toroidal) or "dead" (border cells stay dead)')
        self.width = width
        self.height = height
        self.edge = edge
        self.generation = 0
        if random_fill > 0:
            rng = np.random.default_rng()
            self.grid = (rng.random((height, width)) < min(1.0, random_fill)).astype(np.uint8)
        else:
            self.grid = np.zeros((height, width), dtype=np.uint8)

    # ---------------------------------------------------------------- basics

    def population(self) -> int:
        return int(self.grid.sum())

    def clear(self) -> None:
        self.grid[:] = 0
        self.generation = 0

    def bounding_box(self) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) inclusive box around live cells, or None if empty."""
        return bounding_box_of(self.grid)

    # ------------------------------------------------------------ simulation

    def next_grid(self, g: np.ndarray | None = None) -> np.ndarray:
        return next_grid_of(self.grid if g is None else g, self.edge)

    def step(self, generations: int = 1) -> None:
        for _ in range(generations):
            self.grid = self.next_grid()
            self.generation += 1

    def dynamics(self) -> str:
        """'empty' | 'static' | 'period-2' | 'evolving' (peeks ahead, doesn't mutate)."""
        if self.population() == 0:
            return "empty"
        g1 = self.next_grid()
        if np.array_equal(g1, self.grid):
            return "static"
        if np.array_equal(self.next_grid(g1), self.grid):
            return "period-2"
        return "evolving"

    # --------------------------------------------------------------- editing

    def set_cells(self, coords, value: int) -> tuple[int, list[tuple[int, int]]]:
        """Set cells to value (0/1). Wrap out-of-bounds coords on toroidal
        worlds, skip them on dead-edge worlds. Returns (applied, skipped)."""
        applied, skipped = 0, []
        v = 1 if value else 0
        for pair in coords:
            x, y = int(pair[0]), int(pair[1])
            if self.edge == "wrap":
                x %= self.width
                y %= self.height
            elif not (0 <= x < self.width and 0 <= y < self.height):
                skipped.append((x, y))
                continue
            self.grid[y, x] = v
            applied += 1
        return applied, skipped

    # ------------------------------------------------------------------ view

    def ascii_view(self, x: int, y: int, w: int, h: int) -> str:
        """ASCII window with an x-axis ruler and y row labels (absolute coords).
        '#' = alive, '.' = dead."""
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1 = min(self.width, x0 + int(w))
        y1 = min(self.height, y0 + int(h))
        if x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"view window is empty after clipping to the {self.width}x{self.height} board"
            )
        sub = self.grid[y0:y1, x0:x1]
        gutter = len(str(y1 - 1))
        pad = " " * (gutter + 1)
        tens = pad + "".join(str(((x0 + i) // 10) % 10) for i in range(x1 - x0))
        units = pad + "".join(str((x0 + i) % 10) for i in range(x1 - x0))
        rows = [
            f"{y0 + j:>{gutter}} " + "".join("#" if c else "." for c in sub[j])
            for j in range(y1 - y0)
        ]
        header = (
            f"view x=[{x0}..{x1 - 1}] y=[{y0}..{y1 - 1}] of {self.width}x{self.height} board | "
            f"gen {self.generation} | pop in view {int(sub.sum())} / total {self.population()} | "
            "'#'=alive '.'=dead | (0,0) is top-left, x right, y down"
        )
        return "\n".join([header, tens, units, *rows])


# ---------------------------------------------- multi-generation simulation


class SimulationError(ValueError):
    """Invalid advance_generations/preview_generations parameters. Carries a
    machine-readable `code` and extra detail so callers can return a
    structured error instead of a bare message string."""

    def __init__(self, code: str, message: str, **detail):
        super().__init__(message)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict:
        return {"error": self.code, "message": str(self), **self.detail}


def state_hash(g: np.ndarray) -> str:
    """Exact-state fingerprint: identical only for bit-for-bit identical
    boards (same live cells at the same positions)."""
    return hashlib.blake2b(np.packbits(g, axis=None).tobytes(), digest_size=8).hexdigest()


def shape_hash(g: np.ndarray) -> str:
    """Translation-independent fingerprint: cropped to the live-cell
    bounding box, so a pattern that has merely moved (e.g. a glider gliding)
    hashes the same as it did before it moved. "empty" for an all-dead grid."""
    box = bounding_box_of(g)
    if box is None:
        return "empty"
    x0, y0, x1, y1 = box
    crop = g[y0:y1 + 1, x0:x1 + 1]
    h = hashlib.blake2b(digest_size=8)
    h.update(crop.shape[0].to_bytes(4, "little"))
    h.update(crop.shape[1].to_bytes(4, "little"))
    h.update(np.packbits(crop, axis=None).tobytes())
    return h.hexdigest()


def validate_sim_params(count: int, sample_every: int, cell_count: int) -> None:
    """Raise SimulationError for anything advance_generations/
    preview_generations can't handle. `cell_count` is the board's
    width*height, used to size the cell-steps budget (see
    MAX_SIM_CELL_STEPS above) - a big board gets a lower generation cap than
    a small one, so a single call stays fast either way. Callers catch this
    and return err.to_dict() as the tool's (non-exceptional) JSON result."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise SimulationError("invalid_count", "count must be a positive integer", count=count)
    if count > MAX_SIM_GENERATIONS:
        raise SimulationError(
            "count_too_large",
            f"count must be between 1 and {MAX_SIM_GENERATIONS}",
            count=count, max_count=MAX_SIM_GENERATIONS,
        )
    if count * cell_count > MAX_SIM_CELL_STEPS:
        max_count_for_board = max(1, MAX_SIM_CELL_STEPS // cell_count)
        raise SimulationError(
            "count_too_large_for_board",
            f"count={count} on this {cell_count}-cell board would do {count * cell_count:,} "
            f"cell-generations of work (max {MAX_SIM_CELL_STEPS:,}). "
            f"Use count <= {max_count_for_board} on a board this size.",
            count=count, cell_count=cell_count, max_cell_steps=MAX_SIM_CELL_STEPS,
            max_count_for_board=max_count_for_board,
        )
    if not isinstance(sample_every, int) or isinstance(sample_every, bool) or sample_every < 1:
        raise SimulationError(
            "invalid_sample_every", "sample_every must be a positive integer", sample_every=sample_every
        )
    if sample_every > count:
        raise SimulationError(
            "sample_every_too_large",
            "sample_every cannot exceed count",
            sample_every=sample_every, count=count,
        )
    n_samples = count // sample_every + (1 if count % sample_every else 0)
    if n_samples > MAX_SIM_SAMPLES:
        min_sample_every = -(-count // MAX_SIM_SAMPLES)  # ceil division
        raise SimulationError(
            "too_many_samples",
            f"count={count} with sample_every={sample_every} would return {n_samples} samples "
            f"(max {MAX_SIM_SAMPLES}). Use sample_every >= {min_sample_every}.",
            count=count, sample_every=sample_every, requested_samples=n_samples,
            max_samples=MAX_SIM_SAMPLES, min_sample_every=min_sample_every,
        )


def simulate(g: np.ndarray, edge: str, count: int, sample_every: int, start_generation: int = 0):
    """Advance a *copy* of `g` by `count` generations (never mutates the
    input array), yielding (sample_dict, grid_at_that_generation) once every
    `sample_every` generations - always including a final sample at exactly
    `start_generation + count`, even if that's not on the sample_every grid.

    Once the board goes extinct or reaches a static (unchanging) state,
    stops actually simulating - the state can't change further, so
    remaining generations are reported as unchanged without doing the work.
    A short exact repeat (within the last `PERIOD_DETECT_WINDOW` states) is
    reported as `repeating_period`, but - unlike the extinct/static case -
    is not fast-forwarded, since a moving pattern's shape can repeat via
    shape_hash without its absolute position ever repeating.

    Call next(gen) in a loop until StopIteration; its `.value` is
    {"summary": {...}}. Used identically by advance_generations (which
    commits each yielded grid to the live world) and preview_generations
    (which discards the grid and only keeps the stats) - guaranteeing they
    agree exactly, since both walk this same generator."""
    g = g.copy()
    pop = int(g.sum())
    min_pop = max_pop = pop

    recent: deque[tuple[str, int]] = deque(maxlen=PERIOD_DETECT_WINDOW)
    recent.append((state_hash(g), 0))

    births_acc = deaths_acc = 0
    extinct_at: int | None = None
    static_at: int | None = None
    repeating_period: int | None = None
    frozen = False
    step = 0
    n_samples = 0

    while step < count:
        if not frozen:
            new_g = next_grid_of(g, edge)
            births = int(np.count_nonzero((new_g == 1) & (g == 0)))
            deaths = int(np.count_nonzero((new_g == 0) & (g == 1)))
            births_acc += births
            deaths_acc += deaths
            g = new_g
            step += 1
            pop = int(g.sum())
            min_pop = min(min_pop, pop)
            max_pop = max(max_pop, pop)

            if pop == 0 and extinct_at is None:
                extinct_at = start_generation + step
                frozen = True
            elif births == 0 and deaths == 0 and static_at is None:
                static_at = start_generation + step
                frozen = True

            if repeating_period is None:
                h = state_hash(g)
                for prev_hash, prev_step in recent:
                    if prev_hash == h:
                        repeating_period = step - prev_step
                        break
                recent.append((h, step))
        else:
            step += 1  # frozen: nothing changes, just advance the counter

        if step % sample_every == 0 or step == count:
            box = bounding_box_of(g)
            n_samples += 1
            yield (
                {
                    "generation": start_generation + step,
                    "population": pop,
                    "births": births_acc,
                    "deaths": deaths_acc,
                    "bbox": list(box) if box else None,
                    "state_hash": state_hash(g),
                    "shape_hash": shape_hash(g),
                },
                g,
            )
            births_acc = deaths_acc = 0

    return {
        "summary": {
            "start_generation": start_generation,
            "end_generation": start_generation + count,
            "min_population": min_pop,
            "max_population": max_pop,
            "final_population": pop,
            "final_bbox": list(bounding_box_of(g)) if bounding_box_of(g) else None,
            "extinct": extinct_at is not None,
            "extinct_at_generation": extinct_at,
            "stable": static_at is not None,
            "stable_at_generation": static_at,
            "repeating_period": repeating_period,
            "sample_count": n_samples,
        }
    }


# -------------------------------------------------------------------- RLE

_HEADER_RE = re.compile(r"^\s*x\s*=", re.IGNORECASE)


def parse_rle(rle: str) -> tuple[list[tuple[int, int]], int, int]:
    """Parse a Game of Life RLE pattern into relative live-cell coordinates.

    Returns (cells, width, height) where cells are (dx, dy) offsets from the
    pattern's top-left. Lenient: header line and '!' terminator optional,
    '#' comment lines ignored, letters other than 'b' treated as alive.
    """
    body_parts = []
    for line in rle.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if _HEADER_RE.match(line):
            continue
        body_parts.append(line)
    body = "".join(body_parts)

    cells: list[tuple[int, int]] = []
    x = y = 0
    run = ""
    for ch in body:
        if ch.isdigit():
            run += ch
        elif ch in "bB.":
            x += int(run or 1)
            run = ""
        elif ch == "$":
            y += int(run or 1)
            x = 0
            run = ""
        elif ch == "!":
            break
        elif ch.isalpha() or ch in "oO*":
            count = int(run or 1)
            run = ""
            cells.extend((x + i, y) for i in range(count))
            x += count
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"unexpected character {ch!r} in RLE pattern")

    if not cells:
        raise ValueError("RLE pattern contains no live cells")
    w = max(dx for dx, _ in cells) + 1
    h = max(dy for _, dy in cells) + 1
    return cells, w, h
