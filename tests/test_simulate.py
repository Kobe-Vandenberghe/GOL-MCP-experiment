"""The simulate() generator, its summary, and parameter validation.

These pin the contract that advance_generations and preview_generations both
depend on — including the copy semantics that the server's commit loop relies
on (see test_concurrency.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from gol_world import (
    MAX_SIM_GENERATIONS,
    MAX_SIM_SAMPLES,
    SimulationError,
    World,
    next_grid_of,
    shape_hash,
    simulate,
    state_hash,
    validate_sim_params,
)

from conftest import seeded_grid


def run(g, edge="wrap", count=10, sample_every=1, start=0):
    """Drain the generator: returns (samples, summary)."""
    samples = []
    it = simulate(g, edge, count, sample_every, start)
    while True:
        try:
            sample, _grid = next(it)
        except StopIteration as stop:
            return samples, stop.value["summary"]
        samples.append(sample)


def blinker(size=16):
    w = World(size, size)
    w.set_cells([(5, 5), (6, 5), (7, 5)], 1)
    return w.grid


def block(size=16):
    w = World(size, size)
    w.set_cells([(4, 4), (5, 4), (4, 5), (5, 5)], 1)
    return w.grid


def glider(size=32):
    w = World(size, size)
    w.set_cells([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)], 1)
    return w.grid


# ---------------------------------------------------------- copy semantics

def test_does_not_mutate_the_callers_array():
    g = seeded_grid(32, 32, 0.3, seed=5)
    before = g.copy()
    run(g, count=25)
    assert np.array_equal(g, before)


def test_yields_its_own_live_working_array_not_a_copy():
    """Documents the sharp edge the server has to defend against: the yielded
    grid IS the generator's internal state, so an external in-place write
    leaks into subsequent generations."""
    g = seeded_grid(32, 32, 0.3, seed=3)
    it = simulate(g, "wrap", 10, 1, 0)
    _, gen1 = next(it)
    clean_gen1 = gen1.copy()

    for y, x in np.argwhere(clean_gen1 == 0)[:5]:
        gen1[y, x] = 1                       # in-place, as World.set_cells does

    _, gen2 = next(it)
    assert np.array_equal(gen2, next_grid_of(gen1, "wrap"))
    assert not np.array_equal(gen2, next_grid_of(clean_gen1, "wrap"))


# ------------------------------------------------------------ sample cadence

def test_samples_every_generation_by_default():
    samples, _ = run(blinker(), count=6, sample_every=1)
    assert [s["generation"] for s in samples] == [1, 2, 3, 4, 5, 6]


def test_sample_every_thins_the_samples():
    samples, _ = run(blinker(), count=10, sample_every=5)
    assert [s["generation"] for s in samples] == [5, 10]


def test_always_includes_a_final_sample_off_the_sample_grid():
    samples, _ = run(blinker(), count=7, sample_every=5)
    assert [s["generation"] for s in samples] == [5, 7]


def test_generations_are_offset_by_start_generation():
    samples, summary = run(blinker(), count=3, start=100)
    assert [s["generation"] for s in samples] == [101, 102, 103]
    assert summary["start_generation"] == 100
    assert summary["end_generation"] == 103


def test_summary_sample_count_matches_the_samples_returned():
    samples, summary = run(glider(), count=20, sample_every=3)
    assert summary["sample_count"] == len(samples)


# ------------------------------------------------------------------ metrics

def test_births_and_deaths_accumulate_between_samples():
    samples, _ = run(blinker(), count=4, sample_every=2)
    # a blinker flips 2 cells off and 2 on each generation
    assert samples[0]["births"] == 4 and samples[0]["deaths"] == 4


def test_births_and_deaths_reset_after_each_sample():
    samples, _ = run(blinker(), count=4, sample_every=1)
    assert all(s["births"] == 2 and s["deaths"] == 2 for s in samples)


def test_population_tracks_the_live_cell_count():
    samples, _ = run(block(), count=3)
    assert all(s["population"] == 4 for s in samples)


def test_bbox_is_none_once_the_board_is_empty():
    w = World(16, 16)
    w.set_cells([(8, 8)], 1)           # lone cell dies immediately
    samples, _ = run(w.grid, count=2)
    assert samples[0]["bbox"] is None


def test_bbox_tracks_a_moving_pattern():
    samples, _ = run(glider(), count=4)
    assert samples[-1]["bbox"] == [1, 1, 3, 3]


def test_min_and_max_population_span_the_whole_run():
    _, summary = run(glider(), count=12)
    assert summary["min_population"] == summary["max_population"] == 5


# ------------------------------------------------------------------- hashes

def test_state_hash_is_stable_for_identical_boards():
    assert state_hash(block()) == state_hash(block())


def test_state_hash_changes_when_a_pattern_moves():
    samples, _ = run(glider(), count=4)
    assert samples[-1]["state_hash"] != state_hash(glider())


def test_shape_hash_is_translation_independent():
    samples, _ = run(glider(), count=4)
    assert samples[-1]["shape_hash"] == shape_hash(glider())


def test_shape_hash_is_empty_for_a_dead_board():
    assert shape_hash(np.zeros((8, 8), dtype=np.uint8)) == "empty"


def test_shape_hash_distinguishes_different_shapes():
    assert shape_hash(block()) != shape_hash(blinker())


# --------------------------------------------------- extinction & stability

def test_detects_extinction_and_reports_the_generation():
    w = World(16, 16)
    w.set_cells([(8, 8)], 1)
    _, summary = run(w.grid, count=10)
    assert summary["extinct"] is True
    assert summary["extinct_at_generation"] == 1
    assert summary["final_population"] == 0


def test_detects_a_static_board_and_reports_the_generation():
    _, summary = run(block(), count=10)
    assert summary["stable"] is True
    assert summary["stable_at_generation"] == 1


def test_keeps_emitting_samples_after_freezing():
    samples, _ = run(block(), count=6, sample_every=2)
    assert [s["generation"] for s in samples] == [2, 4, 6]
    assert all(s["population"] == 4 for s in samples)


def test_frozen_run_reports_no_further_births_or_deaths():
    samples, _ = run(block(), count=6, sample_every=1)
    assert all(s["births"] == 0 and s["deaths"] == 0 for s in samples)


def test_a_glider_never_goes_extinct_or_static():
    _, summary = run(glider(), count=40)
    assert summary["extinct"] is False and summary["stable"] is False


# ------------------------------------------------------------- periodicity

def test_detects_a_period_2_oscillator():
    _, summary = run(blinker(), count=10)
    assert summary["repeating_period"] == 2


def test_a_still_life_reports_period_1():
    _, summary = run(block(), count=10)
    assert summary["repeating_period"] == 1


def test_a_moving_glider_reports_no_period():
    # its shape repeats, but its absolute position never does
    _, summary = run(glider(size=64), count=40)
    assert summary["repeating_period"] is None


# -------------------------------------------------------------- validation

def test_accepts_valid_parameters():
    validate_sim_params(100, 1, 160 * 100)


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_a_non_positive_count(count):
    with pytest.raises(SimulationError) as e:
        validate_sim_params(count, 1, 1000)
    assert e.value.code == "invalid_count"


def test_rejects_a_boolean_count():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(True, 1, 1000)
    assert e.value.code == "invalid_count"


def test_rejects_a_count_over_the_flat_cap():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(MAX_SIM_GENERATIONS + 1, 100, 64)
    assert e.value.code == "count_too_large"


def test_rejects_a_count_too_large_for_the_board_and_says_what_fits():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(20000, 100, 1024 * 1024)
    assert e.value.code == "count_too_large_for_board"
    limit = e.value.detail["max_count_for_board"]
    validate_sim_params(limit, limit, 1024 * 1024)      # the advertised limit works


def test_rejects_a_non_positive_sample_every():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(10, 0, 1000)
    assert e.value.code == "invalid_sample_every"


def test_rejects_sample_every_larger_than_count():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(10, 11, 1000)
    assert e.value.code == "sample_every_too_large"


def test_rejects_too_many_samples_and_says_the_minimum_that_fits():
    with pytest.raises(SimulationError) as e:
        validate_sim_params(MAX_SIM_SAMPLES * 2, 1, 64)
    assert e.value.code == "too_many_samples"
    validate_sim_params(MAX_SIM_SAMPLES * 2, e.value.detail["min_sample_every"], 64)


def test_error_serializes_to_a_dict_with_code_and_message():
    try:
        validate_sim_params(0, 1, 1000)
    except SimulationError as e:
        d = e.to_dict()
    assert d["error"] == "invalid_count"
    assert isinstance(d["message"], str) and d["message"]
    assert d["count"] == 0
