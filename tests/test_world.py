"""World: construction, cell editing, stepping, dynamics, ASCII rendering."""
from __future__ import annotations

import numpy as np
import pytest

from gol_world import MAX_DIM, MIN_DIM, World, bounding_box_of, next_grid_of


# ------------------------------------------------------------- construction

def test_defaults_to_empty_160x100_wrap_world():
    w = World()
    assert (w.width, w.height, w.edge) == (160, 100, "wrap")
    assert w.generation == 0
    assert w.population() == 0


@pytest.mark.parametrize("width,height", [
    (MIN_DIM - 1, 64), (64, MIN_DIM - 1), (MAX_DIM + 1, 64), (64, MAX_DIM + 1),
])
def test_rejects_out_of_range_dimensions(width, height):
    with pytest.raises(ValueError, match="between"):
        World(width, height)


def test_accepts_the_exact_dimension_bounds():
    assert World(MIN_DIM, MIN_DIM).population() == 0
    assert World(MAX_DIM, MIN_DIM).width == MAX_DIM


def test_rejects_unknown_edge_mode():
    with pytest.raises(ValueError, match="wrap"):
        World(32, 32, edge="reflect")


def test_random_fill_populates_roughly_the_requested_density():
    w = World(200, 200, random_fill=0.5)
    assert 0.45 < w.population() / (200 * 200) < 0.55


def test_random_fill_zero_leaves_the_board_empty():
    assert World(64, 64, random_fill=0.0).population() == 0


# ------------------------------------------------------------------ editing

def test_set_cells_applies_and_counts():
    w = World(32, 32)
    applied, skipped = w.set_cells([(1, 1), (2, 2), (3, 3)], 1)
    assert (applied, skipped) == (3, [])
    assert w.population() == 3
    assert w.grid[1, 1] == 1


def test_set_cells_uses_x_y_order_not_row_column():
    w = World(32, 16)
    w.set_cells([(5, 2)], 1)
    assert w.grid[2, 5] == 1          # grid is indexed [y, x]
    assert w.grid[5, 2] == 0


def test_wrap_world_wraps_out_of_range_coordinates():
    w = World(10, 10, edge="wrap")
    applied, skipped = w.set_cells([(12, 3), (-1, -1)], 1)
    assert (applied, skipped) == (2, [])
    assert w.grid[3, 2] == 1          # x=12 -> 2
    assert w.grid[9, 9] == 1          # (-1,-1) -> (9,9)


def test_dead_edge_world_skips_and_reports_out_of_range_coordinates():
    w = World(10, 10, edge="dead")
    applied, skipped = w.set_cells([(5, 5), (12, 3), (-1, 0)], 1)
    assert applied == 1
    assert skipped == [(12, 3), (-1, 0)]
    assert w.population() == 1


def test_set_cells_can_kill_cells():
    w = World(16, 16)
    w.set_cells([(1, 1), (2, 2)], 1)
    applied, _ = w.set_cells([(1, 1)], 0)
    assert applied == 1
    assert w.population() == 1


def test_clear_empties_the_board_and_resets_the_generation():
    w = World(16, 16, random_fill=1.0)
    w.step(3)
    w.clear()
    assert w.population() == 0
    assert w.generation == 0


def test_clear_keeps_dimensions_and_edge():
    w = World(24, 12, edge="dead")
    w.clear()
    assert (w.width, w.height, w.edge) == (24, 12, "dead")


# --------------------------------------------------------------- simulation

def test_blinker_oscillates_with_period_two():
    w = World(16, 16)
    w.set_cells([(5, 5), (6, 5), (7, 5)], 1)
    before = w.grid.copy()
    w.step(1)
    assert not np.array_equal(w.grid, before)   # horizontal -> vertical
    w.step(1)
    assert np.array_equal(w.grid, before)


def test_block_is_a_still_life():
    w = World(16, 16)
    w.set_cells([(4, 4), (5, 4), (4, 5), (5, 5)], 1)
    before = w.grid.copy()
    w.step(5)
    assert np.array_equal(w.grid, before)


def test_glider_translates_by_one_cell_diagonally_every_four_generations():
    w = World(32, 32)
    w.set_cells([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)], 1)
    w.step(4)
    assert w.population() == 5
    assert bounding_box_of(w.grid) == (1, 1, 3, 3)   # moved +1,+1


def test_step_advances_the_generation_counter():
    w = World(16, 16)
    w.step(7)
    assert w.generation == 7


def test_lone_cell_dies_of_underpopulation():
    w = World(16, 16)
    w.set_cells([(8, 8)], 1)
    w.step(1)
    assert w.population() == 0


def test_wrap_edge_lets_a_pattern_interact_across_the_border():
    # blinker straddling the left/right border: only oscillates if it wraps
    w = World(10, 10, edge="wrap")
    w.set_cells([(9, 5), (0, 5), (1, 5)], 1)
    w.step(2)
    assert w.population() == 3
    assert w.grid[5, 9] == 1 and w.grid[5, 0] == 1 and w.grid[5, 1] == 1


def test_dead_edge_does_not_wrap_neighbors_across_the_border():
    wrap = World(10, 10, edge="wrap")
    dead = World(10, 10, edge="dead")
    cells = [(9, 5), (0, 5), (1, 5)]
    wrap.set_cells(cells, 1)
    dead.set_cells(cells, 1)
    wrap.step(1)
    dead.step(1)
    assert not np.array_equal(wrap.grid, dead.grid)


def test_next_grid_does_not_mutate_the_world():
    w = World(16, 16, random_fill=0.4)
    before = w.grid.copy()
    w.next_grid()
    assert np.array_equal(w.grid, before)
    assert w.generation == 0


def test_next_grid_of_does_not_mutate_its_input():
    g = World(16, 16, random_fill=0.4).grid
    before = g.copy()
    next_grid_of(g, "wrap")
    assert np.array_equal(g, before)


# ----------------------------------------------------------------- dynamics

def test_dynamics_reports_empty_for_a_dead_board():
    assert World(16, 16).dynamics() == "empty"


def test_dynamics_reports_static_for_a_still_life():
    w = World(16, 16)
    w.set_cells([(4, 4), (5, 4), (4, 5), (5, 5)], 1)
    assert w.dynamics() == "static"


def test_dynamics_reports_period_2_for_a_blinker():
    w = World(16, 16)
    w.set_cells([(5, 5), (6, 5), (7, 5)], 1)
    assert w.dynamics() == "period-2"


def test_dynamics_reports_evolving_for_a_glider():
    w = World(32, 32)
    w.set_cells([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)], 1)
    assert w.dynamics() == "evolving"


def test_dynamics_does_not_advance_the_world():
    w = World(32, 32, random_fill=0.3)
    before, gen = w.grid.copy(), w.generation
    w.dynamics()
    assert np.array_equal(w.grid, before) and w.generation == gen


# ------------------------------------------------------------ bounding box

def test_bounding_box_is_none_for_an_empty_board():
    assert World(16, 16).bounding_box() is None


def test_bounding_box_is_inclusive_on_both_corners():
    w = World(32, 32)
    w.set_cells([(3, 7), (10, 20)], 1)
    assert w.bounding_box() == (3, 7, 10, 20)


def test_bounding_box_of_a_single_cell_is_degenerate():
    w = World(16, 16)
    w.set_cells([(4, 9)], 1)
    assert w.bounding_box() == (4, 9, 4, 9)


# ------------------------------------------------------------ ascii_view

def test_ascii_view_marks_alive_and_dead_cells():
    w = World(16, 16)
    w.set_cells([(0, 0), (2, 0)], 1)
    rows = w.ascii_view(0, 0, 4, 2).splitlines()
    assert rows[3].endswith("#.#.")
    assert rows[4].endswith("....")


def test_ascii_view_header_reports_the_window_and_totals():
    w = World(16, 16)
    w.set_cells([(1, 1)], 1)
    header = w.ascii_view(0, 0, 8, 8).splitlines()[0]
    assert "x=[0..7] y=[0..7]" in header
    assert "16x16" in header
    assert "pop in view 1 / total 1" in header


def test_ascii_view_labels_rows_with_absolute_y_coordinates():
    w = World(32, 32)
    rows = w.ascii_view(5, 10, 4, 3).splitlines()[3:]
    assert [r.split()[0] for r in rows] == ["10", "11", "12"]


def test_ascii_view_clips_a_window_that_overruns_the_board():
    w = World(10, 10)
    body = w.ascii_view(8, 8, 50, 50).splitlines()[3:]
    assert len(body) == 2                       # rows 8..9 only
    assert all(len(r.split()[1]) == 2 for r in body)


def test_ascii_view_rejects_a_window_entirely_off_the_board():
    w = World(10, 10)
    with pytest.raises(ValueError, match="empty after clipping"):
        w.ascii_view(50, 50, 4, 4)
