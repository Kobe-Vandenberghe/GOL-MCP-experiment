"""RLE parsing. Deliberately lenient — these tests pin how lenient."""
from __future__ import annotations

import pytest

from gol_world import parse_rle


def test_parses_a_blinker():
    cells, w, h = parse_rle("3o!")
    assert sorted(cells) == [(0, 0), (1, 0), (2, 0)]
    assert (w, h) == (3, 1)


def test_parses_a_glider():
    cells, w, h = parse_rle("bob$2bo$3o!")
    assert sorted(cells) == [(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]
    assert (w, h) == (3, 3)


def test_returns_offsets_relative_to_the_pattern_origin():
    cells, _, _ = parse_rle("2bo!")
    assert cells == [(2, 0)]          # leading dead cells still shift x


def test_run_counts_apply_to_the_following_symbol():
    cells, w, h = parse_rle("4o!")
    assert len(cells) == 4 and (w, h) == (4, 1)


def test_dollar_advances_a_row_and_resets_x():
    cells, _, h = parse_rle("o$o!")
    assert sorted(cells) == [(0, 0), (0, 1)]
    assert h == 2


def test_counted_dollar_skips_blank_rows():
    cells, _, h = parse_rle("o3$o!")
    assert sorted(cells) == [(0, 0), (0, 3)]
    assert h == 4


def test_ignores_the_x_y_header_line():
    cells, w, h = parse_rle("x = 3, y = 1, rule = B3/S23\n3o!")
    assert len(cells) == 3 and (w, h) == (3, 1)


def test_ignores_comment_lines():
    cells, _, _ = parse_rle("#N Blinker\n#C a comment\n3o!")
    assert len(cells) == 3


def test_joins_a_pattern_split_across_multiple_lines():
    one = parse_rle("bob$2bo$3o!")[0]
    many = parse_rle("bob$\n2bo$\n3o!")[0]
    assert sorted(one) == sorted(many)


def test_terminator_is_optional():
    assert parse_rle("3o")[0] == parse_rle("3o!")[0]


def test_stops_reading_after_the_terminator():
    cells, _, _ = parse_rle("o!ooooo")
    assert cells == [(0, 0)]


def test_treats_dot_as_a_dead_cell():
    cells, _, _ = parse_rle("..o!")
    assert cells == [(2, 0)]


def test_treats_asterisk_as_a_live_cell():
    assert parse_rle("*!")[0] == [(0, 0)]


def test_is_case_insensitive():
    assert sorted(parse_rle("BOB$2BO$3O!")[0]) == sorted(parse_rle("bob$2bo$3o!")[0])


def test_ignores_whitespace_inside_the_body():
    assert sorted(parse_rle("b o b $ 2 b o $ 3 o !")[0]) == sorted(parse_rle("bob$2bo$3o!")[0])


def test_width_and_height_come_from_live_cells_not_the_header():
    # header claims 10x10; the actual live cells span 3x1
    _, w, h = parse_rle("x = 10, y = 10\n3o!")
    assert (w, h) == (3, 1)


def test_trailing_dead_cells_do_not_widen_the_pattern():
    _, w, _ = parse_rle("o2b!")
    assert w == 1


def test_rejects_a_pattern_with_no_live_cells():
    with pytest.raises(ValueError, match="no live cells"):
        parse_rle("3b!")


def test_rejects_an_empty_string():
    with pytest.raises(ValueError, match="no live cells"):
        parse_rle("")


def test_rejects_an_unexpected_character():
    with pytest.raises(ValueError, match="unexpected character"):
        parse_rle("3o%!")
