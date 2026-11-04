import pytest

from editp.llm import explicit_replacements, layout_map, normalize, parse_json, replace_in_lines


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('好的，結果如下：{"a": [1]} 以上') == {"a": [1]}
    with pytest.raises(ValueError):
        parse_json("沒有 JSON")


def test_explicit_replacements_only_for_literal_clauses():
    assert explicit_replacements("把「迷局」改成「謎局」") == [("迷局", "謎局")]
    assert explicit_replacements("迷局改成謎局，破解改成化解") == [("迷局", "謎局"), ("破解", "化解")]
    assert explicit_replacements("刪除右上角的亂碼") is None
    assert explicit_replacements("") is None


LINES = [{"id": 0, "text": "亂世迷局"}, {"id": 1, "text": "電話：0912"}]


def test_replace_in_lines_edits_matching_lines():
    assert replace_in_lines(LINES, [("迷局", "謎局")]) == [{"id": 0, "new_text": "亂世謎局"}]


def test_replace_in_lines_defers_to_llm():
    assert replace_in_lines(LINES, [("不存在", "x")]) is None
    assert replace_in_lines(LINES, [("電話", "0987")]) is None  # a field label: the user means its value


def test_normalize_keeps_real_changes_and_last_edit_wins():
    edits = [
        {"id": 0, "new_text": "第一版"},
        {"id": 0, "new_text": " 亂世謎局 "},
        {"id": 1, "new_text": "電話：0912"},  # unchanged
        {"id": 9, "new_text": "x"},  # unknown line
        {"id": 1, "new_text": None},
        "garbage",
    ]
    assert normalize(edits, LINES) == [{"id": 0, "new_text": "亂世謎局"}]


def test_layout_map_groups_rows_and_orders_by_x():
    lines = [
        {"id": 0, "text": "右", "box": [600, 0, 700, 20]},
        {"id": 1, "text": "左", "box": [0, 2, 100, 22]},
        {"id": 2, "text": "下", "box": [0, 100, 100, 120]},
    ]
    rows = layout_map(lines, 800, cols=80).split("\n")
    assert len(rows) == 2
    assert rows[0].index("[1]左") < rows[0].index("[0]右")
    assert rows[1].strip() == "[2]下"
