import json

import pytest

from app.style.character_types import DEFAULT_PATH, CharacterTypesError, load_character_types


def write(tmp_path, data) -> object:
    path = tmp_path / "types.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_shipped_file_is_valid():
    types = load_character_types(DEFAULT_PATH)
    labels = {t.label for t in types}
    assert {"Influencer", "Fashion Model", "Cosplay Character"} <= labels
    assert all(t.description and t.guidance for t in types)


def test_unknown_keys_are_ignored(tmp_path):
    path = write(tmp_path, {"_comment": "x", "character_types": [{"id": "a", "label": " A ", "extra": 1}]})
    [t] = load_character_types(path)
    assert (t.id, t.label, t.description, t.guidance) == ("a", "A", "", "")


@pytest.mark.parametrize("data, message", [
    ("{nope", "not valid JSON"),
    ([], "character_types"),
    ({"character_types": {}}, "character_types"),
    ({"character_types": ["x"]}, "must be an object"),
    ({"character_types": [{"label": "A"}]}, '"id"'),
    ({"character_types": [{"id": "Has Space", "label": "A"}]}, '"id"'),
    ({"character_types": [{"id": "a", "label": ""}]}, '"label"'),
    ({"character_types": [{"id": "a", "label": "A"}, {"id": "a", "label": "B"}]}, "duplicate"),
    ({"character_types": [{"id": "a", "label": "A", "guidance": 3}]}, '"guidance"'),
])
def test_invalid_files_are_explained(tmp_path, data, message):
    with pytest.raises(CharacterTypesError, match=message):
        load_character_types(write(tmp_path, data))


def test_missing_file(tmp_path):
    with pytest.raises(CharacterTypesError, match="not found"):
        load_character_types(tmp_path / "missing.json")
