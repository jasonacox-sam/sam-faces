"""Tests for sam_faces.pets — the registry + the model-answer matching logic (no Ollama needed)."""

from sam_faces import pets


def test_registry_roundtrip(tmp_path, monkeypatch):
    import sam_faces.database as db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "people.db")
    pets.add_pet("Bailey", "dog", "small tan long-haired dog, graying muzzle")
    pets.add_pet("Bailey", "dog", "updated description")  # upsert, not duplicate
    names = [p["name"] for p in pets.list_pets()]
    assert names == ["Bailey"]
    assert pets.list_pets()[0]["description"] == "updated description"


def test_match_picks_registered_name():
    roster = [{"name": "Bailey"}, {"name": "Bruno"}]
    assert pets._match("Bailey", roster) == "Bailey"
    assert pets._match("This looks like Bruno.", roster) == "Bruno"
    assert pets._match("UNKNOWN_ANIMAL", roster) is None
    assert pets._match("NONE", roster) is None
