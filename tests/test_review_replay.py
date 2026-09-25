"""Tests for the review verdict replay (``/api/review-replay``)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import db  # noqa: E402
from app.main import app, get_conn  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# Match ids created by this module, shared across its ordered tests.
STATE: dict[str, int] = {}


def _match_save(client, query: str) -> dict:
    r = client.post("/api/match", json={"query": query, "save": True})
    assert r.status_code == 200
    return r.json()


def _review(client, match_id: int, action: str, selected_id: int | None = None) -> None:
    body: dict = {"action": action}
    if selected_id is not None:
        body["selected_id"] = selected_id
    r = client.post(f"/api/matches/{match_id}/review", json=body)
    assert r.status_code == 200, r.text


def _replay(client, category: str | None = None) -> dict:
    qs = f"?category={category}" if category else ""
    r = client.get(f"/api/review-replay{qs}")
    assert r.status_code == 200
    return r.json()


def _item_for(data: dict, match_id: int) -> dict:
    matches = [it for it in data["items"] if it["match_id"] == match_id]
    assert len(matches) == 1, [it["match_id"] for it in data["items"]]
    return matches[0]


def test_replay_categories_and_summary(client):
    before = _replay(client)

    # 1) consistent: an auto-matched query the human confirmed -> the replay
    #    auto-hits the very same address.
    m1 = _match_save(client, "123 main st ny")
    assert m1["status"] == "auto_matched"
    _review(client, m1["match_id"], "confirmed")
    STATE["consistent"] = m1["match_id"]

    # 2) conflict: the human overrode the auto-hit "100 Lake Drive" to
    #    "1000 Lake Drive"; the replay still auto-hits the former.
    m2 = _match_save(client, "100 Lake Dr, Austin TX")
    assert m2["status"] == "auto_matched"
    other = next(
        c["address"]["id"] for c in m2["candidates"]
        if "1000 Lake Drive" in c["address"]["full_text"]
    )
    assert other != m2["best_id"]
    _review(client, m2["match_id"], "overridden", other)
    STATE["conflict"] = m2["match_id"]

    # 3) missed: the human confirmed a needs_review record; the replay still
    #    cannot auto-hit the chosen address (house number 1010 vs 1000).
    m3 = _match_save(client, "1010 Lake Dr, Austin TX")
    assert m3["status"] == "needs_review"
    _review(client, m3["match_id"], "confirmed")
    STATE["missed"] = m3["match_id"]

    # 4) consistent (for now): a no_match query the human judged no-match.
    m4 = _match_save(client, "999 Zxqwv Replay Blvd, Nowhereton")
    assert m4["status"] == "no_match"
    _review(client, m4["match_id"], "rejected")
    STATE["rejected"] = m4["match_id"]

    data = _replay(client)

    it1 = _item_for(data, STATE["consistent"])
    assert it1["category"] == "consistent"
    assert it1["category_label"] == "一致"
    assert it1["review_action"] == "confirmed"
    assert it1["review_action_label"] == "确认首选"
    assert it1["selected_id"] == m1["best_id"]
    assert it1["selected_full_text"] == "123 Main Street"
    assert it1["selected_missing"] is False
    assert it1["replay_status"] == "auto_matched"
    assert it1["replay_best_id"] == it1["selected_id"]
    assert it1["replay_best_full_text"] == "123 Main Street"
    assert it1["replay"]["best_id"] == it1["selected_id"]
    assert it1["replay"]["candidates"]

    it2 = _item_for(data, STATE["conflict"])
    assert it2["category"] == "conflict"
    assert it2["category_label"] == "冲突"
    assert it2["review_action"] == "overridden"
    assert it2["review_action_label"] == "改派"
    assert it2["selected_id"] == other
    assert it2["selected_full_text"] == "1000 Lake Drive"
    assert it2["replay_status"] == "auto_matched"
    assert it2["replay_best_id"] == m2["best_id"]
    assert it2["replay_best_id"] != it2["selected_id"]

    it3 = _item_for(data, STATE["missed"])
    assert it3["category"] == "missed"
    assert it3["category_label"] == "漏命中"
    assert it3["replay_status"] == "needs_review"
    assert it3["replay_best_id"] == it3["selected_id"]  # preferred but not auto

    it4 = _item_for(data, STATE["rejected"])
    assert it4["category"] == "consistent"
    assert it4["review_action"] == "rejected"
    assert it4["review_action_label"] == "判无匹配"
    assert it4["selected_id"] is None
    assert it4["replay_status"] == "no_match"

    # summary counts: exactly four new reviewed records on top of `before`
    assert data["total_reviewed"] == before["total_reviewed"] + 4
    assert data["consistent"] == before["consistent"] + 2
    assert data["conflict"] == before["conflict"] + 1
    assert data["missed"] == before["missed"] + 1
    assert data["false_hit"] == before["false_hit"]
    assert data["risk_count"] == (
        data["conflict"] + data["missed"] + data["false_hit"]
    )
    assert data["total_reviewed"] == (
        data["consistent"] + data["conflict"] + data["missed"] + data["false_hit"]
    )
    assert data["thresholds"]["auto_threshold"] == pytest.approx(0.80)

    # risks first (conflict < false_hit < missed < consistent), then match id
    rank = {"conflict": 0, "false_hit": 1, "missed": 2, "consistent": 3}
    keys = [(rank[it["category"]], it["match_id"]) for it in data["items"]]
    assert keys == sorted(keys)

    # every item carries the original verdict, the current best and a reason
    for it in data["items"]:
        assert it["verdict_reason"]
        assert "replay_best_score" in it
        assert it["replay"]["query"] == it["query"]


def test_upload_flips_rejected_record_to_false_hit(client):
    # The rejected query from the previous test becomes auto-matchable once
    # the missing address is uploaded -> 误命中 (false_hit).
    before = _replay(client)
    csv_bytes = b"id,address,city,zip\nRPL-999,999 Zxqwv Replay Blvd,Nowhereton,99999\n"
    up = client.post("/api/addresses/upload",
                     files={"file": ("r.csv", csv_bytes, "text/csv")})
    assert up.status_code == 200
    assert up.json()["inserted"] == 1

    data = _replay(client)
    item = _item_for(data, STATE["rejected"])
    assert item["category"] == "false_hit"
    assert item["category_label"] == "误命中"
    assert item["replay_status"] == "auto_matched"
    assert "Zxqwv Replay" in item["replay_best_full_text"]
    assert data["false_hit"] == before["false_hit"] + 1
    assert data["consistent"] == before["consistent"] - 1
    assert data["total_reviewed"] == before["total_reviewed"]


def test_category_filter_keeps_summary_counts(client):
    full = _replay(client)
    assert full["total_reviewed"] >= 4
    for cat in ("consistent", "conflict", "missed", "false_hit"):
        filtered = _replay(client, cat)
        assert all(it["category"] == cat for it in filtered["items"])
        assert len(filtered["items"]) == full[cat]
        # summary counts are independent of the filter
        assert filtered["total_reviewed"] == full["total_reviewed"]
        assert filtered[cat] == full[cat]
        assert filtered["risk_count"] == full["risk_count"]
    # every category built by this module is represented
    for cat in ("consistent", "conflict", "missed", "false_hit"):
        assert full[cat] >= 1


def test_invalid_category_filter(client):
    r = client.get("/api/review-replay", params={"category": "bogus"})
    assert r.status_code == 400


def test_selected_address_deleted_keeps_snapshot(client):
    # Confirm an auto-hit, then delete the chosen address directly: the FK
    # nulls selected_id in the log row, but the replay restores the original
    # id + full text from the review-time snapshot and flags it as deleted.
    r = client.post("/api/addresses", json={
        "addresses": [{"address": "555 Zxqwv Replaydel Avenue, Delton 90001"}],
        "skip_duplicates": True,
    })
    assert r.json()["inserted"] == 1
    addr = next(
        a for a in client.get("/api/addresses?limit=500").json()["items"]
        if "Replaydel" in a["full_text"]
    )
    m = _match_save(client, "555 Zxqwv Replaydel Ave, Delton 90001")
    assert m["status"] == "auto_matched"
    _review(client, m["match_id"], "confirmed")

    # The review persisted the snapshot alongside the FK.
    log = client.get(f"/api/matches/{m['match_id']}").json()
    assert log["selected_id"] == addr["id"]
    assert log["selected_id_snapshot"] == addr["id"]
    assert log["selected_full_text"] == addr["full_text"]

    conn = get_conn()
    conn.execute("DELETE FROM addresses WHERE id = ?", (addr["id"],))
    conn.commit()
    # The FK column itself is still nulled by the deletion...
    assert client.get(f"/api/matches/{m['match_id']}").json()["selected_id"] is None

    data = _replay(client)
    item = _item_for(data, m["match_id"])
    # ...but the replay shows the original pick from the snapshot.
    assert item["selected_known"] is True
    assert item["selected_id"] == addr["id"]
    assert item["selected_full_text"] == addr["full_text"]
    assert item["selected_missing"] is True
    assert item["review_score"] == pytest.approx(m["best_score"])
    assert item["category"] == "missed"
    assert f"#{addr['id']}" in item["verdict_reason"]
    assert "已删除" in item["verdict_reason"]


def test_overridden_address_deleted_keeps_snapshot(client):
    # An overridden (改派) decision is snapshotted the same way; after the
    # chosen address is deleted the replay still names it and classifies the
    # record as a conflict with the current auto-hit.
    r = client.post("/api/addresses", json={
        "addresses": [{"address": "77 Zxqwv Ovrsnap Road, Ovrton 90002"}],
        "skip_duplicates": True,
    })
    assert r.json()["inserted"] == 1
    addr = next(
        a for a in client.get("/api/addresses?limit=500").json()["items"]
        if "Ovrsnap" in a["full_text"]
    )
    m = _match_save(client, "77 Zxqwv Ovrsnap Rd, Ovrton 90002")
    assert m["status"] == "auto_matched"
    other = next(
        c["address"]["id"] for c in m["candidates"]
        if c["address"]["id"] != m["best_id"]
    )
    _review(client, m["match_id"], "overridden", other)

    log = client.get(f"/api/matches/{m['match_id']}").json()
    assert log["selected_id_snapshot"] == other
    assert log["selected_full_text"]

    conn = get_conn()
    conn.execute("DELETE FROM addresses WHERE id = ?", (other,))
    conn.commit()

    item = _item_for(_replay(client), m["match_id"])
    assert item["selected_known"] is True
    assert item["selected_id"] == other
    assert item["selected_full_text"] == log["selected_full_text"]
    assert item["selected_missing"] is True
    # the replay still auto-hits the remaining address -> conflict
    assert item["category"] == "conflict"
    assert item["replay_best_id"] == addr["id"]
    assert f"#{other}" in item["verdict_reason"]
    assert "已删除" in item["verdict_reason"]


def test_group_review_persists_snapshot(client):
    # Same-case merged review (整组确认首选): every member of the group gets
    # the snapshot of the chosen address, which the replay keeps showing
    # after the address is deleted.
    r = client.post("/api/addresses", json={
        "addresses": [{"address": "404 Zxqwv Grpsnap Blvd, Grptown 90003"}],
        "skip_duplicates": True,
    })
    assert r.json()["inserted"] == 1
    addr = next(
        a for a in client.get("/api/addresses?limit=500").json()["items"]
        if "Grpsnap" in a["full_text"]
    )
    # Two spellings of the same case: house 405 vs stored 404 -> needs_review,
    # identical normalised key and candidate scores -> one review group.
    m1 = _match_save(client, "405 Zxqwv Grpsnap Blvd, Grptown 90003")
    m2 = _match_save(client, "405 Zxqwv Grpsnap Boulevard, Grptown 90003")
    assert {m1["status"], m2["status"]} == {"needs_review"}

    groups = client.get("/api/review-groups").json()["groups"]
    group = next(
        g for g in groups
        if {m1["match_id"], m2["match_id"]} == set(g["match_ids"])
    )
    r = client.post("/api/review-groups/review", json={
        "group_key": group["group_key"],
        "match_ids": group["match_ids"],
        "action": "confirmed",
    })
    assert r.status_code == 200, r.text

    for mid in (m1["match_id"], m2["match_id"]):
        log = client.get(f"/api/matches/{mid}").json()
        assert log["review_action"] == "confirmed"
        assert log["selected_id_snapshot"] == addr["id"]
        assert log["selected_full_text"] == addr["full_text"]

    conn = get_conn()
    conn.execute("DELETE FROM addresses WHERE id = ?", (addr["id"],))
    conn.commit()

    data = _replay(client)
    for mid in (m1["match_id"], m2["match_id"]):
        item = _item_for(data, mid)
        assert item["selected_known"] is True
        assert item["selected_id"] == addr["id"]
        assert item["selected_full_text"] == addr["full_text"]
        assert item["selected_missing"] is True
        assert item["category"] == "missed"


def test_group_override_persists_snapshot(client):
    # Same-case merged review (整组改派共同候选): the shared reassign target
    # is snapshotted on every member and survives its later deletion.
    r = client.post("/api/addresses", json={
        "addresses": [
            {"address": "404 Zxqwv Ovgrp Blvd, Ovgtown 90004"},
            {"address": "406 Zxqwv Ovgrp Blvd, Ovgtown 90004"},
        ],
        "skip_duplicates": True,
    })
    assert r.json()["inserted"] == 2
    addrs = {
        a["house"]: a
        for a in client.get("/api/addresses?limit=500").json()["items"]
        if "Ovgrp" in a["full_text"]
    }
    target = addrs["406"]
    # house 405 ties 404/406 -> needs_review for both spellings, one group.
    m1 = _match_save(client, "405 Zxqwv Ovgrp Blvd, Ovgtown 90004")
    m2 = _match_save(client, "No. 405 Zxqwv Ovgrp Blvd, Ovgtown 90004")
    assert {m1["status"], m2["status"]} == {"needs_review"}

    groups = client.get("/api/review-groups").json()["groups"]
    group = next(
        g for g in groups
        if {m1["match_id"], m2["match_id"]} == set(g["match_ids"])
    )
    r = client.post("/api/review-groups/review", json={
        "group_key": group["group_key"],
        "match_ids": group["match_ids"],
        "action": "overridden",
        "selected_id": target["id"],
    })
    assert r.status_code == 200, r.text

    for mid in (m1["match_id"], m2["match_id"]):
        log = client.get(f"/api/matches/{mid}").json()
        assert log["review_action"] == "overridden"
        assert log["selected_id_snapshot"] == target["id"]
        assert log["selected_full_text"] == target["full_text"]

    conn = get_conn()
    conn.execute("DELETE FROM addresses WHERE id = ?", (target["id"],))
    conn.commit()

    data = _replay(client)
    for mid in (m1["match_id"], m2["match_id"]):
        item = _item_for(data, mid)
        assert item["selected_known"] is True
        assert item["selected_id"] == target["id"]
        assert item["selected_full_text"] == target["full_text"]
        assert item["selected_missing"] is True
        # 404 remains but only reaches needs_review -> 漏命中
        assert item["category"] == "missed"
        assert f"#{target['id']}" in item["verdict_reason"]


def test_legacy_record_without_snapshot_shows_unknown(client):
    # A record reviewed before snapshots existed: the FK was nulled when the
    # address was deleted and no snapshot columns were ever written. The
    # replay must say the original selection is unknown instead of guessing
    # an address from the stored candidate list.
    m = _match_save(client, "100 Lake Dr, Austin TX")
    assert m["status"] == "auto_matched"
    _review(client, m["match_id"], "confirmed")

    # Simulate the legacy state: FK nulled by a deletion, no snapshot saved.
    conn = get_conn()
    conn.execute(
        "UPDATE match_logs SET selected_id = NULL, selected_id_snapshot = NULL,"
        " selected_full_text = NULL WHERE id = ?",
        (m["match_id"],),
    )
    conn.commit()

    item = _item_for(_replay(client), m["match_id"])
    assert item["selected_known"] is False
    assert item["selected_id"] is None
    # no guessing from the candidate list even though candidates exist
    assert item["selected_full_text"] is None
    assert item["selected_missing"] is False
    assert "原选择未知" in item["verdict_reason"]
    # the replay auto-hit cannot be matched to the unknown pick -> conflict
    assert item["category"] == "conflict"
    assert item["replay_status"] == "auto_matched"


def test_replay_is_read_only(client):
    logs_before = client.get("/api/matches?limit=1").json()["total"]
    reviewed = client.get("/api/matches?status=reviewed&limit=1").json()["items"]
    assert reviewed, "expected at least one reviewed record"
    record_before = reviewed[0]

    _replay(client)
    _replay(client, "conflict")
    _replay(client, "consistent")

    # no new match logs were written
    assert client.get("/api/matches?limit=1").json()["total"] == logs_before
    # the reviewed record itself is untouched
    after = client.get(f"/api/matches/{record_before['id']}").json()
    assert after["status"] == "reviewed"
    assert after["review_action"] == record_before["review_action"]
    assert after["selected_id"] == record_before["selected_id"]
    assert after["score"] == record_before["score"]
    assert after["reviewed_at"] == record_before["reviewed_at"]


def test_empty_replay_on_fresh_database():
    # A throwaway database with only the sample addresses (and no reviewed
    # records at all) yields an empty replay.
    import os
    import tempfile
    from pathlib import Path

    from app import review_replay
    from app.config import get_settings
    from app import ingest as ingest_mod

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    old = os.environ.get("ADDRESS_DB_PATH")
    os.environ["ADDRESS_DB_PATH"] = path
    conn = None
    try:
        conn = db.connect()
        db.init_db(conn)
        sample = Path(get_settings().sample_csv)
        ingest_mod.add_many(conn, ingest_mod.parse_csv(sample.read_bytes()))
        out = review_replay.compute_replay(conn, get_settings())
        assert out["items"] == []
        assert out["total_reviewed"] == 0
        assert out["consistent"] == 0
        assert out["conflict"] == 0
        assert out["missed"] == 0
        assert out["false_hit"] == 0
        assert out["risk_count"] == 0
    finally:
        if conn is not None:
            conn.close()
        if old is None:
            os.environ.pop("ADDRESS_DB_PATH", None)
        else:
            os.environ["ADDRESS_DB_PATH"] = old
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass
