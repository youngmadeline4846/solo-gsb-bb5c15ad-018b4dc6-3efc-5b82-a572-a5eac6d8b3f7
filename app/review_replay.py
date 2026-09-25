"""复核结论回放 (review verdict replay).

Every saved human review decision is re-checked against the **current**
standard address table with the **default** rules: the reviewed record's
original query is scored again (plain scoring -- the review-consensus memory
is not consulted) and the fresh verdict is compared with the human decision:

* ``confirmed`` / ``overridden`` (positive decisions):
    - the replay auto-hits the very address the human chose -> ``consistent`` 一致
    - the replay auto-hits a *different* address            -> ``conflict``   冲突
    - the replay does not auto-hit anything                 -> ``missed``     漏命中
* ``rejected`` (判无匹配):
    - the replay now auto-hits some address                 -> ``false_hit``  误命中
    - the replay still does not auto-hit                    -> ``consistent`` 一致

Each item carries the original verdict (action, chosen address, review-time
score), the current preferred candidate and score, plus a human-readable
reason, so rule or address-table changes that endanger past decisions are
easy to spot. The summary counts every category and items can be filtered
by category.

The replay is read-only: it only runs the scorer against the live table. It
never writes match logs and never modifies the reviewed records.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db
from .config import Settings
from .matcher import match_address, result_to_dict

CATEGORY_CONSISTENT = "consistent"
CATEGORY_CONFLICT = "conflict"
CATEGORY_MISSED = "missed"
CATEGORY_FALSE_HIT = "false_hit"

VALID_CATEGORIES = (
    CATEGORY_CONSISTENT,
    CATEGORY_CONFLICT,
    CATEGORY_MISSED,
    CATEGORY_FALSE_HIT,
)

CATEGORY_LABELS = {
    CATEGORY_CONSISTENT: "一致",
    CATEGORY_CONFLICT: "冲突",
    CATEGORY_MISSED: "漏命中",
    CATEGORY_FALSE_HIT: "误命中",
}

ACTION_LABELS = {
    "confirmed": "确认首选",
    "overridden": "改派",
    "rejected": "判无匹配",
}

_STATUS_LABELS = {
    "auto_matched": "自动命中",
    "needs_review": "待确认",
    "no_match": "无匹配",
}

# Display order: risks first (a conflicting auto-hit is the most urgent, a
# merely missed one the least), consistent records last; within a category
# the oldest records come first.
_CATEGORY_RANK = {
    CATEGORY_CONFLICT: 0,
    CATEGORY_FALSE_HIT: 1,
    CATEGORY_MISSED: 2,
    CATEGORY_CONSISTENT: 3,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _classify(action: str, selected_id: int | None, selected_missing: bool,
              status: str, best_id: int | None, best_score: float | None
              ) -> tuple[str, str]:
    """Replay category + human-readable reason for one reviewed record."""
    if action in ("confirmed", "overridden"):
        chosen = f"#{selected_id}" if selected_id is not None else "（已删除地址）"
        gone = "；注意人工所选地址已不在当前标准地址表中" if selected_missing else ""
        if status == "auto_matched" and best_id == selected_id:
            return CATEGORY_CONSISTENT, (
                f"当前回放自动命中人工所选地址 {chosen}"
                f"（{_pct(best_score)}），与人工结论一致"
            )
        if status == "auto_matched":
            return CATEGORY_CONFLICT, (
                f"当前回放自动命中 #{best_id}（{_pct(best_score)}），"
                f"与人工所选地址 {chosen} 不同{gone}"
            )
        return CATEGORY_MISSED, (
            f"当前回放未自动命中（状态：{_STATUS_LABELS.get(status, status)}），"
            f"人工所选地址 {chosen} 按当前规则无法自动复现{gone}"
        )
    # rejected：人工判无匹配
    if status == "auto_matched":
        return CATEGORY_FALSE_HIT, (
            f"人工判为无匹配，但当前回放自动命中 #{best_id}"
            f"（{_pct(best_score)}），原拒绝结论可能被新地址或新规则覆盖"
        )
    return CATEGORY_CONSISTENT, (
        "人工判为无匹配，当前回放仍未自动命中"
        f"（状态：{_STATUS_LABELS.get(status, status)}），结论一致"
    )


def compute_replay(conn: sqlite3.Connection, settings: Settings,
                   category: str | None = None) -> dict[str, Any]:
    """Re-run every reviewed record's query on the live table and compare.

    ``category`` optionally restricts the returned items to one category;
    the summary counts always cover every reviewed record. The replay is
    recomputed on each call -- a freshly changed table or threshold is
    reflected the next time it is opened. Nothing is persisted: no match
    log is written and no review record is modified.
    """
    rows = conn.execute(
        "SELECT * FROM match_logs WHERE status = 'reviewed' ORDER BY id"
    ).fetchall()
    # One snapshot of the current table is shared by every replay in this call.
    addresses = db.all_addresses(conn)
    addr_by_id = {row["id"]: row for row in addresses}

    items: list[dict[str, Any]] = []
    for row in rows:
        action = row["review_action"]
        selected_id = row["selected_id"]

        # The human-chosen address text: prefer the live table, fall back to
        # the candidate snapshot stored with the log (the address may be gone;
        # the FK then already nulled selected_id).
        selected_missing = False
        selected_text = None
        if action in ("confirmed", "overridden"):
            if selected_id is not None and selected_id in addr_by_id:
                selected_text = addr_by_id[selected_id]["full_text"]
            else:
                selected_missing = True
                if selected_id is not None:
                    for cand in json.loads(row["candidates_json"] or "[]"):
                        if cand["address"]["id"] == selected_id:
                            selected_text = cand["address"]["full_text"]
                            break

        # Pure scoring replay: no review-consensus lookup, nothing persisted.
        result = match_address(row["query"], addresses, settings)
        best = result.best
        best_id = best.address["id"] if best else None
        best_score = best.score if best else None
        cat, verdict_reason = _classify(
            action, selected_id, selected_missing,
            result.status, best_id, best_score,
        )
        items.append({
            "match_id": row["id"],
            "query": row["query"],
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
            "review_action": action,
            "review_action_label": ACTION_LABELS.get(action, action),
            "selected_id": selected_id,
            "selected_full_text": selected_text,
            "selected_missing": selected_missing,
            "review_score": row["score"],
            "category": cat,
            "category_label": CATEGORY_LABELS[cat],
            "verdict_reason": verdict_reason,
            "replay_status": result.status,
            "replay_reason": result.reason,
            "replay_best_id": best_id,
            "replay_best_score": best_score,
            "replay_best_full_text": best.address["full_text"] if best else None,
            "replay": result_to_dict(result),
        })

    items.sort(key=lambda it: (_CATEGORY_RANK[it["category"]], it["match_id"]))

    counts = {
        cat: sum(1 for it in items if it["category"] == cat)
        for cat in VALID_CATEGORIES
    }
    summary = {
        "total_reviewed": len(items),
        **counts,
        "risk_count": len(items) - counts[CATEGORY_CONSISTENT],
    }

    if category is not None:
        items = [it for it in items if it["category"] == category]

    return {
        "generated_at": _now(),
        **summary,
        "thresholds": {
            "auto_threshold": settings.auto_threshold,
            "margin_threshold": settings.margin_threshold,
            "reject_threshold": settings.reject_threshold,
            "token_weight": settings.token_weight,
        },
        "items": items,
    }
