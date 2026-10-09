"""Execution-grounded reward. Differences from the old train.py:

- `database` is a REQUIRED routing input (no silent default-DB execution).
- Returns a dict with the raw composite reward AND a normalized
  result-jaccard in [0, 1]; metric code must use the normalized fields,
  never the raw composite, for anything named "accuracy".
"""

from __future__ import annotations

import re


def normalize_cypher(text: str) -> str:
    return " ".join(str(text).strip().rstrip(";").lower().split())


def shape_ok(pred: str, gold: str) -> tuple[bool, str]:
    if not pred.strip():
        return False, "empty"
    if not re.match(r"^\s*(?:(?:EXPLAIN|PROFILE)\s+)?(?:OPTIONAL\s+MATCH|MATCH|CALL|WITH|UNWIND|CREATE|MERGE)\b",
                    pred, re.IGNORECASE):
        return False, "missing_query_start"
    if re.search(r"\bRETURN\b", gold, re.IGNORECASE) and not re.search(r"\bRETURN\b", pred, re.IGNORECASE):
        return False, "missing_return"
    return True, "ok"


def result_jaccard(gold_rows, pred_rows) -> float:
    g = set(tuple(r) for r in gold_rows)
    p = set(tuple(r) for r in pred_rows)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def compute_reward(pred_cypher: str, gold_cypher: str, neo4j_client,
                   database: str, gold_rows=None) -> dict:
    """Execute pred (+ gold unless gold_rows given) on `database`.

    gold_rows: precomputed list-of-tuples for the gold query (preferred in
    training loops to avoid re-executing gold every step).
    """
    pred = str(pred_cypher or "").strip().rstrip(";").strip()
    gold = str(gold_cypher or "").strip().rstrip(";").strip()
    ok, reason = shape_ok(pred, gold)
    if not ok:
        return {"branch": "shape_fail", "reason": reason,
                "composite": -3.0, "jaccard": 0.0, "exec_ok": False}

    if gold_rows is None:
        gold_rows = neo4j_client.execute(gold, database=database)
        if gold_rows is None:
            return {"branch": "gold_exec_fail", "composite": 0.0,
                    "jaccard": 0.0, "exec_ok": False}

    pred_rows = neo4j_client.execute(pred, database=database)
    exec_ok = pred_rows is not None
    jac = result_jaccard(gold_rows, pred_rows if exec_ok else [])

    exact = normalize_cypher(pred) == normalize_cypher(gold)
    if exact and (exec_ok or not gold_rows):
        branch, base, jac = "exact_match", 3.0, 1.0
    elif exec_ok and jac > 0:
        branch, base = "partial_match", 0.5
    elif exec_ok:
        branch, base = "executed_no_overlap", 0.2
    else:
        branch, base = "exec_fail", -3.0

    garbage = min(2.0, sum(1 for ch in pred if ord(ch) > 127) / max(len(pred), 1) * 8)
    composite = max(-6.0, min(6.0, base + 2.0 * jac - garbage))
    return {"branch": branch, "composite": composite, "jaccard": jac,
            "exact": exact, "exec_ok": exec_ok}
