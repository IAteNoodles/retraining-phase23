"""Execution-grounded reward (GRPO-era, inspired by the historical design).

Per candidate it returns execution Jaccard plus structural/schema/entity
terms and penalties. Metric code must use the normalized fields (jaccard,
structure, ...) for anything named "accuracy" - never raw composites.

Historical weights kept where verified: structure +0.8 (0.30 AST proxy +
0.25 prop + 0.20 rel + 0.15 label + 0.10 clause F1), schema +0.3, entity
+0.2, garbage -0.5 (capped), trivial-query -0.3.
`database` is a REQUIRED routing input (no silent default-DB execution).
"""

from __future__ import annotations

import difflib
import re

NON_PROPS = {"year", "years", "quarter", "quarters", "month", "months",
             "week", "weeks", "day", "days", "ordinalday", "dayofweek",
             "hour", "hours", "minute", "minutes", "second", "seconds",
             "millisecond", "milliseconds", "microsecond", "microseconds",
             "epochmillis", "epochseconds", "id"}

CLAUSES = ["match", "where", "return", "order by", "limit", "with", "call",
           "unwind", "optional match", "distinct"]


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


def _f1(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    p = len(pred_set & gold_set) / len(pred_set)
    r = len(pred_set & gold_set) / len(gold_set)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def parse_elements(cypher: str):
    """Labels, rel types, props, clause set referenced by a query (lowercased)."""
    t = str(cypher or "")
    labels = set(re.findall(r"\(:([A-Za-z_]\w*)", t))
    labels |= set(re.findall(r"\([A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)", t))
    rels = set(re.findall(r"\[\s*:([A-Za-z_]\w*)", t))
    rels |= set(re.findall(r"\[[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)", t))
    props = set(re.findall(r"\.\s*([A-Za-z_]\w*)", t)) - NON_PROPS
    low = t.lower()
    clauses = {c for c in CLAUSES if re.search(r"\b" + re.escape(c) + r"\b", low)}
    return ({x.lower() for x in labels}, {x.lower() for x in rels},
            {x.lower() for x in props}, clauses)


def structure_score(pred: str, gold: str) -> dict:
    """0.30 AST-proxy (token-sequence ratio) + 0.25/0.20/0.15/0.10 F1s."""
    pl, rl, pp, pc = parse_elements(pred)
    gl, gr, gp, gc = parse_elements(gold)
    ast = difflib.SequenceMatcher(None, normalize_cypher(pred).split(),
                                  normalize_cypher(gold).split()).ratio()
    out = {"ast": ast, "label_f1": _f1(pl, gl), "rel_f1": _f1(rl, gr),
           "prop_f1": _f1(pp, gp), "clause_f1": _f1(pc, gc)}
    out["composite"] = (0.30 * out["ast"] + 0.25 * out["prop_f1"]
                        + 0.20 * out["rel_f1"] + 0.15 * out["label_f1"]
                        + 0.10 * out["clause_f1"])
    return out


def schema_score(input_text: str, pred_labels: set, pred_rels: set, pred_props: set) -> float:
    """Fraction of predicted schema elements present in the input schema.
    1.0 when the prediction references nothing (vacuous)."""
    from .data_checks import _schema_vocab
    if not input_text:
        return 0.0
    vl, vr, vp = _schema_vocab(input_text)
    tot = len(pred_labels) + len(pred_rels) + len(pred_props)
    if not tot:
        return 1.0
    ok = (len(pred_labels & vl) + len(pred_rels & vr) + len(pred_props & vp))
    return ok / tot


def entity_score(question: str, pred: str) -> float:
    """Fraction of quoted literals/numbers in the question that appear in pred.
    1.0 when the question carries no literals to ground."""
    lits = set(re.findall(r"'([^']+)'|\"([^\"]+)\"", str(question or "")))
    flat = {a or b for a, b in lits}
    flat |= set(re.findall(r"\b\d[\d,.]*\b", str(question or "")))
    if not flat:
        return 1.0
    low = str(pred or "").lower()
    return sum(1 for x in flat if x.lower() in low) / len(flat)


def triviality(pred: str) -> float:
    """1.0 if the query looks like a shortcut (no MATCH, constant RETURN)."""
    t = str(pred or "")
    if not re.search(r"\bMATCH\b", t, re.IGNORECASE) and re.search(r"\bRETURN\b", t, re.IGNORECASE):
        return 1.0
    if re.match(r"^\s*RETURN\s+(1|0|true|false|'[^']*'|\"[^\"]*\"|\d[\d,.]*)\s*;?\s*$",
                t, re.IGNORECASE):
        return 1.0
    return 0.0


def garbage_score(pred: str) -> float:
    """0..2: non-ASCII load + bracket imbalance + extra statements after ';'."""
    t = str(pred or "")
    load = min(1.0, sum(1 for ch in t if ord(ch) > 127) / max(len(t), 1) * 8)
    brackets = min(1.0, 0.25 * (abs(t.count("(") - t.count(")"))
                                + abs(t.count("{") - t.count("}"))
                                + abs(t.count("[") - t.count("]"))))
    stmts = [s for s in t.split(";") if s.strip()]
    trailing = 0.0
    if len(stmts) > 1 and any(re.search(r"\b(MATCH|RETURN|CREATE|MERGE|CALL|WITH)\b", s, re.IGNORECASE)
                              for s in stmts[1:]):
        trailing = 1.0
    return min(2.0, load + brackets + trailing)


def compute_reward(pred_cypher: str, gold_cypher: str, neo4j_client,
                   database: str, gold_rows=None, input_text: str = "",
                   question: str = "") -> dict:
    """Execute pred (+ gold unless gold_rows given) on `database`.

    gold_rows: precomputed list-of-tuples for the gold query (preferred in
    training loops to avoid re-executing gold every step).
    Returns execution + structural/schema/entity terms and penalties.
    """
    pred = str(pred_cypher or "").strip().rstrip(";").strip()
    gold = str(gold_cypher or "").strip().rstrip(";").strip()
    ok, reason = shape_ok(pred, gold)
    if not ok:
        return {"branch": "shape_fail", "reason": reason,
                "composite": -3.0, "jaccard": 0.0, "exec_ok": False,
                "structure": 0.0, "schema": 0.0, "entity": 0.0,
                "garbage": 0.0, "trivial": 0.0}

    if gold_rows is None:
        gold_rows = neo4j_client.execute(gold, database=database)
        if gold_rows is None:
            return {"branch": "gold_exec_fail", "composite": 0.0,
                    "jaccard": 0.0, "exec_ok": False,
                    "structure": 0.0, "schema": 0.0, "entity": 0.0,
                    "garbage": 0.0, "trivial": 0.0}

    pred_rows = neo4j_client.execute(pred, database=database)
    exec_ok = pred_rows is not None
    # A failed execution NEVER matches - even when gold is legitimately empty
    # (otherwise broken queries score perfect on empty golds: reward hacking).
    jac = result_jaccard(gold_rows, pred_rows) if exec_ok else 0.0

    exact = normalize_cypher(pred) == normalize_cypher(gold)
    if exact and (exec_ok or not gold_rows):
        branch, base, jac = "exact_match", 3.0, 1.0
    elif exec_ok and jac > 0:
        branch, base = "partial_match", 0.5
    elif exec_ok:
        branch, base = "executed_no_overlap", 0.2
    else:
        branch, base = "exec_fail", -3.0

    struct = structure_score(pred, gold)
    pl, pr, pp, _ = parse_elements(pred)
    schema = schema_score(input_text, pl, pr, pp) if input_text else 0.0
    if not question and input_text and "question:" in str(input_text):
        question = str(input_text).split("question:", 1)[1]
    entity = entity_score(question, pred)
    garbage = garbage_score(pred)
    trivial = triviality(pred)
    composite = max(-6.0, min(6.0, base + 2.0 * jac + 0.8 * struct["composite"]
                              + 0.3 * schema + 0.2 * entity
                              - 0.5 * garbage - 0.3 * trivial))
    try:
        exec_set = frozenset(tuple(r) for r in (pred_rows or [])) if exec_ok else None
    except TypeError:
        exec_set = None
    return {"branch": branch, "composite": composite, "jaccard": jac,
            "exact": exact, "exec_ok": exec_ok, "exec_set": exec_set,
            "structure": round(struct["composite"], 4),
            "label_f1": round(struct["label_f1"], 4),
            "rel_f1": round(struct["rel_f1"], 4),
            "prop_f1": round(struct["prop_f1"], 4),
            "clause_f1": round(struct["clause_f1"], 4),
            "schema": round(schema, 4), "entity": round(entity, 4),
            "garbage": round(garbage, 4), "trivial": trivial}
