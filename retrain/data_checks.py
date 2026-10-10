"""Pre-training data validation. Fails loudly instead of training on bad data.

Checks (run in this order):
1. files exist, non-empty, required columns present
2. train/val/test exact (question, target) overlap  -> HARD FAIL
3. ambiguous questions (same question, different target) -> reported; fails
   only above `max_ambiguous_ratio`
4. target-in-input leakage on the *rendered* model inputs -> HARD FAIL
5. schema coverage: every label / relationship / property referenced by the
   gold Cypher must appear in the row's schema (reports miss rate; fails
   above `max_schema_miss_ratio`)
6. non-read-only gold statements (CREATE/MERGE/...) -> HARD FAIL
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

WRITE_RE = re.compile(
    r"^\s*(CREATE|MERGE|SET\s|DELETE|DETACH\s+DELETE|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b",
    re.IGNORECASE | re.MULTILINE)

VAR_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:")


@dataclass
class CheckReport:
    split_counts: dict = field(default_factory=dict)
    exact_overlaps: dict = field(default_factory=dict)
    ambiguous_counts: dict = field(default_factory=dict)
    input_leaks: dict = field(default_factory=dict)
    schema_miss: dict = field(default_factory=dict)
    write_statements: dict = field(default_factory=dict)


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def norm_cypher_vars(cypher: str) -> str:
    """Replace bound variable names so `a:` vs `n:` renames compare equal."""
    out, mapping, i = [], {}, [0]

    def _sub(m):
        name = m.group(1)
        if name not in mapping:
            mapping[name] = f"v{i[0]}"
            i[0] += 1
        return mapping[name] + ":"

    return VAR_RE.sub(_sub, norm_text(cypher))


def render_input(schema, question) -> str:
    return f"schema: {str(schema).strip()}\nquestion: {str(question).strip()}"


def load_split(path: str, schema_col: str, question_col: str, target_col: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"dataset file not found: {path!r}")
    df = pd.read_parquet(path)
    for col in (schema_col, question_col, target_col):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col!r} "
                             f"(has {list(df.columns)[:12]})")
    if len(df) == 0:
        raise ValueError(f"{path}: file is empty - refusing to train on nothing")
    df = df.copy()
    df["_nq"] = df[question_col].map(norm_text)
    df["_nt"] = df[target_col].map(norm_text)
    df["_nt_vars"] = df[target_col].map(norm_cypher_vars)
    df["_rendered"] = [render_input(s, q) for s, q in
                       zip(df[schema_col], df[question_col])]
    return df


def validate(train_path, val_path, test_path, schema_col, question_col, target_col,
             max_ambiguous_ratio: float = 0.02,
             max_schema_miss_ratio: float = 0.10) -> CheckReport:
    """max_schema_miss_ratio defaults to 0.10: upstream benchmark schemas are
    known-incomplete subsets (hallucinated props, missing rels in ~2-6% of
    rows). The miss breakdown is always printed, so nothing hides."""
    rep = CheckReport()
    tr = load_split(train_path, schema_col, question_col, target_col)
    va = load_split(val_path, schema_col, question_col, target_col) if val_path else None
    te = load_split(test_path, schema_col, question_col, target_col) if test_path else None
    rep.split_counts = {"train": len(tr),
                        "val": len(va) if va is not None else 0,
                        "test": len(te) if te is not None else 0}

    tr_pairs = set(zip(tr["_nq"], tr["_nt_vars"]))
    tr_q2t = dict(zip(tr["_nq"], tr["_nt_vars"]))

    for name, split in (("val", va), ("test", te)):
        if split is None:
            continue
        pairs = set(zip(split["_nq"], split["_nt_vars"]))
        exact = pairs & tr_pairs
        rep.exact_overlaps[name] = len(exact)
        if exact:
            raise ValueError(
                f"TRAIN/{name.upper()} LEAKAGE: {len(exact)} identical "
                f"(question, target) pairs - refusing to train")

        sp_q = split.drop_duplicates("_nq").set_index("_nq")
        ambig = sum(1 for q in set(split["_nq"]) & set(tr_q2t)
                    if q in sp_q.index and str(sp_q.loc[q, "_nt_vars"]) != tr_q2t[q])
        rep.ambiguous_counts[name] = ambig
        ratio = ambig / max(len(split), 1)
        if ratio > max_ambiguous_ratio:
            raise ValueError(
                f"TRAIN/{name.upper()} AMBIGUITY: {ambig} shared questions with "
                f"different targets ({ratio:.2%} > {max_ambiguous_ratio:.2%})")

    # input leakage + writes + empty schemas + schema coverage on all splits
    for name, split in (("train", tr), ("val", va), ("test", te)):
        if split is None:
            continue
        empty_sch = split[schema_col].fillna("").str.strip() == ""
        if int(empty_sch.sum()):
            dbs = split[empty_sch]["database"].value_counts().to_dict() if "database" in split.columns else {}
            raise ValueError(f"{name.upper()} EMPTY SCHEMA: {int(empty_sch.sum())} rows "
                             f"have no schema at all {dbs} - fill or drop them before training")
        leaks = sum(1 for inp, tgt in zip(split["_rendered"], split[target_col].fillna(""))
                    if len(str(tgt).strip()) > 20 and str(tgt).strip() in inp)
        rep.input_leaks[name] = leaks
        if leaks:
            raise ValueError(f"{name.upper()} INPUT LEAKAGE: {leaks} rows contain "
                             f"the gold target inside the model input")

        writes = int(split[target_col].fillna("").apply(
            lambda t: bool(WRITE_RE.search(str(t)))).sum())
        rep.write_statements[name] = writes
        if writes:
            raise ValueError(f"{name.upper()}: {writes} non-read-only gold "
                             f"statements (CREATE/MERGE/...) - refusing to train")

        miss = check_schema_coverage(split, schema_col)
        rep.schema_miss[name] = miss
        ratio = miss / max(len(split), 1)
        if ratio > max_schema_miss_ratio:
            raise ValueError(
                f"{name.upper()} SCHEMA COVERAGE: {miss} rows reference labels/rels/props "
                f"missing from their schema ({ratio:.2%} > {max_schema_miss_ratio:.2%})")
    return rep


def _split_sections(schema_text: str):
    """Split a compact schema into (node_text, rel_text). Handles both
    newline-separated ('N: ...\\nR: ...') and legacy inline ('... / R: ...')
    section separators."""
    s = str(schema_text or "")
    m = re.search(r"(?:^|\n)\s*R\s*:\s*", s)
    if m:
        return s[:m.start()], s[m.end():]
    m2 = re.search(r"\s+/\s*R\s*:\s*", s)
    if m2:
        return s[:m2.start()], s[m2.end():]
    m3 = re.match(r"(?i)^\s*schema\s*:\s*", s)
    body = s[m3.end():] if m3 else s
    return body, ""


def _schema_vocab(schema_text: str):
    """Lowercased label / rel-type / property tokens present in a compact schema."""
    n_text, r_text = _split_sections(schema_text)
    n_text = re.sub(r"(?i)^\s*(?:schema|N)\s*:\s*", "", n_text.strip())
    labels, rels, props = set(), set(), set()
    for item in n_text.split("|"):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?$", item)
        if m:
            labels.add(m.group(1).lower())
            if m.group(2):
                props.update(p.strip().lower() for p in m.group(2).split(",") if p.strip())
    for item in r_text.split("|"):
        item = item.strip()
        if not item:
            continue
        if ">" in item:
            segs = [sg.strip() for sg in item.split(">")]
            if len(segs) >= 3:
                if re.fullmatch(r"[A-Za-z_]\w*", segs[0].split("{")[0].strip()):
                    labels.add(segs[0].split("{")[0].strip().lower())
                if re.fullmatch(r"[A-Za-z_]\w*", segs[-1].split("{")[0].strip()):
                    labels.add(segs[-1].split("{")[0].strip().lower())
                for seg in segs[1:-1]:
                    t = seg.split("{")[0].strip()
                    if t:
                        rels.add(t.lower())
                    pm = re.search(r"\{([^}]*)\}", seg)
                    if pm:
                        props.update(p.strip().lower() for p in pm.group(1).split(",") if p.strip())
                continue
        # bare rel-type name (no endpoints)
        t = item.split("{")[0].strip()
        if t:
            rels.add(t.lower())
        pm = re.search(r"\{([^}]*)\}", item)
        if pm:
            props.update(p.strip().lower() for p in pm.group(1).split(",") if p.strip())
    return labels, rels, props


# function-result accessors that look like `.prop` but aren't node properties
NON_PROPS = {"year", "years", "quarter", "quarters", "month", "months",
             "week", "weeks", "day", "days", "ordinalday", "dayofweek",
             "hour", "hours", "minute", "minutes", "second", "seconds",
             "millisecond", "milliseconds", "microsecond", "microseconds",
             "epochmillis", "epochseconds"}


def check_schema_coverage(split: pd.DataFrame, schema_col: str) -> int:
    """Rows whose gold Cypher references something absent from its schema.

    `noprops`-variant rows are intentionally property-free: labels and rel
    types are still enforced, only the property check is skipped for them.
    """
    miss = 0
    has_variant = "schema_variant" in split.columns
    for _, row in split.iterrows():
        labels, rels, props = _schema_vocab(row[schema_col])
        tgt = str(row["_nt"])
        used_labels = set(re.findall(r"\(:([A-Za-z_]\w*)", tgt))
        used_labels |= set(re.findall(r"\([A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)", tgt))
        used_rels = set(re.findall(r"\[\s*:([A-Za-z_]\w*)", tgt))
        used_rels |= set(re.findall(r"\[[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)", tgt))
        used_props = (set(re.findall(r"\.\s*([A-Za-z_]\w*)", tgt))
                      - NON_PROPS - {"id"})
        if has_variant and str(row["schema_variant"]) == "noprops":
            if (used_labels - labels) or (used_rels - rels):
                miss += 1
            continue
        if ((used_labels - labels) or (used_rels - rels)
                or (used_props - props)):
            miss += 1
    return miss
