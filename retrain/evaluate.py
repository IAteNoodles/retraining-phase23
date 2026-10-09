"""Shared generation eval. Used by BOTH training-time eval and standalone
testing, so numbers are comparable.

Every execution is routed to the row's own database (never a global default),
and execution metrics are honest Jaccard statistics, not raw-reward averages:
  execution_accuracy = fraction of samples with result-Jaccard >= threshold
  mean_result_jaccard = mean over samples of the Jaccard itself
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .rewards import normalize_cypher, result_jaccard


def _levenshtein(a, b) -> int:
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_generation_eval(model, dataset, tokenizer, collator, max_target_length,
                        output_path, label="eval", eval_batch_size=8,
                        neo4j_client=None, exec_jaccard_threshold=0.85,
                        max_samples=0, device="cuda",
                        rolling_jsonl_path="", rolling_json_max_records=1000,
                        phase_name=""):
    from torch.utils.data import Subset
    if max_samples and max_samples > 0 and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
    if len(dataset) == 0:
        print(f"{label}: empty dataset, skipping")
        return None

    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False,
                        collate_fn=collator)
    preds, golds, dbs, gold_answers, inputs = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating ({label})"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_target_length, do_sample=False, num_beams=4,
                length_penalty=0.6)
            preds.append(generated.cpu())
            golds.append(batch["labels"])
            n = len(batch["labels"])
            dbs.extend(batch.get("database", [None] * n))
            gold_answers.extend(batch.get("gold_answer", [None] * n))
            inputs.extend(batch.get("input_text", [None] * n))
            del input_ids, attention_mask, generated

    max_len = max(t.shape[1] for t in preds)
    P = torch.cat([torch.nn.functional.pad(t, (0, max_len - t.shape[1]),
                                           value=tokenizer.pad_token_id) for t in preds], dim=0)
    max_g = max(t.shape[1] for t in golds)
    G = torch.cat([torch.nn.functional.pad(t, (0, max_g - t.shape[1]), value=-100)
                   for t in golds], dim=0)
    decoded_preds = tokenizer.batch_decode(P, skip_special_tokens=True)
    g_np = np.where(G.numpy() != -100, G.numpy(), tokenizer.pad_token_id)
    decoded_golds = tokenizer.batch_decode(g_np, skip_special_tokens=True)

    exact = sum(normalize_cypher(p) == normalize_cypher(g)
                for p, g in zip(decoded_preds, decoded_golds))
    total = len(decoded_preds)
    ed_sum, tok_sum = 0, 0
    for p, g in zip(decoded_preds, decoded_golds):
        pt, gt = normalize_cypher(p).split(), normalize_cypher(g).split()
        ed_sum += _levenshtein(pt, gt)
        tok_sum += len(gt)

    metrics = {
        "exact_match": round(exact / max(total, 1), 4),
        "token_error_rate": round(ed_sum / max(tok_sum, 1), 4),
        "total_samples": total,
    }

    if neo4j_client is not None:
        jaccards, branches = [], Counter()
        for pred, gold, db, ga in zip(decoded_preds, decoded_golds, dbs, gold_answers):
            gold_rows = None
            if ga:
                try:
                    gold_rows = [tuple(r) for r in
                                 (json.loads(ga) if isinstance(ga, str) else ga)]
                except Exception:
                    gold_rows = None
            if gold_rows is None:
                gr = neo4j_client.execute(gold.strip(), database=db)
                gold_rows = [] if gr is None else [tuple(r) for r in gr]
            pr = neo4j_client.execute(pred.strip(), database=db)
            j = result_jaccard(gold_rows, [] if pr is None else [tuple(r) for r in pr])
            jaccards.append(j)
            branches["exec_ok" if pr is not None else "exec_fail"] += 1
        metrics["execution_accuracy"] = round(
            sum(1 for j in jaccards if j >= exec_jaccard_threshold) / max(total, 1), 4)
        metrics["mean_result_jaccard"] = round(sum(jaccards) / max(total, 1), 4)
        metrics["exec_branches"] = dict(branches)

    print(f"\n{label} metrics: {metrics}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    if rolling_jsonl_path:
        append_rolling_records(
            rolling_jsonl_path, rolling_json_max_records, phase_name, label,
            decoded_preds, decoded_golds, dbs, inputs,
            jaccards if neo4j_client is not None else None)
        rotate_rolling_log(rolling_jsonl_path, label)
    return metrics


def rotate_rolling_log(path, label):
    """Archive the current rolling log next to itself (timestamped), then
    reset it to blank so the next eval starts fresh."""
    import datetime
    import shutil
    if not path or not os.path.exists(path):
        return ""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base, ext = os.path.splitext(path)
    archived = f"{base}.{label}.{stamp}{ext or '.jsonl'}"
    shutil.copyfile(path, archived)
    open(path, "w").close()  # blank for the next round
    print(f"  [rolling] archived {path} -> {archived}; reset to blank")
    return archived


def append_rolling_records(path, max_records, phase_name, label,
                           preds, golds, dbs, inputs, jaccards=None):
    """Append per-sample records to a JSONL trace + capped JSON snapshot."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    records = []
    for i, (p, g) in enumerate(zip(preds, golds)):
        records.append({
            "time": now, "phase": phase_name, "eval_label": label,
            "input": inputs[i] if i < len(inputs) else None,
            "output": p.strip(), "gold": g.strip(),
            "jaccard": round(jaccards[i], 4) if jaccards is not None else None,
            "database": dbs[i] if i < len(dbs) else None,
        })
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    snap = path + ".snapshot.json"
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(snap, "w", encoding="utf-8") as f:
            json.dump([json.loads(x) for x in lines[-max_records:]], f, indent=1)
    except Exception:
        pass
