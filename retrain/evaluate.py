"""Shared generation eval. Used by BOTH training-time eval and standalone
testing, so numbers are comparable.

Every execution is routed to the row's own database (never a global default),
and execution metrics are honest Jaccard statistics, never raw-reward averages:
  execution_accuracy = fraction of samples with result-Jaccard >= threshold
  mean_result_jaccard = mean over samples of the Jaccard itself
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import TrainerCallback

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
    # checkpointing is backward-only machinery; under inference_mode it only
    # forces use_cache=False (O(n^2) decode). Disable for the loop.
    had_ckpt = False
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            had_ckpt = True
    except Exception:
        had_ckpt = False
    try:
        _use_cache = model.config.use_cache
    except Exception:
        _use_cache = None
    try:
        model.config.use_cache = True
    except Exception:
        pass
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

    try:
        if _use_cache is not None:
            model.config.use_cache = _use_cache
        if had_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass

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
        # eval records go to a LABELED SIBLING file; the training stream is
        # never blanked (rotation used to archive+wipe it mid-run, "losing"
        # all RL records at the end of every train).
        base, ext = os.path.splitext(rolling_jsonl_path)
        eval_path = f"{base}.{label}{ext or '.jsonl'}"
        append_rolling_records(
            eval_path, rolling_json_max_records, phase_name, label,
            decoded_preds, decoded_golds, dbs, inputs,
            jaccards if neo4j_client is not None else None)
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
            f.flush()  # visible to tail -Wait immediately
    snap = path + ".snapshot.json"
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(snap, "w", encoding="utf-8") as f:
            json.dump([json.loads(x) for x in lines[-max_records:]], f, indent=1)
    except Exception:
        pass


class ExecBestCallback(TrainerCallback):
    """Best-model selection SOLELY on execution accuracy.

    At every in-training eval, runs generation eval on a FIXED val subset
    (same indices each time, so scores are comparable) and keeps the adapter
    with the highest execution_accuracy (ties broken by mean_result_jaccard).
    Requires a Neo4j client; without one it stays inert.
    """

    def __init__(self, tokenizer, val_dataset, collator, out_dir: str,
                 neo4j_client=None, exec_samples: int = 100,
                 threshold: float = 0.85, max_target_length: int = 320,
                 eval_batch_size: int = 2, seed: int = 42,
                 upload_repo: str = "", upload_token: str = "",
                 upload_prefix: str = "", phase_name: str = ""):
        self.tokenizer = tokenizer
        self.collator = collator
        self.best_dir = os.path.join(out_dir, "best_exec")
        self.neo4j_client = neo4j_client
        self.threshold = threshold
        self.max_target_length = max_target_length
        self.eval_batch_size = eval_batch_size
        self.upload_repo = upload_repo
        self.upload_token = upload_token
        self.upload_prefix = upload_prefix.strip("/") if upload_prefix else ""
        self.phase_name = phase_name
        n = len(val_dataset)
        k = min(exec_samples, n) if exec_samples > 0 else n
        rng = np.random.default_rng(seed)
        self.subset_idx = sorted(rng.choice(n, size=k, replace=False).tolist())
        self.subset = Subset(val_dataset, self.subset_idx)
        self.best_acc = None
        self.best_jac = None
        self.trainer_ref = None

    def _score(self, metrics: dict):
        return (metrics.get("execution_accuracy", 0.0),
                metrics.get("mean_result_jaccard", 0.0))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.neo4j_client is None:
            return
        trainer = self.trainer_ref or kwargs.get("trainer")
        if trainer is None:
            return
        out_dir = os.path.join(os.path.dirname(os.path.abspath(self.best_dir)),
                               "exec_eval_tmp")
        m = run_generation_eval(
            trainer.model, self.subset, self.tokenizer, self.collator,
            self.max_target_length, os.path.join(out_dir, "metrics.json"),
            label=f"exec@{state.global_step}", eval_batch_size=self.eval_batch_size,
            neo4j_client=self.neo4j_client, exec_jaccard_threshold=self.threshold,
            phase_name=self.phase_name)
        if not m:
            return
        acc, jac = self._score(m)
        better = (self.best_acc is None or acc > self.best_acc
                  or (acc == self.best_acc and jac > (self.best_jac or 0.0)))
        print(f"  [exec-best] step={state.global_step} acc={acc:.4f} "
              f"mean_jac={jac:.4f} best={self.best_acc}", flush=True)
        if better:
            self.best_acc, self.best_jac = acc, jac
            os.makedirs(self.best_dir, exist_ok=True)
            trainer.save_model(self.best_dir)
            print(f"  [exec-best] NEW BEST -> {self.best_dir}", flush=True)
            if self.upload_repo:
                try:
                    from .sync import upload_dir
                    path = ((self.upload_prefix + "/best_exec").strip("/")
                            if self.upload_prefix else "best_exec")
                    upload_dir(self.best_dir, self.upload_repo, path,
                               token=self.upload_token)
                except Exception as e:
                    print(f"  [exec-best] hub upload failed (continuing): {e}", flush=True)
