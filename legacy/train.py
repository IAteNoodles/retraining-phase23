import argparse
from contextlib import nullcontext
import datetime
import gc
import inspect
import json
import os
import re
import requests
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    LogitsProcessor,
    LogitsProcessorList,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dataset import Text2CypherDataset, Text2CypherCollator

try:
    # Import only the symbols we actually use. Some optional/deprecated symbols
    # (e.g. Repository/CommitScheduler) may be missing in newer hub versions.
    from huggingface_hub import HfApi, login, snapshot_download, upload_file, upload_folder
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False
    print(f"[HF] huggingface_hub unavailable: {e}")


def get_vram_info():
    if not torch.cuda.is_available():
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "reserved_gb": 0}
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    used = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    free = total - reserved
    return {"total_gb": total, "used_gb": used, "free_gb": free, "reserved_gb": reserved}


def auto_tune_params(free_vram_gb):
    if free_vram_gb >= 20:
        return {
            "batch_size": 8, "grad_accum": 2, "max_in": 1024, "lora_r": 32, "max_tgt": 320,
            "reason": f"{free_vram_gb:.1f}GB >= 20GB: ample VRAM, max LoRA rank & batch",
        }
    elif free_vram_gb >= 10:
        return {
            "batch_size": 8, "grad_accum": 2, "max_in": 1024, "lora_r": 16, "max_tgt": 320,
            "reason": f"{free_vram_gb:.1f}GB >= 10GB: high VRAM, full seq len, reduced LoRA",
        }
    elif free_vram_gb >= 5:
        return {
            "batch_size": 4, "grad_accum": 8, "max_in": 1024, "lora_r": 64, "max_tgt": 320,
            "reason": f"{free_vram_gb:.1f}GB >= 5GB: checkpointed, eff batch=32, ~28h",
        }
    elif free_vram_gb >= 3.5:
        return {
            "batch_size": 2, "grad_accum": 8, "max_in": 384, "lora_r": 16, "max_tgt": 192,
            "reason": f"{free_vram_gb:.1f}GB >= 3.5GB: reduced seq len, LoRA r=16",
        }
    elif free_vram_gb >= 2:
        return {
            "batch_size": 1, "grad_accum": 16, "max_in": 256, "lora_r": 8, "max_tgt": 128,
            "reason": f"{free_vram_gb:.1f}GB >= 2GB: tight VRAM, small batch, short seq",
        }
    else:
        return {
            "batch_size": 1, "grad_accum": 16, "max_in": 256, "lora_r": 4, "max_tgt": 128,
            "reason": f"{free_vram_gb:.1f}GB < 2GB: minimal VRAM, smallest viable config",
        }


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = sorted(
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ],
        key=lambda x: int(x.split("-")[1]),
    )
    if checkpoints:
        return os.path.join(output_dir, checkpoints[-1])
    return None


def prompt_resume_checkpoint(local_checkpoint, hf_repo_id, hf_token, output_dir):
    """
    Prompt user to choose checkpoint source with menu-based interface.
    
    Checks both local and HF Hub checkpoints, then presents menu:
      1) Resume from HF checkpoint
      2) Resume from local checkpoint
      3) Start fresh
    
    Handles non-interactive mode (Kaggle) by defaulting to: HF > Local > Fresh
    
    Returns: (checkpoint_path, source)  # source: 'local', 'hf', or None
    """
    hf_checkpoint = None
    local_ckpt_name = None
    hf_ckpt_name = None
    
    # 1. Check local checkpoints
    if local_checkpoint:
        local_ckpt_name = os.path.basename(local_checkpoint)
    
    # 2. Check HF Hub checkpoints
    if hf_repo_id:
        print(f"Checking Hugging Face Hub: {hf_repo_id}...")
        hf_checkpoint = download_checkpoint_from_hub(hf_repo_id, output_dir)
        if hf_checkpoint:
            hf_ckpt_name = os.path.basename(hf_checkpoint)
    
    # 3. Present menu and get user choice
    print(f"\n{'='*70}")
    print("CHECKPOINT RESUME OPTIONS")
    print(f"{'='*70}")
    
    menu_items = []
    if hf_ckpt_name:
        menu_items.append((1, f"Resume from HF Hub", hf_ckpt_name, hf_checkpoint, 'hf'))
    if local_ckpt_name:
        menu_items.append((2, f"Resume from Local", local_ckpt_name, local_checkpoint, 'local'))
    menu_items.append((3, "Start Fresh", None, None, None))
    
    # Print menu
    for num, label, ckpt_name, _, _ in menu_items:
        if ckpt_name:
            print(f"  {num}) {label}: {ckpt_name}")
        else:
            print(f"  {num}) {label}")
    
    # Determine default choice
    if hf_ckpt_name:
        default_choice = 1
        default_desc = "HF checkpoint"
    elif local_ckpt_name:
        default_choice = 2
        default_desc = "local checkpoint"
    else:
        default_choice = 3
        default_desc = "fresh start"
    
    print(f"\nDefault: {default_choice} ({default_desc})")
    
    # Try to get user input, fall back to default if non-interactive
    try:
        response = input("Enter choice (1-3): ").strip()
        if not response:
            choice = default_choice
            print(f"[No input provided, using default: {choice}]")
        else:
            try:
                choice = int(response)
            except ValueError:
                choice = default_choice
                print(f"[Invalid input, using default: {choice}]")
    except EOFError:
        choice = default_choice
        print(f"[Non-interactive mode, using default: {choice}]")
    
    # Execute choice
    checkpoint_to_use = None
    source = None
    
    for num, label, ckpt_name, ckpt_path, src in menu_items:
        if num == choice:
            if ckpt_path:
                checkpoint_to_use = ckpt_path
                source = src
                print(f"\n✓ {label}: {ckpt_name}")
            else:
                print(f"\n✓ Starting fresh")
            break
    
    if checkpoint_to_use is None:
        print("\nNo checkpoint selected, starting fresh.")
    
    print(f"{'='*70}\n")
    return checkpoint_to_use, source


def normalize_cypher_eval(text):
    parts = []
    i = 0
    while i < len(text):
        if text[i] in ("'", '"'):
            quote = text[i]
            j = i + 1
            while j < len(text) and text[j] != quote:
                if text[j] == '\\':
                    j += 1
                j += 1
            if j < len(text):
                j += 1
            parts.append(text[i:j])
            i = j
        else:
            parts.append(text[i].lower())
            i += 1
    return ''.join(parts)


def _append_rolling_generation_records(records, rolling_jsonl_path, max_json_records=1000):
    if not rolling_jsonl_path or not records:
        return None
    path = Path(rolling_jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    rolling_json_path = path.with_suffix(".json")
    recent = []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-max_json_records:]
        recent = [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"[Rolling Eval] Could not build rolling JSON snapshot: {e}")
    with rolling_json_path.open("w", encoding="utf-8") as f:
        json.dump(recent, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Rolling Eval] Appended {len(records)} rows to {path}; snapshot: {rolling_json_path}")
    return rolling_json_path


def build_roll_rows(step, best_texts, gold_texts, rewards, branches, databases, input_texts):
    """Pure helper: RL best-per-sample rows for the rolling stream.

    All args are plain lists (no tensors) so this is unit-testable CPU-only.
    rewards may be floats or .item()-ables.
    """
    rows = []
    for i in range(len(best_texts)):
        r = rewards[i]
        rows.append({
            "step": int(step),
            "input": input_texts[i] if i < len(input_texts) else None,
            "pred": str(best_texts[i]).strip(),
            "gold": str(gold_texts[i]).strip(),
            "reward": float(r.item() if hasattr(r, "item") else r),
            "branch": branches[i] if i < len(branches) else "unknown",
            "database": databases[i] if i < len(databases) else None,
        })
    return rows


def run_generation_eval(eval_model, dataset, tokenizer, collator, max_target_length, output_path, label="eval", eval_batch_size=8, neo4j_client=None, phase_name="eval", rolling_jsonl_path=None, rolling_json_max_records=1000):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if len(dataset) == 0:
        print(f"{label} dataset empty, skipping")
        return None

    eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator)
    all_preds = []
    all_labels = []
    all_databases = []
    all_gold_answers = []
    all_input_texts = []

    try:
        with torch.inference_mode():
            _gen_device = next(eval_model.parameters()).device
            for i, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating ({label})")):
                input_ids = batch["input_ids"].to(_gen_device)
                attention_mask = batch["attention_mask"].to(_gen_device)
                labels = batch["labels"]

                generated = eval_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_target_length,
                    do_sample=False,
                    num_beams=4,
                    length_penalty=0.6,
                )
                torch.cuda.synchronize()

                all_preds.append(generated.cpu())
                all_labels.append(labels)
                all_databases.extend(batch.get("database", [None] * len(labels)))
                all_gold_answers.extend(batch.get("gold_answer", [None] * len(labels)))
                all_input_texts.extend(batch.get("input_text", [None] * len(labels)))
                del input_ids, attention_mask, generated

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(eval_loader)}] CUDA: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                    torch.cuda.empty_cache()
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\n*** {label} eval failed at batch index {i} ***")
        print(f"CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
        if all_preds:
            partial_max_len = max(t.shape[1] for t in all_preds)
            partial_preds = torch.cat([torch.nn.functional.pad(t, (0, partial_max_len - t.shape[1]), value=tokenizer.pad_token_id) for t in all_preds], dim=0)
            partial_max_labels = max(t.shape[1] for t in all_labels)
            partial_labels = torch.cat([torch.nn.functional.pad(t, (0, partial_max_labels - t.shape[1]), value=-100) for t in all_labels], dim=0)
            partial_decoded_preds = tokenizer.batch_decode(partial_preds, skip_special_tokens=True)
            partial_labels_np = np.where(partial_labels.numpy() != -100, partial_labels.numpy(), tokenizer.pad_token_id)
            partial_decoded_labels = tokenizer.batch_decode(partial_labels_np, skip_special_tokens=True)
            partial_metrics = {
                "partial": True,
                "label": label,
                "samples_processed": len(partial_decoded_preds),
                "exact_match": round(sum(p.strip() == l.strip() for p, l in zip(partial_decoded_preds, partial_decoded_labels)) / max(len(partial_decoded_preds), 1), 4),
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(partial_metrics, f, indent=2, default=str)
            print(f"Partial {label} metrics saved ({len(partial_decoded_preds)} samples) to {output_path}")
        raise

    max_len = max(t.shape[1] for t in all_preds)
    preds = torch.cat([torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=tokenizer.pad_token_id) for t in all_preds], dim=0)
    max_labels = max(t.shape[1] for t in all_labels)
    labels = torch.cat([torch.nn.functional.pad(t, (0, max_labels - t.shape[1]), value=-100) for t in all_labels], dim=0)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_np = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)

    exact_matches = sum(normalize_cypher_eval(p.strip()) == normalize_cypher_eval(l.strip()) for p, l in zip(decoded_preds, decoded_labels))
    total = len(decoded_preds)
    token_correct = 0
    token_total = 0
    total_edit_dist = 0
    total_label_tokens = 0
    for pred_s, label_s in zip(decoded_preds, decoded_labels):
        pred_s, label_s = normalize_cypher_eval(pred_s.strip()), normalize_cypher_eval(label_s.strip())
        pred_tokens = pred_s.split()
        label_tokens = label_s.split()
        min_len = min(len(pred_tokens), len(label_tokens))
        for j in range(min_len):
            if pred_tokens[j] == label_tokens[j]:
                token_correct += 1
        token_total += max(len(pred_tokens), len(label_tokens))
        ed = token_levenstein(pred_tokens, label_tokens)
        total_edit_dist += ed
        total_label_tokens += len(label_tokens)

    token_edit_distance = total_edit_dist / max(total, 1)
    token_error_rate = total_edit_dist / max(total_label_tokens, 1)

    metrics = {
        "exact_match": round(exact_matches / max(total, 1), 4),
        "token_accuracy": round(token_correct / max(token_total, 1), 4),
        "token_edit_distance": round(token_edit_distance, 4),
        "token_error_rate": round(token_error_rate, 4),
        "total_samples": total,
    }

    if neo4j_client is not None:
        exec_rewards = []
        exec_exact = 0
        exec_infos = []
        for i, (pred, gold) in enumerate(zip(decoded_preds, decoded_labels)):
            r, _, info = compute_reward(
                pred.strip(),
                gold.strip(),
                neo4j_client,
                gold_answer_json=all_gold_answers[i] if i < len(all_gold_answers) else None,
                database=all_databases[i] if i < len(all_databases) else None,
                input_text=all_input_texts[i] if i < len(all_input_texts) else None,
            )
            exec_rewards.append(r)
            exec_infos.append(info)
            if r >= 0.85:
                exec_exact += 1
        metrics["execution_accuracy"] = round(sum(exec_rewards) / max(len(exec_rewards), 1), 4)
        metrics["execution_exact_match"] = round(exec_exact / max(len(exec_rewards), 1), 4)
        print(f"  Execution accuracy: {metrics['execution_accuracy']:.4f}, exact: {metrics['execution_exact_match']:.4f}")
    else:
        exec_rewards = [None] * len(decoded_preds)
        exec_infos = [{} for _ in decoded_preds]

    if rolling_jsonl_path:
        now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        rolling_records = []
        for i, (pred, gold) in enumerate(zip(decoded_preds, decoded_labels)):
            rolling_records.append({
                "time": now,
                "phase": phase_name,
                "eval_label": label,
                "input": all_input_texts[i] if i < len(all_input_texts) else None,
                "output": pred.strip(),
                "gold": gold.strip(),
                "reward": round(exec_rewards[i], 4) if exec_rewards[i] is not None else None,
                "database": all_databases[i] if i < len(all_databases) else None,
                "reward_info": exec_infos[i] if i < len(exec_infos) else {},
            })
        _append_rolling_generation_records(rolling_records, rolling_jsonl_path, max_json_records=rolling_json_max_records)

    print(f"\n{label} eval metrics (with generation): {metrics}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics


def token_levenstein(pred_tokens, label_tokens):
    m, n = len(pred_tokens), len(label_tokens)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if pred_tokens[i - 1] == label_tokens[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def compute_metrics(eval_pred, tokenizer, neo4j_client=None):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels_np = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)

    exact_matches = 0
    norm_matches = 0
    total = len(decoded_preds)
    token_correct = 0
    token_total = 0
    total_edit_dist = 0
    total_label_tokens = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        pred_s = pred.strip()
        label_s = label.strip()
        if pred_s == label_s:
            exact_matches += 1

        norm_pred = " ".join(pred_s.lower().split()).rstrip(";").strip()
        norm_label = " ".join(label_s.lower().split()).rstrip(";").strip()
        if norm_pred == norm_label:
            norm_matches += 1

        pred_tokens = pred_s.split()
        label_tokens = label_s.split()
        min_len = min(len(pred_tokens), len(label_tokens))
        for j in range(min_len):
            if pred_tokens[j] == label_tokens[j]:
                token_correct += 1
        token_total += max(len(pred_tokens), len(label_tokens))

        norm_pred_toks = norm_pred.split()
        norm_label_toks = norm_label.split()
        ed = token_levenstein(norm_pred_toks, norm_label_toks)
        total_edit_dist += ed
        total_label_tokens += len(norm_label_toks)

    token_edit_distance = total_edit_dist / max(len(decoded_preds), 1)
    token_error_rate = total_edit_dist / max(total_label_tokens, 1)

    metrics = {
        "exact_match": round(exact_matches / max(total, 1), 4),
        "norm_exact_match": round(norm_matches / max(total, 1), 4),
        "token_accuracy": round(token_correct / max(token_total, 1), 4),
        "token_edit_distance": round(token_edit_distance, 4),
        "token_error_rate": round(token_error_rate, 4),
    }

    if neo4j_client is not None:
        exec_rewards = []
        exec_exact = 0
        for pred, gold in zip(decoded_preds, decoded_labels):
            r = _reward_value(compute_reward(pred.strip(), gold.strip(), neo4j_client))
            exec_rewards.append(r)
            if r >= 0.85:
                exec_exact += 1
        metrics["execution_accuracy"] = round(sum(exec_rewards) / max(len(exec_rewards), 1), 4)
        metrics["execution_exact_match"] = round(exec_exact / max(len(exec_rewards), 1), 4)

    return metrics


class CacheClearEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        return control


class StepCacheClearCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return control


class MinEpochEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience, min_epochs):
        self.patience = patience
        self.min_epochs = min_epochs
        self.patience_counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        if state.epoch < self.min_epochs:
            self.patience_counter = 0
            return control
        metric_name = args.metric_for_best_model
        if metric_name is None:
            metric_name = "eval_loss"
        current = metrics.get(metric_name)
        if current is None:
            return control
        greater_is_better = args.greater_is_better if args.greater_is_better is not None else False
        if self.best_metric is None:
            self.best_metric = float('inf') if not greater_is_better else float('-inf')
        if (greater_is_better and current > self.best_metric) or (not greater_is_better and current < self.best_metric):
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {self.min_epochs:.1f}+ epochs (patience={self.patience}), best={self.best_metric:.4f}")
                control.should_training_stop = True
        return control


class HFHubUploadCallback(TrainerCallback):
    """Upload checkpoints to HuggingFace Hub after each save."""
    def __init__(self, hf_repo_id, hf_token=None, hf_path_prefix=""):
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.hf_path_prefix = hf_path_prefix.strip("/")
        self.last_uploaded = None

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        if not HF_AVAILABLE or not self.hf_repo_id:
            return
        if not is_main_process():
            return  # DDP: rank 0 uploads once; other ranks skip.
        
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_path.exists() and checkpoint_path != self.last_uploaded:
            print(f"\n[HF Upload] Uploading {checkpoint_path.name} to {self.hf_repo_id}...")
            try:
                _upload_folder_resilient(
                    repo_id=self.hf_repo_id,
                    folder_path=str(checkpoint_path),
                    path_in_repo=join_hf_path(self.hf_path_prefix, checkpoint_path.name),
                    token=self.hf_token,
                )
                self.last_uploaded = checkpoint_path
                print(f"[HF Upload] ✓ {checkpoint_path.name} uploaded successfully")
            except Exception as e:
                print(f"[HF Upload] Failed to upload {checkpoint_path.name}: {e}")


class SaveBeforeEvalCallback(TrainerCallback):
    """Save and upload checkpoint right BEFORE evaluation (so uploads happen during eval)."""
    def __init__(self, hf_repo_id, hf_token=None, hf_path_prefix=""):
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.hf_path_prefix = hf_path_prefix.strip("/")
        self.last_saved_step = None

    def on_evaluate(self, args, state, control, **kwargs):
        """Called right before evaluation. Save and upload checkpoint."""
        if not is_main_process():
            return  # DDP: rank 0 uploads once; other ranks skip.
        if self.last_saved_step == state.global_step:
            # Already saved at this step, skip
            return
        
        # Save checkpoint at current step
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            print(f"\n[SaveBeforeEval] Saving checkpoint at step {state.global_step}...")
            # Save/eval should be aligned so this step is materialized by normal save hooks.
            print("[SaveBeforeEval] Waiting for save hook to materialize checkpoint before/at this eval boundary.")
            self.last_saved_step = state.global_step
        else:
            # Already saved, just upload
            if HF_AVAILABLE and self.hf_repo_id:
                print(f"\n[SaveBeforeEval] Uploading {checkpoint_dir.name} to {self.hf_repo_id} during evaluation...")
                try:
                    _upload_folder_resilient(
                        repo_id=self.hf_repo_id,
                        folder_path=str(checkpoint_dir),
                        path_in_repo=join_hf_path(self.hf_path_prefix, checkpoint_dir.name),
                        token=self.hf_token,
                    )
                    print(f"[SaveBeforeEval] ✓ Upload started (running in parallel with eval)")
                except Exception as e:
                    print(f"[SaveBeforeEval] Upload failed: {e}")


def _distributed_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    env_rank = os.environ.get("RANK")
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_rank is not None:
        try:
            return int(env_rank), max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        except ValueError:
            pass
    if env_local_rank is not None:
        try:
            return int(env_local_rank), max(int(os.environ.get("LOCAL_WORLD_SIZE", "1")), 1)
        except ValueError:
            pass
    return 0, 1


def is_main_process():
    rank, _ = _distributed_info()
    return rank == 0


def resolve_precision_mode(mode):
    selected = (mode or "auto").lower()
    cuda_available = torch.cuda.is_available()
    bf16_capable = cuda_available and torch.cuda.is_bf16_supported()

    if selected == "auto":
        use_bf16 = bf16_capable
        use_fp16 = cuda_available and not bf16_capable
    elif selected == "bf16":
        use_bf16 = bf16_capable
        use_fp16 = False
        if not bf16_capable and is_main_process():
            print("Requested precision=bf16 but GPU/runtime reports no bf16 support; falling back to fp16 on CUDA.")
            use_fp16 = cuda_available
    elif selected == "fp16":
        use_bf16 = False
        use_fp16 = cuda_available
    else:
        use_bf16 = False
        use_fp16 = False

    dtype = torch.float32
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16

    return use_bf16, use_fp16, dtype


def is_peft_wrapped_model(model):
    current = model
    for _ in range(8):
        if isinstance(current, PeftModel):
            return True
        next_model = getattr(current, "module", None)
        if next_model is None:
            next_model = getattr(current, "_orig_mod", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    return False


def get_model_float_dtype(model):
    current = model
    for _ in range(8):
        next_model = getattr(current, "module", None)
        if next_model is None:
            next_model = getattr(current, "_orig_mod", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    for param in current.parameters():
        if torch.is_floating_point(param):
            return param.dtype
    return torch.float32


def ensure_trainable_params_fp32(model):
    converted = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            if param.grad is not None and param.grad.dtype in (torch.float16, torch.bfloat16):
                param.grad = param.grad.float()
            converted += 1
    if converted and is_main_process():
        print(f"Upcasted {converted} trainable parameter tensors to fp32 for stable AMP unscale/grad clipping.")
    return converted


def unify_model_dtype(model, dtype, tag="model"):
    """Force every floating param/buffer to one dtype.

    from_pretrained(..., torch_dtype=X) leaves stray dtypes on sharded
    checkpoints (measured: bf16+fp16 mix on merged_base). A mixed model
    crashes bare forward (dtype mismatch) and NaNs under autocast.
    NOTE: unifying does not change range physics — fp16 unified still
    overflows this model; bf16 unified still needs bf16 kernels.
    """
    try:
        dtypes = {p.dtype for p in model.parameters() if torch.is_floating_point(p)}
    except Exception as e:
        print(f"[DType] {tag}: dtype scan failed: {e}")
        return model
    if len(dtypes) > 1 or (dtypes and next(iter(dtypes)) != dtype):
        model = model.to(dtype)
        print(f"[DType] {tag}: unified {sorted(str(d) for d in dtypes)} -> {dtype}.")
    return model


def _sample_token_length(dataset, idx):
    try:
        item = dataset[idx]
        length = item.get("length") if isinstance(item, dict) else None
        if length is not None:
            return int(length.item() if hasattr(length, "item") else length)
        input_ids = item.get("input_ids") if isinstance(item, dict) else None
        labels = item.get("labels") if isinstance(item, dict) else None
        return int(len(input_ids or []) + len(labels or []))
    except Exception:
        return 1


class TokenBucketBatchSampler(Sampler):
    def __init__(self, dataset, max_batch_size, max_tokens, bucket_size=256, shuffle=True, seed=42, drop_last=False):
        self.dataset = dataset
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_tokens = max(1, int(max_tokens))
        self.bucket_size = max(self.max_batch_size, int(bucket_size))
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.lengths = [_sample_token_length(dataset, idx) for idx in range(len(dataset))]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda idx: self.lengths[idx])
        pools = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle:
            rng.shuffle(pools)
            for pool in pools:
                rng.shuffle(pool)

        batches = []
        for pool in pools:
            batch = []
            batch_tokens = 0
            for idx in pool:
                sample_tokens = max(1, self.lengths[idx])
                would_exceed = batch and (len(batch) >= self.max_batch_size or batch_tokens + sample_tokens > self.max_tokens)
                if would_exceed:
                    if len(batch) == self.max_batch_size or not self.drop_last:
                        batches.append(batch)
                    batch = []
                    batch_tokens = 0
                batch.append(idx)
                batch_tokens += sample_tokens
            if batch and (len(batch) == self.max_batch_size or not self.drop_last):
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        rank, world_size = _distributed_info()
        batches = self._build_batches()
        if world_size > 1:
            batches = batches[rank::world_size]
        return iter(batches)

    def __len__(self):
        total = len(self._build_batches())
        _, world_size = _distributed_info()
        return (total + world_size - 1) // world_size


class SampleCountEvalCallback(TrainerCallback):
    def __init__(self, sample_interval=0):
        self.sample_interval = max(0, int(sample_interval or 0))
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.sample_interval <= 0 or self.trainer is None:
            return control
        if getattr(self.trainer, "_samples_since_eval", 0) >= self.sample_interval:
            print(f"\n[SampleEval] {self.trainer._samples_since_eval} samples crossed; evaluating at step {state.global_step}")
            self.trainer._samples_since_eval = 0
            control.should_evaluate = True
            control.should_save = True
        return control


class VRAMSafeTrainer(Seq2SeqTrainer):
    def __init__(self, *args, vram_limit_gb=None, dynamic_batch=False, max_tokens=50000, length_bucket_size=256, sample_eval_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vram_limit_gb = vram_limit_gb
        self.dynamic_batch = dynamic_batch
        self.max_tokens = max(1, int(max_tokens))
        self.length_bucket_size = max(1, int(length_bucket_size))
        self.sample_eval_interval = max(0, int(sample_eval_interval or 0))
        self._samples_since_eval = 0
        self.eval_oom_retries = 0
        self.max_eval_oom_retries = 3
        self._current_gold_answer = None
        self._current_database = None
        self._current_input_text = None
        self._current_batch_indices = None

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, dict):
            self._current_gold_answer = inputs.pop("gold_answer", None)
            self._current_database = inputs.pop("database", None)
            self._current_input_text = inputs.pop("input_text", None)
            self._current_batch_indices = inputs.pop("index", None)
        return super()._prepare_inputs(inputs)

    def get_train_dataloader(self):
        if not self.dynamic_batch:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        batch_sampler = TokenBucketBatchSampler(
            self.train_dataset,
            max_batch_size=self.args.train_batch_size,
            max_tokens=self.max_tokens,
            bucket_size=self.length_bucket_size,
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        if is_main_process():
            print(f"[DynamicBatch] max_batch={self.args.train_batch_size}, max_tokens={self.max_tokens}, bucket_size={self.length_bucket_size}, batches={len(batch_sampler)}")
        return self.accelerator.prepare(dataloader)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.sample_eval_interval > 0 and isinstance(inputs, dict) and "input_ids" in inputs:
            local_batch = int(inputs["input_ids"].shape[0])
            _, world_size = _distributed_info()
            self._samples_since_eval += local_batch * world_size
        vram = get_vram_info()
        if vram["total_gb"] > 0:
            if self.vram_limit_gb is not None:
                allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                if allocated_gb > self.vram_limit_gb:
                    torch.cuda.empty_cache()
                    gc.collect()
                    allocated_gb2 = torch.cuda.memory_allocated(0) / (1024**3)
                    if allocated_gb2 > self.vram_limit_gb:
                        raise RuntimeError(
                            f"VRAM critical: {allocated_gb2:.1f} GB allocated, "
                            f"limit {self.vram_limit_gb:.1f} GB. "
                            f"Reduce batch size or max_input_length."
                        )
            else:
                used_pct = vram["reserved_gb"] / vram["total_gb"]
                if used_pct > 0.90:
                    torch.cuda.empty_cache()
                    gc.collect()
                    vram2 = get_vram_info()
                    used_pct2 = vram2["reserved_gb"] / vram2["total_gb"]
                    if used_pct2 > 0.95:
                        raise RuntimeError(
                            f"VRAM critical: {vram2['reserved_gb']:.1f}/{vram2['total_gb']:.1f} GB used. "
                            f"Reduce batch size or max_input_length."
                        )
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                vram_now = get_vram_info()
                raise RuntimeError(
                    f"OOM during training step. VRAM: {vram_now['reserved_gb']:.1f}/"
                    f"{vram_now['total_gb']:.1f} GB. "
                    f"Try: --batch_size 1 --max_input_length 256 --lora_r 4"
                )
            raise

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Evaluate with OOM recovery by reducing eval batch size."""
        try:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.eval_oom_retries < self.max_eval_oom_retries:
                # Reduce eval batch size by 10%
                original_batch = self.args.per_device_eval_batch_size
                new_batch = max(1, int(original_batch * 0.9))
                
                if new_batch == original_batch:
                    new_batch = original_batch - 1
                
                print(f"\n[Eval OOM] Reducing eval batch size: {original_batch} → {new_batch}")
                self.args.per_device_eval_batch_size = new_batch
                self.eval_oom_retries += 1
                
                torch.cuda.empty_cache()
                gc.collect()
                
                # Retry evaluation
                return self.evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            
            if "out of memory" in str(e).lower():
                print(f"\n[Eval OOM] Failed after {self.max_eval_oom_retries} retries")
            raise


def load_neo4j_database_mapping(mapping_path=""):
    if not mapping_path:
        return {}
    path = Path(mapping_path)
    if not path.exists():
        if is_main_process():
            print(f"Neo4j database mapping file not found: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        if is_main_process():
            print(f"Could not read Neo4j database mapping from {path}: {e}")
        return {}
    mapping = data.get("mapping", data) if isinstance(data, dict) else {}
    if not isinstance(mapping, dict):
        if is_main_process():
            print(f"Neo4j database mapping file has no mapping object: {path}")
        return {}
    resolved = {str(k): str(v) for k, v in mapping.items() if v}
    if is_main_process():
        print(f"Loaded {len(resolved)} Neo4j database mapping entries from {path}")
    return resolved


def get_unique_databases_from_dataset(data_dir, train_parquet="", val_parquet="", test_parquet="", database_mapping=None, max_rows=None):
    """Extract unique databases from dataset parquet files and map them."""
    database_mapping = database_mapping or {}
    unique_dbs = set()
    
    for parquet_name in [train_parquet, val_parquet, test_parquet]:
        if not parquet_name:
            continue
        parquet_path = os.path.join(data_dir, parquet_name)
        if not os.path.exists(parquet_path):
            continue
        try:
            df = pd.read_parquet(parquet_path, columns=["database"] if "database" in pd.read_parquet(parquet_path, nrows=0).columns else None)
            if "database" in df.columns:
                raw_dbs = df["database"].dropna().unique()
                for db in raw_dbs:
                    mapped = str(database_mapping.get(str(db), db))
                    unique_dbs.add(mapped)
        except Exception as e:
            if is_main_process():
                print(f"  Warning: could not extract databases from {parquet_path}: {e}")
    
    return sorted(unique_dbs)


class Neo4jClient:
    def __init__(self, url="", port=15964, user="neo4j", password="berttokg123", database="neo4j", database_mapping=None, proxies=None):
        self._cache = {}
        self.database = database
        self.database_mapping = database_mapping or {}
        self.driver = None
        self.session = None
        self.http_api = "tx"

        # Backward-compatible default: use legacy localhost HTTP endpoint.
        if not url:
            default_database = self.database_mapping.get(str(self.database), str(self.database))
            url = f"http://localhost:{port}/db/{default_database}/tx/commit"

        self.url = url
        parsed = urlparse(url)
        scheme = (parsed.scheme or "http").lower()
        self.mode = "http"

        if scheme in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
            try:
                GraphDatabase = __import__("neo4j", fromlist=["GraphDatabase"]).GraphDatabase
            except Exception as e:
                raise RuntimeError(
                    "Bolt URL provided but neo4j driver is not installed. "
                    "Install with: pip install neo4j"
                ) from e
            self.mode = "bolt"
            self.driver = GraphDatabase.driver(url, auth=(user, password))
        else:
            # HTTP mode; if caller provided host only, append transactional endpoint.
            if parsed.path in ("", "/"):
                base = url.rstrip("/")
                self.url = f"{base}/db/{self.database}/tx/commit"
            elif parsed.path.rstrip("/").endswith("/query/v2"):
                self.http_api = "query"
                if "{database}" in self.url:
                    self.url = self.url.replace("{database}", self._resolve_database(self.database))
            if "{database}" in self.url:
                self.url = self.url.replace("{database}", self._resolve_database(self.database))
            self.session = requests.Session()
            self.session.auth = (user, password)
            self.session.headers.update({"Content-Type": "application/json"})
            if proxies:
                self.session.proxies.update(proxies)

    def _resolve_database(self, database=None):
        database_name = str(database or self.database)
        return self.database_mapping.get(database_name, database_name)

    @staticmethod
    def _flatten(v):
        if isinstance(v, dict):
            return tuple(sorted((k, Neo4jClient._flatten(v)) for k, v in v.items()))
        if isinstance(v, list):
            return tuple(Neo4jClient._flatten(x) for x in v)
        return v

    def execute(self, cypher, timeout=5, database=None):
        cypher_stripped = cypher.strip().rstrip(";")
        resolved_database = self._resolve_database(database)
        cache_key = (cypher_stripped, str(database or self.database), resolved_database)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            if self.mode == "bolt":
                with self.driver.session(database=resolved_database) as db_session:
                    records = db_session.run(cypher_stripped)
                    result = [tuple(self._flatten(v) for v in rec.values()) for rec in records]
            else:
                url = self.url
                if database and "/db/" in self.url:
                    parsed = urlparse(self.url)
                    base = f"{parsed.scheme}://{parsed.netloc}"
                    if self.http_api == "query":
                        url = f"{base}/db/{resolved_database}/query/v2"
                    else:
                        url = f"{base}/db/{resolved_database}/tx/commit"
                if self.http_api == "query":
                    payload = {"statement": cypher_stripped, "parameters": {}}
                else:
                    payload = {"statements": [{"statement": cypher_stripped}]}
                resp = self.session.post(url, json=payload, timeout=timeout)
                data = resp.json()
                if data.get("errors"):
                    result = None
                elif self.http_api == "query":
                    values = data.get("data", {}).get("values", [])
                    result = [tuple(self._flatten(x) for x in row) for row in values]
                else:
                    rows = data["results"][0].get("data", [])
                    result = [tuple(self._flatten(x) for x in r.get("row", [])) for r in rows]
        except Exception:
            result = None
        if result is not None:
            self._cache[cache_key] = result
        return result

    def close(self):
        try:
            if self.driver is not None:
                self.driver.close()
        except Exception:
            pass
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass


def _split_arg_list(value):
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,\s]+", str(value)) if part.strip()]


def _split_port_list(value, fallback=()):
    ports = []
    for part in _split_arg_list(value):
        try:
            port = int(part)
        except ValueError:
            print(f"Ignoring invalid Neo4j port: {part}")
            continue
        if 0 < port < 65536 and port not in ports:
            ports.append(port)
    for port in fallback:
        if port and port not in ports:
            ports.append(port)
    return ports


def _split_optional_port_list(value, fallback=(None,)):
    ports = _split_port_list(value)
    if ports:
        return ports
    return list(fallback)


def _host_and_port(value):
    parsed = urlparse(value if "://" in value else f"//{value}")
    return parsed.hostname, parsed.port, parsed.path


def _https_query_api_url(host, port, database):
    port_part = "" if port is None else f":{port}"
    return f"https://{host}{port_part}/db/{database}/query/v2"


def build_neo4j_candidates(args):
    candidates = []
    seen = set()
    database = args.neo4j_database_mapping.get(args.neo4j_database, args.neo4j_database)

    def add(url, label):
        if url not in seen:
            seen.add(url)
            candidates.append({"url": url, "label": label})

    if args.neo4j_url:
        add(args.neo4j_url, "configured --neo4j_url")

    https_ports = _split_optional_port_list(args.neo4j_url_ports, fallback=(None,))
    bolt_ports = _split_port_list(args.neo4j_ip_ports, fallback=(7687,))

    for raw_url in _split_arg_list(args.neo4j_urls):
        host, explicit_port, path = _host_and_port(raw_url)
        if "://" in raw_url and path not in ("", "/"):
            add(raw_url, "provided URL")
        if not host:
            print(f"Ignoring invalid Neo4j URL/host: {raw_url}")
            continue
        for port in ([explicit_port] if explicit_port else https_ports):
            add(_https_query_api_url(host, port, database), f"HTTPS Query API from {raw_url}")
        for port in ([explicit_port] if explicit_port else bolt_ports):
            add(f"bolt://{host}:{port}", f"Bolt from {raw_url}")

    for raw_ip in _split_arg_list(args.neo4j_ips):
        host, explicit_port, _ = _host_and_port(raw_ip)
        if not host:
            print(f"Ignoring invalid Neo4j IP/host: {raw_ip}")
            continue
        for port in ([explicit_port] if explicit_port else https_ports):
            add(_https_query_api_url(host, port, database), f"HTTPS Query API from {raw_ip}")
        for port in ([explicit_port] if explicit_port else bolt_ports):
            add(f"bolt://{host}:{port}", f"Bolt from {raw_ip}")

    return candidates


def _neo4j_required_databases(args):
    """Determine which databases to check during preflight based on preflight mode."""
    mode = str(getattr(args, 'neo4j_preflight_mode', 'fallback')).strip().lower()
    
    if mode == 'all_mapped':
        # Check all databases in the mapping
        databases = []
        for database in sorted(set(args.neo4j_database_mapping.values())):
            if database not in databases:
                databases.append(database)
    elif mode == 'dataset':
        # Check only databases present in the dataset
        databases = get_unique_databases_from_dataset(
            args.data_dir,
            args.train_parquet,
            args.val_parquet,
            args.test_parquet,
            args.neo4j_database_mapping
        )
        if not databases:
            # Fallback to default if dataset has no databases
            databases = [args.neo4j_database]
    else:  # fallback
        # Check only the default/fallback database
        databases = [args.neo4j_database]
    
    # Ensure at least the default database is included
    mapped_default = args.neo4j_database_mapping.get(args.neo4j_database, args.neo4j_database)
    if mapped_default not in databases:
        databases.insert(0, mapped_default)
    
    # Map through database mapping
    required = []
    for database in databases or ["neo4j"]:
        mapped_database = args.neo4j_database_mapping.get(database, database)
        if mapped_database not in required:
            required.append(mapped_database)
    
    return required


def resolve_neo4j_connection(args):
    if args.neo4j_skip_discovery:
        if args.neo4j_url:
            if is_main_process():
                print(f"Neo4j discovery skipped; using preflight-selected endpoint: {args.neo4j_url}")
        return

    if not (args.neo4j_url or args.neo4j_urls or args.neo4j_ips):
        return

    candidates = build_neo4j_candidates(args)
    if not candidates:
        return

    if is_main_process():
        mode = str(getattr(args, 'neo4j_preflight_mode', 'fallback')).strip().lower()
        print(f"Neo4j preflight mode: {mode}")

    required_databases = _neo4j_required_databases(args)
    if is_main_process():
        print(f"Testing {len(candidates)} Neo4j endpoint candidate(s) across databases: {', '.join(required_databases)}")
    available = []
    for idx, candidate in enumerate(candidates, start=1):
        url = candidate["url"]
        client = None
        try:
            client = Neo4jClient(
                url=url,
                user=args.neo4j_user,
                password=args.neo4j_password,
                database=args.neo4j_database,
                database_mapping=args.neo4j_database_mapping,
            )
            db_results = []
            for database in required_databases:
                rows = client.execute("RETURN 1 AS ok", timeout=args.neo4j_discovery_timeout, database=database)
                db_results.append((database, rows is not None))
            missing = [database for database, ok in db_results if not ok]
            if not missing:
                available.append(candidate)
                if is_main_process():
                    print(f"  [{idx}] OK  {url} ({candidate['label']}; dbs={', '.join(required_databases)})")
            else:
                if is_main_process():
                    print(f"  [{idx}] --  {url} (failed dbs={', '.join(missing)})")
        except Exception as e:
            if is_main_process():
                print(f"  [{idx}] --  {url} ({e})")
        finally:
            if client is not None:
                client.close()

    if not available:
        if is_main_process():
            print("No Neo4j endpoint had all required databases available; keeping the configured/default endpoint.")
        return

    selected = available[0]
    if len(available) > 1 and sys.stdin.isatty() and not args.neo4j_auto_select:
        if is_main_process():
            print("Available Neo4j endpoints:")
            for idx, candidate in enumerate(available, start=1):
                print(f"  {idx}. {candidate['url']} ({candidate['label']})")
            choice = input("Select endpoint to use [1]: ").strip()
        else:
            choice = ""
        if choice:
            try:
                selected = available[max(0, min(len(available) - 1, int(choice) - 1))]
            except ValueError:
                if is_main_process():
                    print("Invalid selection; using the first available endpoint.")

    args.neo4j_url = selected["url"]
    parsed = urlparse(args.neo4j_url)
    if parsed.port:
        args.neo4j_port = parsed.port
    if is_main_process():
        print(f"Using Neo4j endpoint: {args.neo4j_url}")


def _normalize_gold_value(value):
    if isinstance(value, str) and len(value) > 20 and value.startswith("{"):
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return Neo4jClient._flatten(value)


def _parse_gold_answer(gold_answer_json):
    if not gold_answer_json or not isinstance(gold_answer_json, str):
        return None
    try:
        parsed = json.loads(gold_answer_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return set()
    if all(isinstance(row, dict) for row in parsed):
        return set(tuple(_normalize_gold_value(v) for v in row.values()) for row in parsed)
    return set(tuple(_normalize_gold_value(v) for v in row) if isinstance(row, list) else (_normalize_gold_value(row),) for row in parsed)


_SCHEMA_RE = re.compile(r"^(?:reference-)?schema:\s*(.+?)(?:\n|$)", re.MULTILINE | re.IGNORECASE)
_SCHEMA_REL_RE = re.compile(r"^R:\s*(.+?)(?:\n|$)", re.MULTILINE)
_ENTITY_PAT = re.compile(r"\([^)]*:(\w+)[^)]*\)")
_REL_PAT = re.compile(r"\[[^\]]*:(\w+)[^\]]*\]")


def _parse_schema(input_text):
    valid_entities = set()
    valid_rels = set()
    if not input_text:
        return valid_entities, valid_rels
    for schema_match in _SCHEMA_RE.finditer(input_text):
        if "(" not in schema_match.group(1):
            continue
        for segment in schema_match.group(1).split("|"):
            entity = segment.split("(")[0].strip()
            if entity:
                valid_entities.add(entity)
    rel_match = _SCHEMA_REL_RE.search(input_text)
    if rel_match:
        for segment in rel_match.group(1).split("|"):
            parts = segment.split(">")
            if len(parts) == 3:
                valid_rels.add(parts[1].strip())
    return valid_entities, valid_rels


def _parse_cypher_elements(cypher):
    labels = set(_ENTITY_PAT.findall(cypher))
    rel_types = set(_REL_PAT.findall(cypher))
    props = set(re.findall(r"(?<![\w:])([a-z_]\w*)\.([a-z_]\w*)", cypher, re.IGNORECASE))
    directions = set()
    for rel in re.finditer(r"(-\[[^\]]*\]->|<-\[[^\]]*\]-|--\[[^\]]*\]--)", cypher):
        rel_type = ""
        rel_type_match = re.search(r":(\w+)", rel.group(1))
        if rel_type_match:
            rel_type = rel_type_match.group(1)
        if rel.group(1).startswith("<-"):
            directions.add((rel_type, "<-"))
        elif rel.group(1).endswith("->"):
            directions.add((rel_type, "->"))
        else:
            directions.add((rel_type, "--"))
    return labels, rel_types, {prop for _, prop in props}, directions


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _structure_score(pred_cypher, gold_cypher):
    pred_labels, pred_rels, pred_props, pred_dirs = _parse_cypher_elements(pred_cypher)
    gold_labels, gold_rels, gold_props, gold_dirs = _parse_cypher_elements(gold_cypher)
    clause_names = ("MATCH", "WHERE", "RETURN", "ORDER BY", "LIMIT", "WITH", "CALL", "UNWIND")
    clause_score = sum((name in pred_cypher.upper()) == (name in gold_cypher.upper()) for name in clause_names) / len(clause_names)
    scores = {
        "label_f1": _jaccard(pred_labels, gold_labels),
        "rel_f1": _jaccard(pred_rels, gold_rels),
        "prop_f1": _jaccard(pred_props, gold_props),
        "dir_f1": _jaccard(pred_dirs, gold_dirs),
        "clause_f1": clause_score,
    }
    scores["ast_similarity"] = 0.25 * scores["label_f1"] + 0.2 * scores["rel_f1"] + 0.2 * scores["dir_f1"] + 0.2 * scores["prop_f1"] + 0.15 * scores["clause_f1"]
    return scores


def _schema_score(input_text, pred_labels, pred_rels):
    valid_entities, valid_rels = _parse_schema(input_text)
    parts = []
    if pred_labels:
        parts.append(len(pred_labels & valid_entities) / len(pred_labels) if valid_entities else 0.0)
    if pred_rels:
        parts.append(len(pred_rels & valid_rels) / len(pred_rels) if valid_rels else 0.0)
    return sum(parts) / len(parts) if parts else 0.0


def _text_similarity(a, b):
    a = normalize_cypher_eval(a)
    b = normalize_cypher_eval(b)
    max_len = max(len(a), len(b), 1)
    return 1.0 - min(token_levenstein(list(a), list(b)), max_len) / max_len


def _query_shape(text, gold_cypher):
    if not text:
        return False, "empty"
    if not re.match(r"^\s*(?:(?:EXPLAIN|PROFILE)\s+)?(?:OPTIONAL\s+MATCH|MATCH|CALL|WITH|UNWIND|CREATE|MERGE)\b", text, re.IGNORECASE):
        return False, "missing_query_start"
    if re.search(r"\bRETURN\b", gold_cypher, re.IGNORECASE) and not re.search(r"\bRETURN\b", text, re.IGNORECASE):
        return False, "missing_return"
    return True, "ok"


def _reward_value(reward_result):
    if isinstance(reward_result, tuple):
        return float(reward_result[0])
    return float(reward_result)


def print_reward_function_description():
    if not is_main_process():
        return
    print("\n=== Phase 2 Reward Function ===")
    print("Reward range is clipped to [-6, 6]. It combines execution result overlap, structural Cypher similarity, schema/entity alignment, and penalties.")
    print("Positive terms: exact execution match base, result Jaccard, AST/label/relation/property/clause similarity, schema compliance, entity relevance.")
    print("Negative terms: non-Cypher/prose output, failed execution, invalid schema labels/relations, garbage text, length drift, trivial RETURN/WITH-only queries, bracket/case mistakes.")
    print("If a gold_answer column exists, it is used instead of executing the gold query; predictions are still executed against Neo4j using the sample database when available.")
    print("RL loss uses sampled candidate rewards, optional critic baseline, entropy bonus, optional KL penalty against a frozen reference model, and an SFT loss coefficient.\n")


def compute_reward(pred_cypher, gold_cypher, neo4j_client, gold_answer_json=None, database=None, input_text=None):
    pred_cypher = pred_cypher.rstrip(";").strip()
    gold_cypher = gold_cypher.rstrip(";").strip()
    shape_ok, shape_reason = _query_shape(pred_cypher, gold_cypher)
    garbage = min(2.0, sum(1 for char in pred_cypher if ord(char) > 127) / max(len(pred_cypher), 1) * 8)
    length_ratio = len(pred_cypher.split()) / max(len(gold_cypher.split()), 1)
    length_penalty = max(0.0, abs(length_ratio - 1.0) - 0.25)
    if not shape_ok:
        raw = max(-6.0, -3.0 - garbage - length_penalty)
        return raw, False, {"branch": "shape_fail", "shape_reason": shape_reason, "reward": round(raw, 3)}

    gold_set = _parse_gold_answer(gold_answer_json)
    if gold_set is None:
        gold_rows = neo4j_client.execute(gold_cypher, database=database)
        if gold_rows is None:
            return 0.0, False, {"branch": "gold_exec_fail", "reward": 0.0}
        gold_set = set(tuple(row) for row in gold_rows)

    pred_rows = neo4j_client.execute(pred_cypher, database=database)
    exec_ok = pred_rows is not None
    pred_set = set(tuple(row) for row in pred_rows) if exec_ok else set()
    result_jaccard = len(gold_set & pred_set) / len(gold_set | pred_set) if (gold_set | pred_set) else 1.0
    pred_labels, pred_rels, _, _ = _parse_cypher_elements(pred_cypher)
    structure = _structure_score(pred_cypher, gold_cypher)
    schema = _schema_score(input_text, pred_labels, pred_rels) if input_text else 0.0
    invalid_labels, invalid_rels = 0, 0
    if input_text:
        valid_labels, valid_rels = _parse_schema(input_text)
        invalid_labels = len(pred_labels - valid_labels) if valid_labels else 0
        invalid_rels = len(pred_rels - valid_rels) if valid_rels else 0
    invalid_penalty = min(1.5, 0.5 * invalid_labels + 0.5 * invalid_rels)
    bracket_penalty = min(2.0, 0.5 * (abs(pred_cypher.count("(") - pred_cypher.count(")")) + abs(pred_cypher.count("{") - pred_cypher.count("}"))))
    exact_text = normalize_cypher_eval(pred_cypher) == normalize_cypher_eval(gold_cypher)
    if exact_text and (exec_ok or gold_set == pred_set):
        branch = "exact_match"
        base = 3.0
        result_jaccard = 1.0
    elif exec_ok and result_jaccard > 0:
        branch = "partial_match"
        base = 0.5
    elif exec_ok:
        branch = "executed_no_overlap"
        base = 0.2
    else:
        branch = "exec_fail"
        base = -3.0 + 2.0 * _text_similarity(pred_cypher, gold_cypher)

    raw = base + 2.0 * result_jaccard + 0.8 * structure["ast_similarity"] + 0.3 * schema - garbage - length_penalty - invalid_penalty - bracket_penalty
    reward = max(-6.0, min(6.0, raw))
    return reward, exec_ok, {
        "branch": branch,
        "base": round(base, 3),
        "result_jaccard": round(result_jaccard, 3),
        "ast_sim": round(structure["ast_similarity"], 3),
        "label_f1": round(structure["label_f1"], 3),
        "rel_f1": round(structure["rel_f1"], 3),
        "prop_f1": round(structure["prop_f1"], 3),
        "clause_f1": round(structure["clause_f1"], 3),
        "schema_score": round(schema, 3),
        "garbage_penalty": round(-garbage, 3),
        "length_penalty": round(-length_penalty, 3),
        "invalid_penalty_total": round(-invalid_penalty, 3),
        "bracket_penalty": round(-bracket_penalty, 3),
        "exec": int(exec_ok),
        "reward": round(reward, 3),
    }


class CriticNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, encoder_hidden_states):
        pooled = encoder_hidden_states.mean(dim=1)
        return self.net(pooled).squeeze(-1)


class ReinforceTrainer(VRAMSafeTrainer):
    def __init__(self, *args, neo4j_client=None, reinforce_coef=0.3,
                 reward_ema_alpha=0.1, reinforce_tokenizer=None,
                 reinforce_candidates=1, critic_lr=1e-4, disable_critic=False,
                 entropy_coef=0.0, entropy_min=0.0, exact_match_weight=0.3,
                 train_temperature=0.7, train_temperature_min=0.7,
                 train_noise_scale=0.0, train_top_k=50, temp_spread=0.0,
                 temp_spread_min=0.0, kl_coef=0.0, ppo_clip_eps=0.0,
                 kl_ref_model=None, kl_target=1.0, kl_adaptive_alpha=0.0,
                 kl_beta_min=0.0, kl_beta_max=1.0, kl_prefix_len=0,
                 kl_ref_temperature=1.0, sft_coef=1.0,
                 eos_suppress_strength=0.0, eos_suppress_full_until=0,
                 eos_suppress_decay_until=0, chunk_size=1,
                 diversity_coef=0.0, simplicity_coef=0.0,
                 consistency_coef=0.0, dynamic_batch=False, max_tokens=50000,
                 dynamic_generation_length=False, generation_length_tolerance=0.10,
                 rl_warmup_steps=0, roll_stream=None,
                 **kwargs):
        model = kwargs.get('model')
        if model is not None:
            hidden_dim = getattr(model.config, 'd_model', None) or getattr(model.config, 'hidden_size', None)
            if hidden_dim:
                self.critic = CriticNet(hidden_dim)
                self.critic.to(model.device)
                model.critic_head = self.critic
            else:
                self.critic = None
        else:
            self.critic = None
        super().__init__(*args, **kwargs)
        self.neo4j_client = neo4j_client
        self.roll_stream = roll_stream
        self.reinforce_coef = reinforce_coef
        self.reward_ema = 0.5
        self.reward_ema_alpha = reward_ema_alpha
        self.reinforce_tokenizer = reinforce_tokenizer
        self.reinforce_candidates = reinforce_candidates
        self.disable_critic = disable_critic
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.entropy_min = entropy_min
        self.exact_match_weight = exact_match_weight
        self.train_temperature = train_temperature
        self.train_temperature_min = train_temperature_min
        self.train_noise_scale = train_noise_scale
        self.train_top_k = train_top_k
        self.temp_spread = temp_spread
        self.temp_spread_min = temp_spread_min
        self.kl_coef = kl_coef
        self.ppo_clip_eps = ppo_clip_eps
        self.kl_ref_model = kl_ref_model
        self.kl_target = kl_target
        self.kl_adaptive_alpha = kl_adaptive_alpha
        self.kl_beta_min = kl_beta_min
        self.kl_beta_max = kl_beta_max
        self.kl_prefix_len = kl_prefix_len
        self.kl_ref_temperature = kl_ref_temperature
        self.sft_coef = sft_coef
        self.eos_suppress_strength = eos_suppress_strength
        self.eos_suppress_full_until = eos_suppress_full_until
        self.eos_suppress_decay_until = eos_suppress_decay_until
        self.chunk_size = max(1, int(chunk_size))
        self.diversity_coef = diversity_coef
        self.simplicity_coef = simplicity_coef
        self.consistency_coef = consistency_coef
        self.dynamic_batch = dynamic_batch
        self.max_tokens = max(1, int(max_tokens))
        self.dynamic_generation_length = dynamic_generation_length
        self.generation_length_tolerance = max(0.0, generation_length_tolerance)
        self.rl_warmup_steps = max(0, int(rl_warmup_steps))
        self._last_reinforce_step = -5

        inactive = []
        if self.ppo_clip_eps:
            inactive.append("ppo_clip_eps")
        if self.kl_adaptive_alpha:
            inactive.append("kl_adaptive_alpha")
        if self.eos_suppress_strength:
            inactive.append("eos_suppress_strength")
        if self.dynamic_generation_length:
            inactive.append("dynamic_generation_length")
        if self.diversity_coef or self.simplicity_coef or self.consistency_coef:
            inactive.append("diversity/simplicity/consistency bonuses")
        if inactive:
            print(f"  [RL_CONFIG] accepted v1 knobs currently inactive in this lightweight path: {', '.join(inactive)}", flush=True)

    def _candidate_temperature(self, candidate_idx, n_candidates):
        if n_candidates <= 1:
            return max(self.train_temperature_min, self.train_temperature)
        total_steps = getattr(self.args, "max_steps", 0) or 0
        if total_steps > 0:
            progress = min(1.0, max(0.0, self.state.global_step / max(total_steps, 1)))
        else:
            progress = 0.0
        spread = self.temp_spread_min + (self.temp_spread - self.temp_spread_min) * (1.0 - progress)
        return max(self.train_temperature_min, self.train_temperature * (1.0 + candidate_idx * spread))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_for_ops = model.module if hasattr(model, "module") else model
        model_float_dtype = get_model_float_dtype(model_for_ops)
        labels = inputs["labels"]
        do_reinforce = model.training and (self.state.global_step - self._last_reinforce_step >= 2)

        if do_reinforce:
            model_for_ops.config.output_hidden_states = True
            outputs = model(**inputs)
            ce_loss = outputs.loss
            encoder_hidden = outputs.encoder_last_hidden_state
            model_for_ops.config.output_hidden_states = False
        else:
            outputs = model(**inputs)
            ce_loss = outputs.loss

        ce_was_nonfinite = not torch.isfinite(ce_loss)
        if ce_was_nonfinite:
            if is_main_process():
                print(f"  [RL_WARN] non-finite CE loss at step={self.state.global_step}; replacing with 0.0 for stability", flush=True)
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if not model.training:
            return (ce_loss, outputs) if return_outputs else ce_loss

        del outputs

        if do_reinforce:
            if ce_was_nonfinite:
                if is_main_process():
                    print(f"  [RL_WARN] CE was non-finite; skipping RL at step={self.state.global_step} to prevent generation crash", flush=True)
                return ce_loss
            rl_warmup = int(getattr(self, 'rl_warmup_steps', 0))
            if self.state.global_step < rl_warmup:
                if is_main_process():
                    print(f"  [RL_WARN] RL warmup: skipping RL at step={self.state.global_step} (warmup until step {rl_warmup})", flush=True)
                return ce_loss
            self._last_reinforce_step = self.state.global_step
            gc.collect()

            batch_size = inputs["input_ids"].shape[0]
            n_candidates = self.reinforce_candidates
            use_multi = n_candidates > 1

            with torch.no_grad():
                autocast_ctx = nullcontext()
                if torch.cuda.is_available() and model_float_dtype in (torch.float16, torch.bfloat16):
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=model_float_dtype)
                model_for_ops.config.use_cache = True
                with autocast_ctx:
                    if use_multi:
                        class SanitizeLogitsProcessor(LogitsProcessor):
                            """Clamp logits to prevent NaN/Inf in softmax and multinomial."""
                            def __call__(self, input_ids, scores):
                                # Clamp to safe range for softmax
                                scores = torch.clamp(scores, min=-50.0, max=50.0)
                                # Replace any remaining NaN/Inf with 0
                                scores = torch.where(torch.isfinite(scores), scores, torch.tensor(0.0, device=scores.device, dtype=scores.dtype))
                                return scores

                        class GaussianNoiseProcessor(LogitsProcessor):
                            def __init__(self, scale):
                                self.scale = scale

                            def __call__(self, input_ids, scores):
                                return scores + torch.randn_like(scores) * self.scale

                        all_candidate_ids = []
                        candidate_temperatures = []
                        for c in range(n_candidates):
                            torch.cuda.empty_cache()
                            candidate_temperature = self._candidate_temperature(c, n_candidates)
                            candidate_temperatures.append(round(candidate_temperature, 4))
                            logits_processors = LogitsProcessorList()
                            # Always sanitize logits to prevent NaN/Inf in sampling
                            logits_processors.append(SanitizeLogitsProcessor())
                            if self.train_noise_scale > 0:
                                logits_processors.append(GaussianNoiseProcessor(self.train_noise_scale))
                            c_ids = model_for_ops.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=self.args.generation_max_length,
                                do_sample=True,
                                temperature=candidate_temperature,
                                top_k=self.train_top_k,
                                logits_processor=logits_processors if len(logits_processors) else None,
                            )
                            all_candidate_ids.append(c_ids)
                        print(f"  [CANDIDATES] step={self.state.global_step} temperatures={candidate_temperatures} top_k={self.train_top_k} noise={self.train_noise_scale}", flush=True)
                        max_seq_len = max(t.shape[1] for t in all_candidate_ids)
                        gen_ids = torch.cat([torch.nn.functional.pad(t, (0, max_seq_len - t.shape[1]), value=self.reinforce_tokenizer.pad_token_id) for t in all_candidate_ids], dim=0)
                        del all_candidate_ids
                        torch.cuda.empty_cache()
                    else:
                        gen_ids = model_for_ops.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=self.args.generation_max_length,
                            do_sample=False,
                            num_beams=1,
                        )
                model_for_ops.config.use_cache = False
            gc.collect()

            pred_texts = self.reinforce_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            labels_np = labels.cpu().numpy()
            labels_np = np.where(labels_np != -100, labels_np, self.reinforce_tokenizer.pad_token_id)
            gold_texts = self.reinforce_tokenizer.batch_decode(labels_np, skip_special_tokens=True)

            rewards = []
            best_gen_ids = []
            jaccards = []
            exact_matches = []
            branches = []
            gold_answers = self._current_gold_answer or [None] * batch_size
            databases = self._current_database or [None] * batch_size
            input_texts = self._current_input_text or [None] * batch_size
            for i in range(batch_size):
                gold = gold_texts[i].strip()
                if use_multi:
                    candidate_indices = [candidate_idx * batch_size + i for candidate_idx in range(n_candidates)]
                    group_candidates = [pred_texts[idx].strip() for idx in candidate_indices]
                    group_results = [
                        compute_reward(
                            c,
                            gold,
                            self.neo4j_client,
                            gold_answer_json=gold_answers[i] if i < len(gold_answers) else None,
                            database=databases[i] if i < len(databases) else None,
                            input_text=input_texts[i] if i < len(input_texts) else None,
                        )
                        for c in group_candidates
                    ]
                    group_jaccards = [_reward_value(result) for result in group_results]
                    group_ems = [1.0 if c == gold else 0.0 for c in group_candidates]
                    group_scores = [round(j + self.exact_match_weight * em, 4) for j, em in zip(group_jaccards, group_ems)]
                    best_idx = int(np.argmax(group_scores))
                    rewards.append(group_scores[best_idx])
                    jaccards.append(group_jaccards[best_idx])
                    exact_matches.append(group_ems[best_idx])
                    branches.append(group_results[best_idx][2].get("branch", "unknown") if isinstance(group_results[best_idx], tuple) else "legacy")
                    best_gen_ids.append(gen_ids[candidate_indices[best_idx]])
                else:
                    reward_result = compute_reward(
                        pred_texts[i].strip(),
                        gold,
                        self.neo4j_client,
                        gold_answer_json=gold_answers[i] if i < len(gold_answers) else None,
                        database=databases[i] if i < len(databases) else None,
                        input_text=input_texts[i] if i < len(input_texts) else None,
                    )
                    j = _reward_value(reward_result)
                    em = 1.0 if pred_texts[i].strip() == gold else 0.0
                    rewards.append(round(j + self.exact_match_weight * em, 4))
                    jaccards.append(j)
                    exact_matches.append(em)
                    branches.append(reward_result[2].get("branch", "unknown") if isinstance(reward_result, tuple) else "legacy")
                    best_gen_ids.append(gen_ids[i])

            rewards = torch.tensor(rewards, device=inputs["input_ids"].device, dtype=torch.float32)

            if self.critic is not None and not self.disable_critic:
                critic_pred = self.critic(encoder_hidden.detach())
                advantage = rewards - critic_pred
            else:
                advantage = rewards - self.reward_ema

            batch_mean_reward = rewards.mean().item()
            self.reward_ema = (1 - self.reward_ema_alpha) * self.reward_ema + self.reward_ema_alpha * batch_mean_reward
            jac_str = ",".join(f"{j:.2f}" for j in jaccards)
            em_str = ",".join(str(int(e)) for e in exact_matches)
            rwd_str = ",".join(f"{r:.2f}" for r in rewards.tolist())
            if self.critic is not None and not self.disable_critic:
                crit_str = ",".join(f"{c:.2f}" for c in critic_pred.tolist())
                adv_str = ",".join(f"{a:.2f}" for a in advantage.tolist())
                print(f"  [R] step={self.state.global_step} jac=[{jac_str}] em=[{em_str}] rwd=[{rwd_str}] crit=[{crit_str}] adv=[{adv_str}] ema={self.reward_ema:.3f} ce={ce_loss.item():.4f}", flush=True)
            else:
                adv_str = ",".join(f"{a:.2f}" for a in advantage.tolist())
                print(f"  [R] step={self.state.global_step} jac=[{jac_str}] em=[{em_str}] rwd=[{rwd_str}] adv=[{adv_str}] ema={self.reward_ema:.3f} ce={ce_loss.item():.4f}", flush=True)
            print(f"  [R_DETAIL] branches={branches}", flush=True)

            # Stream best-per-sample generations to the laptop rolling sink
            # (rank 0 only, fail-open: RollStream never raises).
            try:
                if (getattr(self, "roll_stream", None) is not None
                        and is_main_process()):
                    _step = int(getattr(getattr(self, "state", None),
                                        "global_step", 0))
                    _best_texts = self.reinforce_tokenizer.batch_decode(
                        best_gen_ids, skip_special_tokens=True)
                    self.roll_stream.add(build_roll_rows(
                        _step, _best_texts, gold_texts,
                        [rewards[i].item() for i in range(batch_size)],
                        branches, databases, input_texts))
            except Exception as _e:
                print(f"  [RollStream] skipped ({_e})", flush=True)

            best_gen_ids = torch.stack(best_gen_ids)
            decoder_input_ids = best_gen_ids[:, :-1]
            target_ids = best_gen_ids[:, 1:]
            outputs2 = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
            log_probs = F.log_softmax(outputs2.logits, dim=-1)
            probs = log_probs.exp()
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            mask = (target_ids != self.reinforce_tokenizer.pad_token_id).float()
            seq_log_prob = (token_log_probs * mask).sum(dim=-1)

            reinforce_loss = -(seq_log_prob * advantage.detach()).mean()
            reinforce_loss = torch.nan_to_num(reinforce_loss, nan=0.0, posinf=0.0, neginf=0.0)
            total_loss = self.sft_coef * ce_loss + self.reinforce_coef * reinforce_loss

            entropy = (-(probs * log_probs).sum(dim=-1) * mask).sum() / mask.sum().clamp_min(1.0)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            if self.entropy_coef:
                entropy_bonus = torch.clamp(entropy - self.entropy_min, min=0.0)
                total_loss = total_loss - self.entropy_coef * entropy_bonus

            if self.kl_coef and self.kl_ref_model is not None:
                with torch.no_grad():
                    ref_outputs = self.kl_ref_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                    )
                    ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
                kl_tokens = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
                kl_loss = (kl_tokens * mask).sum() / mask.sum().clamp_min(1.0)
                kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss = total_loss + self.kl_coef * kl_loss
                del ref_outputs, ref_log_probs, kl_tokens
            else:
                kl_loss = torch.tensor(0.0, device=total_loss.device)

            if self.critic is not None and not self.disable_critic:
                critic_loss = F.mse_loss(critic_pred, rewards)
                critic_loss = torch.nan_to_num(critic_loss, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss = total_loss + 0.1 * critic_loss

            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  [RL_LOSS] reinforce={reinforce_loss.item():.4f} entropy={entropy.item():.4f} kl={kl_loss.item():.4f} total={total_loss.item():.4f}", flush=True)
            del outputs2, log_probs, probs

            return (total_loss, None) if return_outputs else total_loss

        return (ce_loss, None) if return_outputs else ce_loss
        del outputs2
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        del log_probs
        mask = (target_ids != self.reinforce_tokenizer.pad_token_id).float()
        seq_log_prob = (token_log_probs * mask).sum(dim=-1)

        advantage = rewards - self.reward_ema
        reinforce_loss = -(seq_log_prob * advantage).mean()
        total_loss = ce_loss + self.reinforce_coef * reinforce_loss
        return total_loss


def run_training_phase(
    model, train_dataset, val_dataset, collator, tokenizer,
    output_dir, epochs, lr, batch_size, grad_accum,
    max_target_length, predict_with_generate, seed, vram_limit_gb,
    resume_checkpoint=None, phase_name="phase1", min_epochs=0,
    label_smoothing=0.0, weight_decay=0.01, neftune_noise_alpha=5.0,
    eval_steps=200, eval_batch_size=0, use_reinforce=False, neo4j_client=None, roll_stream=None, reinforce_coef=0.3,
    reinforce_candidates=1, critic_lr=1e-4, disable_critic=False,
    reward_ema_alpha=0.1,
    entropy_coef=0.0, entropy_min=0.0, exact_match_weight=0.3,
    train_temperature=0.7, train_temperature_min=0.7,
    train_noise_scale=0.0, train_top_k=50, temp_spread=0.0,
    temp_spread_min=0.0, kl_coef=0.0, ppo_clip_eps=0.0,
    kl_ref_model=None, kl_target=1.0, kl_adaptive_alpha=0.0,
    kl_beta_min=0.0, kl_beta_max=1.0, kl_prefix_len=0,
    kl_ref_temperature=1.0, sft_coef=1.0,
    eos_suppress_strength=0.0, eos_suppress_full_until=0,
    eos_suppress_decay_until=0, chunk_size=1,
    diversity_coef=0.0, simplicity_coef=0.0,
    consistency_coef=0.0, dynamic_batch=False, max_tokens=50000,
    length_bucket_size=256, sample_eval_interval=0,
    dynamic_generation_length=False, generation_length_tolerance=0.10,
    dataloader_num_workers=4, dataloader_pin_memory=True,
    use_bf16=False, use_fp16=False,
    max_steps=0, save_steps=200, hf_repo_id="", hf_token="", hf_path_prefix="",
    eval_after_resume=False, logging_steps=50,
):
    def _compute_metrics(eval_pred):
        return compute_metrics(eval_pred, tokenizer, neo4j_client=neo4j_client)

    compute_metrics_fn = _compute_metrics if predict_with_generate else None
    
    # Default eval_batch_size to train batch_size if not provided
    if eval_batch_size == 0:
        eval_batch_size = batch_size

    using_sample_eval = sample_eval_interval and sample_eval_interval > 0
    effective_eval_steps = 10**9 if using_sample_eval else eval_steps
    effective_save_steps = 10**9 if using_sample_eval else save_steps
    load_best = effective_save_steps % effective_eval_steps == 0

    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        label_smoothing_factor=label_smoothing,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_steps=effective_eval_steps,
        save_strategy="steps",
        save_steps=effective_save_steps,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_execution_accuracy" if (predict_with_generate and neo4j_client is not None) else ("eval_token_error_rate" if predict_with_generate else "eval_loss"),
        greater_is_better=neo4j_client is not None if predict_with_generate else not predict_with_generate,
        predict_with_generate=predict_with_generate,
        generation_max_length=max_target_length,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        report_to="none",
        seed=seed,
        optim="adafactor",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    if max_steps and max_steps > 0:
        training_kwargs["max_steps"] = int(max_steps)

    # PEFT checkpoints save adapter weights, but this Transformers build expects
    # a full pytorch_model.bin when load_best_model_at_end=True.
    if load_best and is_peft_wrapped_model(model):
        training_kwargs["load_best_model_at_end"] = False
        if is_main_process():
            print("load_best_model_at_end disabled for PEFT checkpoints (adapter-only saves); best checkpoint path is still tracked.")

    # Support both naming variants across Transformers versions.
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"

    if "length_column_name" in ta_params:
        training_kwargs["length_column_name"] = "length"

    if "bf16_full_eval" in ta_params:
        training_kwargs["bf16_full_eval"] = bool(use_bf16)

    if "fp16_full_eval" in ta_params:
        training_kwargs["fp16_full_eval"] = bool(use_fp16)

    if "ddp_find_unused_parameters" in ta_params:
        training_kwargs["ddp_find_unused_parameters"] = False

    if "neftune_noise_alpha" in ta_params:
        training_kwargs["neftune_noise_alpha"] = neftune_noise_alpha

    if "logging_steps" in ta_params:
        training_kwargs["logging_steps"] = logging_steps
    if "logging_strategy" in ta_params:
        training_kwargs["logging_strategy"] = "steps"
    if "logging_first_step" in ta_params:
        training_kwargs["logging_first_step"] = True
    if "disable_tqdm" in ta_params:
        training_kwargs["disable_tqdm"] = False
    if "log_level" in ta_params:
        training_kwargs["log_level"] = "info"

    if "warmup_ratio" in ta_params:
        training_kwargs["warmup_ratio"] = 0.06
    else:
        effective_batch = max(batch_size * grad_accum, 1)
        est_total_steps = max(int((len(train_dataset) * epochs) / effective_batch), 1)
        warmup_steps = max(int(est_total_steps * 0.06), 1)
        training_kwargs["warmup_steps"] = warmup_steps
        if is_main_process():
            print(f"warmup_ratio unsupported in this transformers version; using warmup_steps={warmup_steps}")

    if "dataloader_persistent_workers" in ta_params and dataloader_num_workers > 0:
        training_kwargs["dataloader_persistent_workers"] = True
    if "dataloader_prefetch_factor" in ta_params and dataloader_num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = 2

    try:
        training_args = Seq2SeqTrainingArguments(**training_kwargs, group_by_length=True)
    except TypeError:
        if is_main_process():
            print("group_by_length not supported in this Transformers version; skipping")
        training_args = Seq2SeqTrainingArguments(**training_kwargs)

    if is_main_process():
        print(f"  Eval dataset: {len(val_dataset)} samples")
    sample_eval_callback = SampleCountEvalCallback(sample_eval_interval) if using_sample_eval else None
    callbacks = [CacheClearEvalCallback(), StepCacheClearCallback()] 
    if sample_eval_callback is not None:
        callbacks.append(sample_eval_callback)
    callbacks += ([MinEpochEarlyStoppingCallback(patience=5, min_epochs=min_epochs)] if min_epochs > 0 else [])
    callbacks += ([SaveBeforeEvalCallback(hf_repo_id, hf_token, hf_path_prefix)] if hf_repo_id else [])
    callbacks += ([HFHubUploadCallback(hf_repo_id, hf_token, hf_path_prefix)] if hf_repo_id else [])

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        vram_limit_gb=vram_limit_gb,
        dynamic_batch=dynamic_batch,
        max_tokens=max_tokens,
        length_bucket_size=length_bucket_size,
        sample_eval_interval=sample_eval_interval,
    )
    if use_reinforce and neo4j_client is not None:
        trainer = ReinforceTrainer(
            neo4j_client=neo4j_client,
            roll_stream=roll_stream,
            reinforce_coef=reinforce_coef,
            reinforce_tokenizer=tokenizer,
            reinforce_candidates=reinforce_candidates,
            reward_ema_alpha=reward_ema_alpha,
            critic_lr=critic_lr,
            disable_critic=disable_critic,
            entropy_coef=entropy_coef,
            entropy_min=entropy_min,
            exact_match_weight=exact_match_weight,
            train_temperature=train_temperature,
            train_temperature_min=train_temperature_min,
            train_noise_scale=train_noise_scale,
            train_top_k=train_top_k,
            temp_spread=temp_spread,
            temp_spread_min=temp_spread_min,
            kl_coef=kl_coef,
            ppo_clip_eps=ppo_clip_eps,
            kl_ref_model=kl_ref_model,
            kl_target=kl_target,
            kl_adaptive_alpha=kl_adaptive_alpha,
            kl_beta_min=kl_beta_min,
            kl_beta_max=kl_beta_max,
            kl_prefix_len=kl_prefix_len,
            kl_ref_temperature=kl_ref_temperature,
            sft_coef=sft_coef,
            eos_suppress_strength=eos_suppress_strength,
            eos_suppress_full_until=eos_suppress_full_until,
            eos_suppress_decay_until=eos_suppress_decay_until,
            chunk_size=chunk_size,
            diversity_coef=diversity_coef,
            simplicity_coef=simplicity_coef,
            consistency_coef=consistency_coef,
            rl_warmup_steps=getattr(args, 'rl_warmup_steps', 0),
            dynamic_generation_length=dynamic_generation_length,
            generation_length_tolerance=generation_length_tolerance,
            **trainer_kwargs,
        )
    else:
        trainer = VRAMSafeTrainer(**trainer_kwargs)
    if sample_eval_callback is not None:
        sample_eval_callback.trainer = trainer

    # In-loop eval generation must be greedy: base generation_config ships
    # do_sample=True/top_k/top_p, which routes Trainer prediction_step through
    # torch.multinomial and device-asserts on degenerate fp16 probs.
    # Explicit sampling callers (RL, final generation eval) pass their own kwargs.
    try:
        if getattr(model.generation_config, "do_sample", False):
            model.generation_config.do_sample = False
            print("[EvalGen] Forced greedy in-loop eval generation (do_sample=False).")
    except Exception as e:
        print(f"[EvalGen] Could not force greedy eval generation: {e}")

    if eval_after_resume and resume_checkpoint:
        print(f"\n[ResumeEval] Running loss eval immediately after resume from {resume_checkpoint}...")
        try:
            resume_metrics = trainer.evaluate()
            resume_path = os.path.join(output_dir, "resume_eval_metrics.json")
            with open(resume_path, "w", encoding="utf-8") as f:
                json.dump(resume_metrics, f, indent=2, default=str)
            print(f"[ResumeEval] Metrics: {resume_metrics}")
            print(f"[ResumeEval] Saved to {resume_path}")
        except Exception as e:
            print(f"[ResumeEval] Post-resume eval failed (continuing to training): {e}")

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  {phase_name}: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"  lr={lr}, epochs={epochs}, batch={batch_size}x{grad_accum}")
        if dynamic_batch:
            print(f"  dynamic_batch=True, max_tokens={max_tokens}, length_bucket_size={length_bucket_size}")
        if using_sample_eval:
            print(f"  sample_eval_interval={sample_eval_interval} samples (regular eval_steps disabled)")
    effective_batch = batch_size * grad_accum
    total_steps = int((len(train_dataset) * epochs) / effective_batch)
    est_minutes = (total_steps * 3) / 60
    if is_main_process():
        print(f"  Steps: {total_steps}, est: {est_minutes:.0f}m ({est_minutes/60:.1f}h)")
        print(f"{'='*60}\n")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        print(f"\n{phase_name} complete. Metrics: {train_result.metrics}")
    except RuntimeError as e:
        _save_crash_checkpoint(trainer, output_dir, phase_name,
                               hf_repo_id=hf_repo_id, hf_token=hf_token,
                               hf_path_prefix=hf_path_prefix)
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            print(f"\n*** OOM in {phase_name} ***")
            print(f"VRAM: {get_vram_info()}")
            last_ckpt = find_latest_checkpoint(output_dir)
            if last_ckpt:
                print(f"  Last checkpoint: {last_ckpt}")
            return None
        # Non-OOM error: print full details before re-raising
        import traceback
        print(f"\n*** {phase_name} crashed (non-OOM) ***")
        print(f"Error type: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and is_main_process():
        best_checkpoint_copy = os.path.join(output_dir, "best_checkpoint")
        if os.path.abspath(best_checkpoint) != os.path.abspath(best_checkpoint_copy):
            if os.path.isdir(best_checkpoint_copy):
                shutil.rmtree(best_checkpoint_copy)
            shutil.copytree(best_checkpoint, best_checkpoint_copy)
        print(f"Best checkpoint selected by Trainer: {best_checkpoint}")
        print(f"Best checkpoint copied to {best_checkpoint_copy}")
    elif best_checkpoint is None and is_main_process():
        print("No Trainer best checkpoint was selected; saving final model as best. Enable aligned save/eval steps for true best-checkpoint selection.")

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")
    return best_dir


def _save_crash_checkpoint(trainer, output_dir, phase_name,
                           hf_repo_id="", hf_token="", hf_path_prefix=""):
    """Best-effort: persist + upload current weights so ANY crash is resumable.

    Uses standard checkpoint-{step} naming so --resume_checkpoint picks it up.
    After a device-side assert the CUDA context is poisoned and GPU-tensor
    reads fail; OOM and CPU-side errors save fine. Never raises.
    """
    try:
        step = getattr(getattr(trainer, "state", None), "global_step", 0)
    except Exception:
        step = 0
    try:
        trainer._save_checkpoint(trainer.model, None)
        print(f"[CrashSave] {phase_name} weights saved at step {step} "
              f"(resumable via --resume_checkpoint).")
    except Exception as e:
        print(f"[CrashSave] Weight save failed (CUDA context may be poisoned): {e}")
        return
    if HF_AVAILABLE and hf_repo_id:
        crash_dir = os.path.join(output_dir, f"checkpoint-{step}")
        try:
            upload_checkpoint_to_hub(crash_dir, hf_repo_id, hf_token=hf_token,
                                     hf_path_prefix=hf_path_prefix)
            print(f"[CrashSave] Crash checkpoint uploaded to {hf_repo_id}.")
        except Exception as e:
            print(f"[CrashSave] Crash upload failed: {e}")


def login_to_hf(hf_token):
    """Login to Hugging Face Hub."""
    if not HF_AVAILABLE or not hf_token:
        return
    try:
        login(token=hf_token)
        print("✓ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"Failed to login to HF: {e}")


def join_hf_path(prefix, name):
    prefix = (prefix or "").strip("/")
    return f"{prefix}/{name}" if prefix else name


def download_checkpoint_from_hub(hf_repo_id, local_dir):
    """Download latest checkpoint from HF Hub repo."""
    if not HF_AVAILABLE or not hf_repo_id:
        return None
    try:
        print(f"Checking for checkpoints in {hf_repo_id}...")
        api = HfApi()
        repo_files = api.list_repo_files(hf_repo_id, repo_type="model")
        checkpoints = sorted(
            [f.rfilename.split('/')[0] for f in repo_files 
             if f.rfilename.startswith("checkpoint-") and '/' in f.rfilename],
            key=lambda x: int(x.split('-')[1])
        )
        if not checkpoints:
            print(f"No checkpoints found in {hf_repo_id}")
            return None
        
        latest = checkpoints[-1]
        print(f"Found latest checkpoint: {latest}")
        print(f"Downloading from {hf_repo_id}/{latest}...")
        
        checkpoint_path = Path(local_dir) / latest
        snapshot_download(
            hf_repo_id, 
            repo_type="model",
            allow_patterns=f"{latest}/*",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        if checkpoint_path.exists():
            print(f"✓ Downloaded to {checkpoint_path}")
            return str(checkpoint_path)
        else:
            print(f"Download completed but path not found: {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Failed to download from hub: {e}")
        return None
        return None


def upload_checkpoint_to_hub(checkpoint_path, hf_repo_id, hf_token=None, hf_path_prefix=""):
    """Upload a checkpoint to HF Hub repo."""
    if not HF_AVAILABLE:
        print("huggingface_hub not installed; skipping upload.")
        return
    try:
        print(f"Uploading {checkpoint_path} to {hf_repo_id}...")
        _upload_folder_resilient(
            repo_id=hf_repo_id,
            folder_path=checkpoint_path,
            path_in_repo=join_hf_path(hf_path_prefix, Path(checkpoint_path).name),
            token=hf_token,
        )
        print(f"Uploaded {checkpoint_path}")
    except Exception as e:
        print(f"Failed to upload to hub: {e}")


def upload_rolling_data_to_hf(output_dir, hf_repo_id, hf_token=None, hf_path_prefix=""):
    """Upload rolling eval logs next to the model/checkpoints on HF."""
    if not HF_AVAILABLE or not hf_repo_id:
        return
    for filename in ("rolling_data.jsonl", "rolling_data.json",
                       "final_eval_metrics.json", "final_test_metrics.json",
                       "resume_eval_metrics.json",
                       "run_config.json", "training_args.json"):
        local_path = Path(output_dir) / filename
        if not local_path.exists():
            continue
        try:
            remote_path = join_hf_path(hf_path_prefix, filename)
            upload_file(
                repo_id=hf_repo_id,
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                token=hf_token,
                repo_type="model",
            )
            print(f"[HF Upload] Uploaded rolling eval log to {hf_repo_id}/{remote_path}")
        except Exception as e:
            print(f"[HF Upload] Failed to upload rolling eval log {local_path}: {e}")


def _upload_folder_resilient(repo_id, folder_path, path_in_repo, token=None):
    """
    Upload folder to HF Hub with a fallback for invalid model-card metadata.
    Some PEFT checkpoints include README metadata (e.g. local base_model path)
    that Hub validation rejects. Retry without README.md in that case.
    """
    try:
        upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            token=token,
            repo_type="model",
        )
    except Exception as e:
        msg = str(e)
        if "Invalid metadata in README.md" in msg or "base_model" in msg:
            print("[HF Upload] README metadata rejected by Hub; retrying without README.md")
            upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                token=token,
                repo_type="model",
                ignore_patterns=["README.md"],
            )
        else:
            raise


def get_hf_top_level_dirs(hf_repo_id, hf_token=None, path_prefix=""):
    """Return top-level directory names present in HF model repo."""
    if not HF_AVAILABLE or not hf_repo_id:
        return set()
    try:
        api = HfApi()
        repo_files = api.list_repo_files(hf_repo_id, repo_type="model", token=hf_token)
        top_dirs = set()
        prefix = (path_prefix or "").strip("/")
        for f in repo_files:
            if prefix:
                if not f.startswith(prefix + "/"):
                    continue
                rel = f[len(prefix) + 1:]
                if "/" in rel:
                    top_dirs.add(rel.split("/", 1)[0])
            elif "/" in f:
                top_dirs.add(f.split("/", 1)[0])
        return top_dirs
    except Exception as e:
        print(f"[HF Sync] Failed to list repo files for {hf_repo_id}: {e}")
        return set()


def sync_missing_local_to_hf(output_dir, hf_repo_id, hf_token=None, include_best=True, hf_path_prefix=""):
    """Upload local model dirs that don't exist in HF repo yet."""
    if not HF_AVAILABLE or not hf_repo_id:
        return

    output_path = Path(output_dir)
    if not output_path.exists():
        return

    remote_dirs = get_hf_top_level_dirs(hf_repo_id, hf_token=hf_token, path_prefix=hf_path_prefix)

    local_dirs = [p for p in output_path.glob("checkpoint-*") if p.is_dir()]
    if include_best:
        best_dir = output_path / "best"
        if best_dir.is_dir():
            local_dirs.append(best_dir)

    missing = [p for p in sorted(local_dirs, key=lambda x: x.name) if p.name not in remote_dirs]
    if not missing:
        print("[HF Sync] No missing local checkpoints to upload.")
        return

    print(f"[HF Sync] Uploading {len(missing)} local folders missing on HF...")
    for p in missing:
        upload_checkpoint_to_hub(str(p), hf_repo_id, hf_token=hf_token, hf_path_prefix=hf_path_prefix)


def retain_best_recent_checkpoints(output_dir, keep_best=2, keep_recent=3):
    """Keep only the best N and most recent M checkpoints; delete others."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    checkpoints = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                        key=lambda x: int(x.name.split('-')[1]))
    
    if len(checkpoints) <= (keep_best + keep_recent):
        return
    
    # Score checkpoints by eval loss (best) and step number (most recent)
    scored = []
    for cp in checkpoints:
        metrics_file = cp / "trainer_state.json"
        step = int(cp.name.split('-')[1])
        loss = float('inf')
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    state = json.load(f)
                # Prefer explicit best metric if present.
                bm = state.get("best_metric", None)
                if isinstance(bm, (int, float)):
                    loss = float(bm)
                # Fallback: get eval_loss nearest to this checkpoint step.
                if loss == float('inf'):
                    hist = state.get("log_history", [])
                    best_eval_loss = float('inf')
                    for row in hist:
                        if not isinstance(row, dict):
                            continue
                        if "eval_loss" in row and isinstance(row.get("step", None), int) and row["step"] <= step:
                            try:
                                best_eval_loss = min(best_eval_loss, float(row["eval_loss"]))
                            except Exception:
                                pass
                    loss = best_eval_loss
            except Exception:
                pass
        scored.append((cp, loss, step))
    
    # Keep best N by loss and most recent M by step
    best_cps = sorted(scored, key=lambda x: x[1])[:keep_best]
    recent_cps = sorted(scored, key=lambda x: x[2])[-keep_recent:]
    
    keep = set([cp[0] for cp in best_cps + recent_cps])
    
    for cp, _, _ in scored:
        if cp not in keep:
            print(f"Removing old checkpoint: {cp}")
            shutil.rmtree(cp)


def _checkpoint_step(path):
    try:
        return int(Path(path).name.split("-")[1])
    except Exception:
        return -1


def is_adapter_checkpoint_dir(path):
    path = Path(path)
    return (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def is_full_model_dir(path):
    path = Path(path)
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or any(path.glob("model-*.safetensors"))
        or any(path.glob("pytorch_model-*.bin"))
    )
    return has_config and has_weights and not is_adapter_checkpoint_dir(path)


def resolve_eval_model_dir(eval_model_dir, default_output_dir):
    """Resolve an eval-only source to either a full model dir or PEFT adapter checkpoint dir."""
    roots = []
    if eval_model_dir:
        roots.append(Path(eval_model_dir))
    roots.append(Path(default_output_dir))

    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.append(root)
        for child_name in ("best", "best_checkpoint", "merged_eval"):
            child = root / child_name
            if child.is_dir():
                candidates.append(child)
        checkpoints = sorted(
            [p for p in root.glob("checkpoint-*") if p.is_dir()],
            key=_checkpoint_step,
            reverse=True,
        )
        candidates.extend(checkpoints)

    for candidate in candidates:
        if is_full_model_dir(candidate) or is_adapter_checkpoint_dir(candidate):
            return str(candidate)

    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Could not find a full model or adapter checkpoint for eval-only. "
        f"Searched: {searched}. Expected config/model files, adapter_config.json, "
        "best/, best_checkpoint/, merged_eval/, or checkpoint-* directories."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="t5gemma-270m")
    parser.add_argument("--phase2_base_model_path", type=str, default="", help="Base model path to load for Phase 2-only runs")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=0, help="Evaluation batch size (0 = same as train batch_size, no backprop so can be larger)")
    parser.add_argument("--test_batch_size", type=int, default=0, help="Final test generation batch size (0 = same as eval_batch_size)")
    parser.add_argument("--grad_accum", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--max_input_length", type=int, default=0)
    parser.add_argument("--max_target_length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Show resume selection prompt inside train.py")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Explicit checkpoint path to resume from (recommended for notebook-controlled flow)")
    parser.add_argument("--eval_after_resume", action="store_true", help="Run loss eval immediately after resuming from a checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for torchrun/DDP (also read from LOCAL_RANK env)")
    parser.add_argument("--eval_only", action="store_true", help="Only run post-training eval on best checkpoint")
    parser.add_argument("--eval_model_dir", type=str, default="", help="Explicit full model/checkpoint directory for --eval_only. Can point to best/, best_checkpoint/, checkpoint-*, merged_eval/, or a parent containing those.")
    parser.add_argument("--final_test_only", action="store_true", help="With --eval_only, skip validation eval and run only the final test set")
    parser.add_argument("--train_parquet", type=str, default="splits/train_t5.parquet")
    parser.add_argument("--val_parquet", type=str, default="splits/val_answerable.parquet")
    parser.add_argument("--test_parquet", type=str, default="splits/test_answerable.parquet")
    parser.add_argument("--input_format", type=str, default="plain", choices=["plain", "schema_question"], help="How to build model input text")
    parser.add_argument("--input_col", type=str, default="input", help="Input text column when --input_format=plain")
    parser.add_argument("--target_col", type=str, default="target", help="Target text column")
    parser.add_argument("--schema_col", type=str, default="schema", help="Schema column when --input_format=schema_question")
    parser.add_argument("--question_col", type=str, default="question", help="Question column when --input_format=schema_question")
    parser.add_argument("--eval_with_generate", action=argparse.BooleanOptionalAction, default=True, help="Enable generation during eval (uses more VRAM)")
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=0,
        help="Limit val dataset size in eval-only mode (0 = all samples)",
    )
    parser.add_argument("--train_eval_samples", type=int, default=0,
        help="Limit eval dataset size during training (0 = all)")
    parser.add_argument("--train_samples", type=int, default=0, help="Limit train dataset size (0 = all)")
    parser.add_argument("--val_samples", type=int, default=0, help="Limit validation dataset size (0 = all)")
    parser.add_argument("--test_samples", type=int, default=0, help="Limit test dataset size (0 = all)")
    parser.add_argument("--phase1", action=argparse.BooleanOptionalAction, default=True, help="Run phase 1 curriculum (filters parquet phase==1)")
    parser.add_argument("--phase2", action=argparse.BooleanOptionalAction, default=True, help="Run phase 2 curriculum (filters parquet phase==2)")
    parser.add_argument("--phase2_epochs", type=int, default=0, help="Phase 2 epochs (0 = same as --epochs)")
    parser.add_argument("--min-epochs", type=float, default=0, help="Minimum epochs before early stopping can trigger (0 = no minimum)")
    parser.add_argument("--phase2_lr", type=float, default=0, help="Phase 2 learning rate (0 = lr * 0.5)")
    parser.add_argument("--phase2_reinforce", action="store_true", help="Enable REINFORCE for Phase 2")
    parser.add_argument("--reinforce_coef", type=float, default=1.0, help="Weight for REINFORCE loss term")
    parser.add_argument("--reinforce_candidates", type=int, default=1, help="Generate N candidates per input, take best reward")
    parser.add_argument("--rl_warmup_steps", type=int, default=50, help="Number of SFT-only steps before RL kicks in; avoids RL on an untrained model")
    parser.add_argument("--reward_ema_alpha", type=float, default=0.1, help="EMA smoothing for reward baseline")
    parser.add_argument("--entropy_coef", type=float, default=0.0, help="Entropy bonus coefficient for sampled RL tokens")
    parser.add_argument("--entropy_beta", type=float, default=None, help="Alias for --entropy_coef from trainv1")
    parser.add_argument("--entropy_min", type=float, default=0.0, help="Only reward entropy above this floor")
    parser.add_argument("--exact_match_weight", type=float, default=0.3, help="Extra reward added for exact text match")
    parser.add_argument("--em_weight", type=float, default=None, help="Alias for --exact_match_weight from trainv1")
    parser.add_argument("--train_temperature", type=float, default=0.7, help="Sampling temperature for Phase 2 RL candidates")
    parser.add_argument("--train_temperature_min", type=float, default=0.7, help="Minimum/annealed floor for Phase 2 RL candidate temperature")
    parser.add_argument("--train_noise_scale", type=float, default=0.0, help="Gaussian logit noise added during RL candidate generation")
    parser.add_argument("--train_top_k", type=int, default=50, help="Top-k sampling cutoff for Phase 2 RL candidates")
    parser.add_argument("--top_k", type=int, default=None, help="Alias for --train_top_k from trainv1")
    parser.add_argument("--temp_spread", type=float, default=0.0, help="Multiplicative temperature spread between candidate rounds")
    parser.add_argument("--temp_spread_min", type=float, default=0.0, help="Minimum temperature spread after annealing")
    parser.add_argument("--kl_coef", type=float, default=0.0, help="KL penalty coefficient against the frozen reference model")
    parser.add_argument("--kl_beta", type=float, default=None, help="Alias for --kl_coef from trainv1")
    parser.add_argument("--kl_ref_model_path", type=str, default="", help="Optional reference model path for KL penalty (defaults to Phase 2 base model when KL is enabled)")
    parser.add_argument("--kl_target", type=float, default=1.0, help="Adaptive KL target from trainv1; accepted for tuning/logging")
    parser.add_argument("--kl_adaptive_alpha", type=float, default=0.0, help="Adaptive KL adjustment rate from trainv1; 0 keeps current fixed KL behavior")
    parser.add_argument("--kl_beta_min", type=float, default=0.0, help="Minimum KL coefficient from trainv1")
    parser.add_argument("--kl_beta_max", type=float, default=1.0, help="Maximum KL coefficient from trainv1")
    parser.add_argument("--kl_ref_temperature", type=float, default=1.0, help="Temperature applied to KL reference logits when implemented")
    parser.add_argument("--kl_prefix_len", type=int, default=0, help="Only apply KL to first N tokens when implemented (0=all)")
    parser.add_argument("--sft_coef", type=float, default=1.0, help="Weight for supervised CE loss during RL")
    parser.add_argument("--ppo_clip_eps", type=float, default=0.0, help="PPO clip epsilon from trainv1; 0 keeps current unclipped REINFORCE path")
    parser.add_argument("--chunk_size", type=int, default=1, help="Forward/KL chunk size from trainv1; current lightweight path keeps full selected batch")
    parser.add_argument("--eos_suppress_strength", type=float, default=0.0, help="EOS suppression strength from trainv1; 0 disables")
    parser.add_argument("--eos_suppress_full_until", type=int, default=0, help="EOS full suppression token count from trainv1")
    parser.add_argument("--eos_suppress_decay_until", type=int, default=0, help="EOS suppression decay token count from trainv1")
    parser.add_argument("--diversity_coef", type=float, default=0.0, help="Diversity bonus coefficient from trainv1; 0 disables")
    parser.add_argument("--simplicity_coef", type=float, default=0.0, help="Simplicity bonus coefficient from trainv1; 0 disables")
    parser.add_argument("--consistency_coef", type=float, default=0.0, help="Consistency bonus coefficient from trainv1; 0 disables")
    parser.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=False, help="Token-budgeted REINFORCE batching from trainv1; disabled by default in lightweight path")
    parser.add_argument("--max_tokens", type=int, default=50000, help="Token budget knob from trainv1")
    parser.add_argument("--length_bucket_size", type=int, default=256, help="Number of examples per shuffle/sort pool for dynamic token-length batching")
    parser.add_argument("--sample_eval_interval", type=int, default=0, help="Evaluate/save after this many training samples have crossed since the last eval; 0 uses eval_steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=True, help="Pin DataLoader memory for faster host-to-GPU transfer")
    parser.add_argument("--dynamic_generation_length", action=argparse.BooleanOptionalAction, default=False, help="Gold-relative generation length control from trainv1; disabled by default")
    parser.add_argument("--generation_length_tolerance", type=float, default=0.10, help="Tolerance for dynamic generation length from trainv1")
    parser.add_argument("--eval_sampling_samples", type=int, default=0, help="Sampling eval sample count from trainv1; 0 disables")
    parser.add_argument("--rolling_json_max_records", type=int, default=1000, help="Number of latest rolling eval rows to keep in rolling_data.json")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic network learning rate")
    parser.add_argument("--disable_critic", action="store_true", help="Disable critic baseline, fall back to EMA")
    parser.add_argument("--neo4j_url", type=str, default="", help="Neo4j URL (supports bolt://, neo4j://, http://, https://). If empty, defaults to http://localhost:<neo4j_port>/db/neo4j/tx/commit")
    parser.add_argument("--neo4j_port", type=int, default=15964, help="Neo4j HTTP port")
    parser.add_argument("--neo4j_user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default="berttokg123", help="Neo4j password")
    parser.add_argument("--neo4j_database", type=str, default="neo4j", help="Default Neo4j database used for endpoint tests and queries without a sample database")
    parser.add_argument("--neo4j_databases", type=str, default="", help="Comma/space-separated database names that every discovered Neo4j endpoint must support")
    parser.add_argument("--neo4j_database_mapping_path", type=str, default="", help="JSON file mapping dataset database column values to Neo4j database names")
    parser.add_argument("--neo4j_require_all_mapped_databases", action="store_true", help="During discovery, require every mapped Neo4j database to respond before an endpoint is selectable")
    parser.add_argument("--neo4j_ips", type=str, default="", help="Comma/space-separated Neo4j IPs or IP:port values to try for HTTPS Query API and Bolt")
    parser.add_argument("--neo4j_urls", type=str, default="", help="Comma/space-separated Neo4j hostnames or URLs to try for HTTPS Query API and Bolt")
    parser.add_argument("--neo4j_ip_ports", type=str, default="", help="Comma/space-separated Bolt ports to combine with --neo4j_ips/--neo4j_urls; defaults to 7687")
    parser.add_argument("--neo4j_url_ports", type=str, default="", help="Comma/space-separated HTTPS ports to combine with --neo4j_ips/--neo4j_urls; empty means use the URL without an explicit port")
    parser.add_argument("--neo4j_discovery_timeout", type=float, default=5.0, help="Seconds to wait for each Neo4j endpoint probe")
    parser.add_argument("--neo4j_auto_select", action="store_true", help="Use the first reachable Neo4j endpoint without prompting when multiple are available")
    parser.add_argument("--neo4j_skip_discovery", action="store_true", help="Skip train.py Neo4j discovery because the notebook already preflighted and selected --neo4j_url")
    parser.add_argument("--neo4j_proxies", type=str, default="", help="Optional proxy URL applied to http+https (e.g. socks5://localhost:1055 for userspace-tailscale Kaggle)")
    parser.add_argument("--roll_ws_url", type=str, default="", help="Rolling-stream websocket URL ('' disables streaming RL generations to laptop)")
    parser.add_argument("--roll_ws_auth", type=str, default="", help="Token for rolling-stream messages")
    parser.add_argument("--roll_ws_proxy", type=str, default="", help="Optional socks5://host:port for the rolling stream (userspace-tailscale Kaggle)")
    parser.add_argument("--roll_flush_every", type=int, default=50, help="Rolling-stream rows per flush")
    parser.add_argument("--neo4j_preflight_mode", type=str, default="fallback", choices=["fallback", "dataset", "all_mapped"], help="Preflight DB check mode: fallback=only default DB, dataset=only mapped DBs in data, all_mapped=all DBs in mapping")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--neftune_noise_alpha", type=float, default=5.0, help="NEFTune noise alpha")
    parser.add_argument("--merge_phase1", action="store_true", help="Merge Phase 1 LoRA before Phase 2")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N optimizer steps (0 = unlimited)")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps (default: 200)")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps (default: 200)")
    parser.add_argument("--compile_model", action="store_true", default=False, help="Compile the train model after LoRA attachment with torch.compile when available")
    parser.add_argument("--use_quant", action="store_true", help="Enable 4-bit quantization (default: BF16 full precision)")
    parser.add_argument("--compile_mode", type=str, default="default", help="torch.compile mode for --compile_model (default: 'default'; 'reduce-overhead' uses CUDAGraphs which cause partitioning with custom ops on T4)")
    parser.add_argument("--compile_eval", action="store_true", default=False, help="Compile merged eval model (can increase overhead for tiny eval sets)")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log training metrics every N steps (default: 50)")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Numeric precision policy for train/eval model loading")
    parser.add_argument("--hf_repo_id", type=str, default="", help="HuggingFace repo ID for uploading checkpoints (e.g., username/repo-name)")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace API token for uploading")
    parser.add_argument("--hf_path_prefix", type=str, default="", help="Optional HF repo subdirectory for checkpoint/model sync, e.g. phase_2")
    parser.add_argument("--upload_checkpoints", action="store_true", help="Upload checkpoints to HF Hub after training")
    args = parser.parse_args()

    # DDP: pin this process to its GPU. torchrun sets LOCAL_RANK/WORLD_SIZE;
    # single-process runs leave them unset (rank 0, world 1).
    try:
        _ddp_local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank if args.local_rank >= 0 else 0)))
        _ddp_world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        _ddp_local_rank, _ddp_world = 0, 1
    if _ddp_world > 1 and torch.cuda.is_available():
        torch.cuda.set_device(_ddp_local_rank % max(torch.cuda.device_count(), 1))
        print(f"[DDP] rank {os.environ.get('RANK', _ddp_local_rank)}/{_ddp_world} pinned to cuda:{torch.cuda.current_device()}", flush=True)

    if args.entropy_beta is not None:
        args.entropy_coef = args.entropy_beta
    if args.em_weight is not None:
        args.exact_match_weight = args.em_weight
    if args.top_k is not None:
        args.train_top_k = args.top_k
    if args.kl_beta is not None:
        args.kl_coef = args.kl_beta

    if not is_main_process():
        hf_logging.set_verbosity_error()

    use_bf16, use_fp16, model_dtype = resolve_precision_mode(args.precision)
    if is_main_process():
        if use_bf16:
            precision_label = "bf16"
        elif use_fp16:
            precision_label = "fp16"
        else:
            precision_label = "fp32"
        print(f"Precision policy: {args.precision} -> {precision_label}")

    args.neo4j_database_mapping = load_neo4j_database_mapping(args.neo4j_database_mapping_path)
    if args.phase2_reinforce:
        resolve_neo4j_connection(args)

    os.environ["PYTHONUTF8"] = "1"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "_C") and hasattr(torch._C, "_set_float32_matmul_precision"):
        torch._C._set_float32_matmul_precision("high")

    vram_before = get_vram_info()
    print(f"GPU VRAM: {vram_before['total_gb']:.1f} GB total, {vram_before['free_gb']:.1f} GB free")

    base_model_path = args.model_path
    if args.phase2 and not args.phase1 and args.phase2_base_model_path:
        base_model_path = args.phase2_base_model_path
        print(f"Phase 2-only run: using Phase 2 base model path: {base_model_path}")

    hf_path_prefix = args.hf_path_prefix.strip("/")
    if not hf_path_prefix and args.phase2 and not args.phase1:
        hf_path_prefix = "phase_2"
    if hf_path_prefix:
        print(f"HF checkpoint/model sync prefix: {hf_path_prefix}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    if args.use_quant:
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            local_files_only=True,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
        )
    else:
        print("Loading model in configured precision (no quantization)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=model_dtype,
        )
        model = unify_model_dtype(model, model_dtype, "train")
    
    if not hasattr(model.config, 'decoder_start_token_id') or model.config.decoder_start_token_id is None:
        fallback = getattr(getattr(model.config, "decoder", None), "bos_token_id", None)
        if fallback is not None:
            model.config.decoder_start_token_id = fallback
            print(f"Set decoder_start_token_id = {fallback} (from decoder.bos_token_id)")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    vram_after_model = get_vram_info()
    free_after_model = vram_after_model["free_gb"]
    print(f"VRAM after model load: {vram_after_model['reserved_gb']:.1f}/{vram_after_model['total_gb']:.1f} GB (free: {free_after_model:.1f} GB)")

    auto = auto_tune_params(free_after_model)

    print(f"\n=== Auto-tuned parameters ===")
    print(f"  Free VRAM after model load: {free_after_model:.1f} GB")
    print(f"  batch_size    = {auto['batch_size']}")
    print(f"  grad_accum    = {auto['grad_accum']}")
    print(f"  max_input_len = {auto['max_in']}")
    print(f"  lora_r        = {auto['lora_r']}")
    print(f"  max_tgt_len   = {auto['max_tgt']}")
    print(f"  Reason: {auto['reason']}")
    print(f"=============================\n")

    batch_size = args.batch_size if args.batch_size > 0 else auto["batch_size"]
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else batch_size
    test_batch_size = args.test_batch_size if args.test_batch_size > 0 else eval_batch_size
    grad_accum = args.grad_accum if args.grad_accum > 0 else auto["grad_accum"]
    max_input_length = args.max_input_length if args.max_input_length > 0 else auto["max_in"]
    max_target_length = (
        args.max_target_length if args.max_target_length > 0 else auto["max_tgt"]
    )
    lora_r = args.lora_r if args.lora_r > 0 else auto["lora_r"]

    predict_with_generate = args.eval_with_generate

    print(f"Config: train_batch={batch_size}, eval_batch={eval_batch_size}, test_batch={test_batch_size}, grad_accum={grad_accum}, max_in={max_input_length}, "
          f"max_tgt={max_target_length}, lora_r={lora_r}")
    print(f"Effective batch: {batch_size * grad_accum}")
    print(f"predict_with_generate: {predict_with_generate}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=args.lora_dropout,
        target_modules=r"model\.(encoder\.text_model|decoder)\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    )
    model = get_peft_model(model, lora_config)
    if not args.use_quant:
        model = model.to(model_dtype)
    if use_fp16:
        ensure_trainable_params_fp32(model)
    model.print_trainable_parameters()
    if args.compile_model:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled for training ({args.compile_mode})")
        except Exception as e:
            print(f"torch.compile for training unavailable, continuing uncompiled: {e}")

    vram_after_lora = get_vram_info()
    free_after_lora = vram_after_lora["free_gb"]
    print(f"Gradient checkpointing: ENABLED (batch={batch_size}, need activation memory savings)")

    print(f"VRAM after LoRA: {vram_after_lora['reserved_gb']:.1f}/{vram_after_lora['total_gb']:.1f} GB (free: {free_after_lora:.1f} GB)")

    vram_limit = vram_after_lora["total_gb"] * 0.90 if vram_after_lora["total_gb"] > 0 else None

    train_parquet = os.path.join(args.data_dir, args.train_parquet)
    val_parquet = os.path.join(args.data_dir, args.val_parquet)
    test_parquet = os.path.join(args.data_dir, args.test_parquet)

    dataset_kwargs = dict(
        input_format=args.input_format,
        input_col=args.input_col,
        target_col=args.target_col,
        schema_col=args.schema_col,
        question_col=args.question_col,
    )

    if not os.path.exists(val_parquet):
        raise FileNotFoundError(f"Val parquet not found: {val_parquet}")

    val_dataset = Text2CypherDataset(
        val_parquet, tokenizer,
        max_input_length=max_input_length, max_target_length=max_target_length,
        max_samples=args.val_samples,
        sample_seed=args.seed,
        **dataset_kwargs,
    )
    if args.val_samples > 0:
        print(f"Validation subset enabled: {len(val_dataset)} samples")
    collator = Text2CypherCollator(tokenizer)

    # Login to HF Hub if token provided
    if args.hf_token:
        login_to_hf(args.hf_token)

    # Pre-training sync: upload any locally existing checkpoints that are missing on HF.
    if args.upload_checkpoints and args.hf_repo_id and is_main_process():
        sync_missing_local_to_hf(args.output_dir, args.hf_repo_id, hf_token=args.hf_token, include_best=True, hf_path_prefix=hf_path_prefix)

    resume_checkpoint = None
    if args.resume_checkpoint:
        if os.path.isdir(args.resume_checkpoint):
            resume_checkpoint = args.resume_checkpoint
            print(f"\n[Resume] Using explicit checkpoint: {resume_checkpoint}")
        else:
            print(f"\n[Resume] Explicit checkpoint not found: {args.resume_checkpoint}")
            print("[Resume] Starting fresh (no checkpoint loaded)")
    elif args.resume:
        # Interactive resume selection (useful in terminals; notebook flow should pass --resume_checkpoint)
        local_ckpt = find_latest_checkpoint(args.output_dir)
        if local_ckpt or args.hf_repo_id:
            resume_checkpoint, source = prompt_resume_checkpoint(
                local_ckpt, args.hf_repo_id, args.hf_token, args.output_dir
            )
            if resume_checkpoint:
                print(f"\n[Resume] Using {source} checkpoint: {resume_checkpoint}")
            else:
                print("\n[Resume] Starting fresh (no checkpoint loaded)")
        else:
            print("\n[Resume] No checkpoints found. Starting fresh.")
    else:
        print("\n[Resume] Notebook-controlled flow: starting with no implicit resume.")
    
    train_val_dataset = val_dataset
    if args.train_eval_samples > 0 and len(val_dataset) > args.train_eval_samples:
        import random
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(val_dataset)), args.train_eval_samples)
        train_val_dataset = [val_dataset[i] for i in idxs]
        print(f"Training eval: {args.train_eval_samples}/{len(val_dataset)} samples")

    phase2_output_dir = args.output_dir if args.phase2 and not args.phase1 else args.output_dir + "_p2"
    rolling_jsonl_path = os.path.join(phase2_output_dir if args.phase2 else args.output_dir, "rolling_data.jsonl")

    if not args.eval_only:
        best_dir = None
        if args.phase1:
            train_dataset = Text2CypherDataset(
                train_parquet, tokenizer, phase=1,
                max_input_length=max_input_length, max_target_length=max_target_length,
                max_samples=args.train_samples,
                sample_seed=args.seed,
                **dataset_kwargs,
            )
            if args.train_samples > 0:
                print(f"Train subset enabled: {len(train_dataset)} samples")
            best_dir = run_training_phase(
                model, train_dataset, train_val_dataset, collator, tokenizer,
                args.output_dir, args.epochs, args.lr,
                batch_size, grad_accum, max_target_length,
                predict_with_generate, args.seed, vram_limit,
                resume_checkpoint=resume_checkpoint, phase_name="Phase 1",
                min_epochs=args.min_epochs,
                label_smoothing=args.label_smoothing,
                weight_decay=args.weight_decay,
                neftune_noise_alpha=args.neftune_noise_alpha,
                eval_steps=args.eval_steps,
                eval_batch_size=eval_batch_size,
                max_steps=args.max_steps,
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                dynamic_batch=args.dynamic_batch,
                max_tokens=args.max_tokens,
                length_bucket_size=args.length_bucket_size,
                sample_eval_interval=args.sample_eval_interval,
                dataloader_num_workers=args.dataloader_num_workers,
                dataloader_pin_memory=args.dataloader_pin_memory,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
                hf_repo_id=args.hf_repo_id,
                hf_token=args.hf_token,
                hf_path_prefix=hf_path_prefix,
                eval_after_resume=args.eval_after_resume,
            )
            if best_dir is None:
                return
            del train_dataset
            gc.collect()

        if args.phase2:
            merge_source = best_dir if best_dir is not None else resume_checkpoint
            if args.merge_phase1 and merge_source is not None:
                print(f"\n=== Merging Phase 1 LoRA for Phase 2 (source: {merge_source}) ===")
                merged_phase1_dir = os.path.join(args.output_dir, "merged_phase1")
                if not os.path.exists(os.path.join(merged_phase1_dir, "model.safetensors")):
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                    merge_model = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model_path, torch_dtype=model_dtype, attn_implementation="sdpa",
                    )
                    merge_model = unify_model_dtype(merge_model, model_dtype, "merge-phase1")
                    if not hasattr(merge_model.config, 'decoder_start_token_id') or merge_model.config.decoder_start_token_id is None:
                        fallback = getattr(getattr(merge_model.config, "decoder", None), "bos_token_id", None)
                        if fallback is not None:
                            merge_model.config.decoder_start_token_id = fallback
                            print(f"Set decoder_start_token_id = {fallback} (from decoder.bos_token_id)")
                    merge_model = PeftModel.from_pretrained(merge_model, merge_source)
                    print("Merging LoRA weights...")
                    merge_model = merge_model.merge_and_unload()
                    merge_model.save_pretrained(merged_phase1_dir, safe_serialization=True)
                    tokenizer.save_pretrained(merged_phase1_dir)
                    print(f"Merged model saved to {merged_phase1_dir}")
                    del merge_model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print(f"Merged model already exists at {merged_phase1_dir}")
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                if args.upload_checkpoints and args.hf_repo_id and hf_path_prefix:
                    merged_base_repo_path = join_hf_path(hf_path_prefix, "merged_base")
                    print(f"Uploading merged Phase 1 base to {args.hf_repo_id}/{merged_base_repo_path}...")
                    _upload_folder_resilient(
                        repo_id=args.hf_repo_id,
                        folder_path=merged_phase1_dir,
                        path_in_repo=merged_base_repo_path,
                        token=args.hf_token,
                    )

                print("Reloading merged model in 4-bit with fresh LoRA...")
                bnb_config_p2 = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    merged_phase1_dir,
                    local_files_only=True,
                    quantization_config=bnb_config_p2,
                    attn_implementation="sdpa",
                )
                if not hasattr(model.config, 'decoder_start_token_id') or model.config.decoder_start_token_id is None:
                    fallback = getattr(getattr(model.config, "decoder", None), "bos_token_id", None)
                    if fallback is not None:
                        model.config.decoder_start_token_id = fallback
                        print(f"Set decoder_start_token_id = {fallback} (from decoder.bos_token_id)")
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()

                lora_config_p2 = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=lora_r,
                    lora_alpha=lora_r * 2,
                    lora_dropout=args.lora_dropout,
                    target_modules=r"model\.(encoder\.text_model|decoder)\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
                )
                model = get_peft_model(model, lora_config_p2)
                if not args.use_quant:
                    model = model.to(model_dtype)
                if use_fp16:
                    ensure_trainable_params_fp32(model)
                model.print_trainable_parameters()
                if args.compile_model:
                    try:
                        model = torch.compile(model, mode=args.compile_mode)
                        print(f"torch.compile enabled for Phase 2 training ({args.compile_mode})")
                    except Exception as e:
                        print(f"torch.compile for Phase 2 training unavailable, continuing uncompiled: {e}")
                vram_after_relora = get_vram_info()
                print(f"VRAM after fresh LoRA: {vram_after_relora['reserved_gb']:.1f}/{vram_after_relora['total_gb']:.1f} GB")

            phase2_dataset = Text2CypherDataset(
                train_parquet, tokenizer, phase=2,
                max_input_length=max_input_length, max_target_length=max_target_length,
                max_samples=args.train_samples,
                sample_seed=args.seed,
                **dataset_kwargs,
            )
            if len(phase2_dataset) > 0:
                phase2_lr = args.phase2_lr if args.phase2_lr > 0 else args.lr * 0.5
                phase2_epochs = args.phase2_epochs if args.phase2_epochs > 0 else args.epochs
                phase2_resume = None
                if args.resume:
                    p2_latest = find_latest_checkpoint(phase2_output_dir)
                    if p2_latest:
                        phase2_resume = p2_latest
                        print(f"Resuming Phase 2 from checkpoint: {phase2_resume}")
                        from safetensors.torch import load_file
                        adapter_weights = load_file(os.path.join(p2_latest, "adapter_model.safetensors"))
                        model.load_state_dict(adapter_weights, strict=False)
                        print("Loaded Phase 2 adapter weights from checkpoint")
                neo4j_client = None
                roll_stream = None
                if args.phase2_reinforce and args.roll_ws_url:
                    try:
                        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                        from phase2_link import RollStream
                        _proxy = None
                        if args.roll_ws_proxy:
                            _proxy = args.roll_ws_proxy.replace("socks5://", "").split(":")
                            _proxy = (_proxy[0], int(_proxy[1]))
                        roll_stream = RollStream(args.roll_ws_url, auth=args.roll_ws_auth or None,
                                                 proxy=_proxy, flush_every=args.roll_flush_every)
                        if is_main_process():
                            print(f"Rolling stream -> {args.roll_ws_url} every {args.roll_flush_every} rows")
                    except Exception as e:
                        print(f"Rolling stream disabled ({e})")
                        roll_stream = None
                if args.phase2_reinforce:
                    print_reward_function_description()
                    proxies = ({"http": args.neo4j_proxies,
                                "https": args.neo4j_proxies}
                               if args.neo4j_proxies else None)
                    neo4j_client = Neo4jClient(
                        url=args.neo4j_url,
                        port=args.neo4j_port,
                        user=args.neo4j_user,
                        password=args.neo4j_password,
                        database=args.neo4j_database,
                        database_mapping=args.neo4j_database_mapping,
                        proxies=proxies,
                    )
                    if is_main_process():
                        if args.neo4j_url:
                            print(f"Neo4j client created for REINFORCE ({args.neo4j_url})")
                        else:
                            print(f"Neo4j client created for REINFORCE (http://localhost:{args.neo4j_port})")
                kl_ref_model = None
                kl_ref_path = args.kl_ref_model_path or (base_model_path if args.kl_coef > 0 else "")
                if args.phase2_reinforce and args.kl_coef > 0 and kl_ref_path:
                    print(f"Loading frozen KL reference model: {kl_ref_path}")
                    kl_ref_model = AutoModelForSeq2SeqLM.from_pretrained(
                        kl_ref_path,
                        local_files_only=True,
                        attn_implementation="sdpa",
                        torch_dtype=model_dtype,
                    ).to(model.device)
                    kl_ref_model = unify_model_dtype(kl_ref_model, model_dtype, "kl-ref")
                    kl_ref_model.eval()
                    for param in kl_ref_model.parameters():
                        param.requires_grad_(False)
                phase2_best = run_training_phase(
                    model, phase2_dataset, train_val_dataset, collator, tokenizer,
                    phase2_output_dir, phase2_epochs, phase2_lr,
                    batch_size, grad_accum, max_target_length,
                    predict_with_generate, args.seed, vram_limit,
                    resume_checkpoint=phase2_resume, phase_name="Phase 2",
                    label_smoothing=args.label_smoothing,
                    weight_decay=args.weight_decay,
                    neftune_noise_alpha=args.neftune_noise_alpha,
                    eval_steps=args.eval_steps,
                    eval_batch_size=eval_batch_size,
                    use_reinforce=args.phase2_reinforce,
                    neo4j_client=neo4j_client,
                    roll_stream=roll_stream,
                    reinforce_coef=args.reinforce_coef,
                    reinforce_candidates=args.reinforce_candidates,
                    reward_ema_alpha=args.reward_ema_alpha,
                    critic_lr=args.critic_lr,
                    disable_critic=args.disable_critic,
                    entropy_coef=args.entropy_coef,
                    entropy_min=args.entropy_min,
                    exact_match_weight=args.exact_match_weight,
                    train_temperature=args.train_temperature,
                    train_temperature_min=args.train_temperature_min,
                    train_noise_scale=args.train_noise_scale,
                    train_top_k=args.train_top_k,
                    temp_spread=args.temp_spread,
                    temp_spread_min=args.temp_spread_min,
                    kl_coef=args.kl_coef,
                    ppo_clip_eps=args.ppo_clip_eps,
                    kl_ref_model=kl_ref_model,
                    kl_target=args.kl_target,
                    kl_adaptive_alpha=args.kl_adaptive_alpha,
                    kl_beta_min=args.kl_beta_min,
                    kl_beta_max=args.kl_beta_max,
                    kl_prefix_len=args.kl_prefix_len,
                    kl_ref_temperature=args.kl_ref_temperature,
                    sft_coef=args.sft_coef,
                    eos_suppress_strength=args.eos_suppress_strength,
                    eos_suppress_full_until=args.eos_suppress_full_until,
                    eos_suppress_decay_until=args.eos_suppress_decay_until,
                    chunk_size=args.chunk_size,
                    diversity_coef=args.diversity_coef,
                    simplicity_coef=args.simplicity_coef,
                    consistency_coef=args.consistency_coef,
                    dynamic_batch=args.dynamic_batch,
                    max_tokens=args.max_tokens,
                    length_bucket_size=args.length_bucket_size,
                    sample_eval_interval=args.sample_eval_interval,
                    dataloader_num_workers=args.dataloader_num_workers,
                    dataloader_pin_memory=args.dataloader_pin_memory,
                    use_bf16=use_bf16,
                    use_fp16=use_fp16,
                    dynamic_generation_length=args.dynamic_generation_length,
                    generation_length_tolerance=args.generation_length_tolerance,
                    max_steps=args.max_steps,
                    save_steps=args.save_steps,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=args.hf_token,
                    hf_path_prefix=hf_path_prefix,
                    eval_after_resume=args.eval_after_resume,
                )
                if phase2_best is not None:
                    best_dir = phase2_best
                else:
                    print("Phase 2 failed or OOMed; keeping previous Phase 2 checkpoint for retry")
                    with open(os.path.join(args.output_dir, "PHASE2_FAILED"), "w") as f:
                        f.write("Phase 2 failed or OOMed. Retry with a lower batch size and --resume to continue from the latest Phase 2 checkpoint.\n")
                    raise RuntimeError("Phase 2 failed or OOMed; retry with a lower batch size and --resume")
                del phase2_dataset
                gc.collect()
            else:
                print("Phase 2 dataset empty, skipping phase 2")

        if best_dir is None or not os.path.isdir(best_dir):
            if resume_checkpoint and os.path.isdir(resume_checkpoint):
                best_dir = resume_checkpoint
                print(f"No best/ dir, falling back to resume checkpoint: {best_dir}")
            else:
                latest = find_latest_checkpoint(args.output_dir)
                if latest:
                    best_dir = latest
                    print(f"No best/ dir, falling back to latest checkpoint: {best_dir}")
                else:
                    best_dir = os.path.join(args.output_dir, "best")
    else:
        if args.eval_model_dir:
            best_dir = resolve_eval_model_dir(args.eval_model_dir, phase2_output_dir if args.phase2 else args.output_dir)
            print(f"Eval-only mode. Using explicit model/checkpoint source: {best_dir}")
        elif args.phase2:
            best_dir = os.path.join(phase2_output_dir, "best")
        else:
            best_dir = os.path.join(args.output_dir, "best")
        if not os.path.isdir(best_dir):
            latest = find_latest_checkpoint(phase2_output_dir if args.phase2 else args.output_dir)
            if latest:
                best_dir = latest
            elif resume_checkpoint and os.path.isdir(resume_checkpoint):
                best_dir = resume_checkpoint
            else:
                latest = find_latest_checkpoint(args.output_dir)
                if latest:
                    best_dir = latest
        if not os.path.isdir(best_dir):
            raise FileNotFoundError(f"Eval-only source not found: {best_dir}")
        print(f"Eval-only mode. Using model/checkpoint from {best_dir}")

    if not is_main_process():
        print("Non-main distributed rank finished training; skipping merge/eval/upload.")
        return

    eval_artifact_dir = os.path.join(args.output_dir, "eval_only_results") if args.eval_only else best_dir
    os.makedirs(eval_artifact_dir, exist_ok=True)

    print("\nRunning evaluation with generation...")
    torch.cuda.empty_cache()
    gc.collect()

    # Cleanup: retain only best 2 + recent 3 checkpoints
    print("Cleaning up old checkpoints (keeping best 2 + recent 3)...")
    retain_best_recent_checkpoints(args.output_dir, keep_best=2, keep_recent=3)
    if args.phase2:
        retain_best_recent_checkpoints(phase2_output_dir, keep_best=2, keep_recent=3)
    
    # Upload checkpoints to HF Hub if requested
    if args.upload_checkpoints and args.hf_repo_id:
        print(f"Syncing local checkpoints to {args.hf_repo_id}...")
        sync_missing_local_to_hf(args.output_dir, args.hf_repo_id, hf_token=args.hf_token, include_best=True, hf_path_prefix=hf_path_prefix)
        if args.phase2 and phase2_output_dir != args.output_dir:
            sync_missing_local_to_hf(phase2_output_dir, args.hf_repo_id, hf_token=args.hf_token, include_best=True, hf_path_prefix=hf_path_prefix)

    eval_source_is_full_model = is_full_model_dir(best_dir)
    if eval_source_is_full_model:
        merged_dir = best_dir
    elif args.eval_only:
        safe_source_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(best_dir).name) or "eval_source"
        merged_dir = os.path.join(eval_artifact_dir, f"merged_{safe_source_name}")
    else:
        merged_dir = os.path.join(args.output_dir, "merged_eval")

    source_marker_path = os.path.join(merged_dir, ".eval_source_path")
    if args.eval_only and not eval_source_is_full_model and os.path.isdir(merged_dir):
        try:
            previous_source = Path(source_marker_path).read_text(encoding="utf-8").strip()
        except Exception:
            previous_source = ""
        if previous_source != os.path.abspath(best_dir):
            print(f"Eval-only merged dir {merged_dir} was built from a different source; rebuilding.")
            shutil.rmtree(merged_dir)

    if eval_source_is_full_model:
        print(f"Eval source is a full merged model; loading directly from {merged_dir}")
        del model
    elif not os.path.exists(os.path.join(merged_dir, "model.safetensors")):
        print("Loading base model in configured precision for LoRA merge...")
        if args.eval_only:
            print(f"Eval-only adapter/checkpoint source: {best_dir}")
            print(f"Merging eval-only source into temporary directory: {merged_dir}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        merge_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path, torch_dtype=model_dtype, attn_implementation="sdpa",
        )
        merge_model = unify_model_dtype(merge_model, model_dtype, "merge-eval")
        if not hasattr(merge_model.config, 'decoder_start_token_id') or merge_model.config.decoder_start_token_id is None:
            fallback = getattr(getattr(merge_model.config, "decoder", None), "bos_token_id", None)
            if fallback is not None:
                merge_model.config.decoder_start_token_id = fallback
                print(f"Set decoder_start_token_id = {fallback} (from decoder.bos_token_id)")
        merge_model = PeftModel.from_pretrained(merge_model, best_dir)
        print("Merging LoRA weights into full model...")
        merge_model = merge_model.merge_and_unload()
        merge_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        if args.eval_only:
            Path(source_marker_path).write_text(os.path.abspath(best_dir), encoding="utf-8")
        print(f"Merged model saved to {merged_dir}")
        del merge_model
    else:
        print(f"Merged model already exists at {merged_dir}")
        del model
    torch.cuda.empty_cache()
    gc.collect()

    if args.upload_checkpoints and args.hf_repo_id and hf_path_prefix:
        print(f"Uploading merged model to {args.hf_repo_id}/{hf_path_prefix} for future Phase 2 base loading...")
        _upload_folder_resilient(
            repo_id=args.hf_repo_id,
            folder_path=merged_dir,
            path_in_repo=hf_path_prefix,
            token=args.hf_token,
        )
        upload_rolling_data_to_hf(phase2_output_dir if args.phase2 else args.output_dir, args.hf_repo_id, hf_token=args.hf_token, hf_path_prefix=hf_path_prefix)

    print("Loading merged model in native precision for evaluation...")
    eval_model = AutoModelForSeq2SeqLM.from_pretrained(
        merged_dir, attn_implementation="sdpa",
    )
    if not hasattr(eval_model.config, 'decoder_start_token_id') or eval_model.config.decoder_start_token_id is None:
        fallback = getattr(getattr(eval_model.config, "decoder", None), "bos_token_id", None)
        if fallback is not None:
            eval_model.config.decoder_start_token_id = fallback
            print(f"Set decoder_start_token_id = {fallback} (from decoder.bos_token_id)")
    eval_model_dtype = get_model_float_dtype(eval_model)
    _eval_dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    eval_model.to(_eval_dev)
    eval_model.eval()
    eval_model.config.use_cache = True
    if is_main_process():
        print(f"Eval model loaded in dtype: {eval_model_dtype}")
    if args.compile_eval:
        try:
            eval_model = torch.compile(eval_model, mode="default")
            print("torch.compile enabled for eval (default)")
        except Exception as e:
            print(f"torch.compile not available, skipping: {e}")

    if args.eval_samples > 0 and args.eval_samples < len(val_dataset):
        eval_val_dataset = [val_dataset[i] for i in range(args.eval_samples)]
        print(f"Val eval samples limited to {len(eval_val_dataset)}")
    else:
        eval_val_dataset = val_dataset

    eval_neo4j = None
    if args.phase2_reinforce:
        eval_proxies = ({"http": args.neo4j_proxies, "https": args.neo4j_proxies}
                        if args.neo4j_proxies else None)
        eval_neo4j = Neo4jClient(
            url=args.neo4j_url,
            port=args.neo4j_port, user=args.neo4j_user, password=args.neo4j_password,
            database=args.neo4j_database,
            database_mapping=args.neo4j_database_mapping,
            proxies=eval_proxies,
        )
        if args.neo4j_url:
            print(f"Neo4j client created for eval execution metrics ({args.neo4j_url})")
        else:
            print(f"Neo4j client created for eval execution metrics (http://localhost:{args.neo4j_port})")

    if args.final_test_only:
        print("Final-test-only mode: skipping validation generation eval.")
    else:
        val_metrics_path = os.path.join(eval_artifact_dir, "eval_metrics.json")
        val_metrics = run_generation_eval(
            eval_model, eval_val_dataset, tokenizer, collator,
            max_target_length, val_metrics_path, label="val", eval_batch_size=eval_batch_size,
            neo4j_client=eval_neo4j, phase_name="Phase 2" if args.phase2 else "Phase 1",
            rolling_jsonl_path=rolling_jsonl_path, rolling_json_max_records=args.rolling_json_max_records,
        )
        if val_metrics is not None:
            final_val_metrics_path = os.path.join(args.output_dir, "final_eval_metrics.json")
            with open(final_val_metrics_path, "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2, default=str)
            print(f"Final eval metrics also saved to {final_val_metrics_path}")

    if os.path.exists(test_parquet):
        print(f"\nLoading test dataset from {test_parquet}...")
        test_dataset = Text2CypherDataset(
            test_parquet, tokenizer,
            max_input_length=max_input_length, max_target_length=max_target_length,
            max_samples=args.test_samples,
            sample_seed=args.seed,
            **dataset_kwargs,
        )
        if args.test_samples > 0:
            print(f"Test subset enabled: {len(test_dataset)} samples")
        print(f"Test dataset: {len(test_dataset)} samples")
        test_metrics_path = os.path.join(eval_artifact_dir, "test_metrics.json")
        test_metrics = run_generation_eval(
            eval_model, test_dataset, tokenizer, collator,
            max_target_length, test_metrics_path, label="test", eval_batch_size=test_batch_size,
            neo4j_client=eval_neo4j, phase_name="Phase 2" if args.phase2 else "Phase 1",
            rolling_jsonl_path=rolling_jsonl_path, rolling_json_max_records=args.rolling_json_max_records,
        )
        if test_metrics is not None:
            final_test_metrics_path = os.path.join(args.output_dir, "final_test_metrics.json")
            with open(final_test_metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2, default=str)
            print(f"Final test metrics also saved to {final_test_metrics_path}")
    else:
        print(f"Test parquet not found: {test_parquet}, skipping test eval")

    del eval_model
    torch.cuda.empty_cache()
    gc.collect()

    if args.upload_checkpoints and args.hf_repo_id:
        upload_rolling_data_to_hf(phase2_output_dir if args.phase2 else args.output_dir, args.hf_repo_id, hf_token=args.hf_token, hf_path_prefix=hf_path_prefix)


if __name__ == "__main__":
    main()