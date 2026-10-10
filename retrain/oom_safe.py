"""OOM-redo: never let a fat micro-batch kill a run.

Wraps `training_step`: on CUDA OOM it
  1. clears cache,
  2. shrinks future micro-batch budgets (x0.7, floored) so it happens less,
  3. splits the failed micro-batch in half and retries each half
     ("takes items out, puts them back in the bag"),
  4. if even ONE sample OOMs alone, logs it to `skipped_oom.jsonl` and moves
     on instead of crashing the run.

Splitting is exact for GRPO (advantages are per-prompt groups, untouched by
splitting across prompts) and near-exact for SFT (token-mean over halves).
"""

from __future__ import annotations

import gc
import json
import os

import torch


def _split_inputs(inputs: dict):
    """Split a collated batch along dim 0 into two halves (tensors narrowed,
    python lists sliced)."""
    n = None
    for v in inputs.values():
        if torch.is_tensor(v):
            n = v.shape[0]
            break
    if n is None:
        return None
    halves = []
    for lo, hi in ((0, n // 2), (n // 2, n)):
        if hi <= lo:
            continue
        part = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                part[k] = v[lo:hi]
            elif isinstance(v, list):
                part[k] = v[lo:hi]
            else:
                part[k] = v
        halves.append(part)
    return halves if len(halves) == 2 else None


class OOMSafeMixin:
    """Add before the Trainer base class. Requires nothing else."""

    oom_shrink = 0.7
    oom_min_tokens = 1000
    oom_min_target_tokens = 200

    def _record_skipped(self, inputs, step=None, err=""):
        try:
            def _len(x):
                try:
                    return int(x.shape[-1]) if torch.is_tensor(x) else len(x)
                except Exception:
                    return -1
            rec = {
                "step": step if step is not None else getattr(
                    getattr(self, "state", None), "global_step", "?"),
                "index": str(inputs.get("index", "?")),
                "database": str(inputs.get("database", "?")),
                "input_tokens": _len(inputs.get("input_ids")),
                "target_tokens": _len(inputs.get("labels")),
                "error": str(err)[:300],
            }
            path = os.path.join(self.args.output_dir, "skipped_oom.jsonl")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _shrink_budgets(self):
        cfg = getattr(self, "bucket_cfg", None)
        if isinstance(cfg, dict):
            old_tok, old_tgt = cfg.get("max_tokens"), cfg.get("max_target_tokens")
            cfg["max_tokens"] = max(self.oom_min_tokens,
                                    int(cfg.get("max_tokens", 6000) * self.oom_shrink))
            cfg["max_target_tokens"] = max(
                self.oom_min_target_tokens,
                int(cfg.get("max_target_tokens", 500) * self.oom_shrink))
            print(f"  [oom-redo] budgets {old_tok}/{old_tgt}tgt -> "
                  f"{cfg['max_tokens']}/{cfg['max_target_tokens']}tgt "
                  f"(grad_accum unchanged: steps now see fewer tokens)",
                  flush=True)
            try:  # keep run_config honest about the moving budgets
                import json as _json3
                _rc = os.path.join(self.args.output_dir, "run_config.json")
                if os.path.exists(_rc):
                    with open(_rc, encoding="utf-8") as _f:
                        _d = _json3.load(_f)
                    _d["micro_tokens_current"] = cfg["max_tokens"]
                    _d["micro_target_current"] = cfg["max_target_tokens"]
                    with open(_rc, "w", encoding="utf-8") as _f:
                        _json3.dump(_d, _f, indent=1, default=str)
            except Exception:
                pass

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            err = str(e)[:300]  # `e` is deleted after the block; keep the text
        # --- OOM path ---
        step = getattr(getattr(self, "state", None), "global_step", "?")
        torch.cuda.empty_cache()
        gc.collect()
        self._shrink_budgets()
        halves = _split_inputs(inputs)
        if not halves:
            self._record_skipped(inputs, step, err)
            print(f"  [oom-redo] step={step}: single sample still OOMs - logged + skipped",
                  flush=True)
            dev = model.device if hasattr(model, "device") else "cuda"
            return torch.tensor(0.0, device=dev)
        try:
            n0 = inputs["input_ids"].shape[0] if torch.is_tensor(inputs.get("input_ids")) else "?"
        except Exception:
            n0 = "?"
        print(f"  [oom-redo] step={step}: micro-batch OOM (n={n0}), "
              f"retrying as 2 halves", flush=True)
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        total = 0.0
        for half in halves:
            total = total + self.training_step(model, half, num_items_in_batch)
        return total
