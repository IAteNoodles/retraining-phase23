"""SFT stage. Fixes vs the old train.py:

- `greater_is_better` is derived from the actual metric (loss -> False).
- Best checkpoint is tracked explicitly (PEFT-safe adapter copy) instead of
  mislabeling the final model as "best".
- No silent behavior changes; no dead knobs.
"""

from __future__ import annotations

import gc
import inspect
import os
import shutil

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from .dataset import Text2CypherCollator

class MinEpochEarlyStopping(TrainerCallback):
    """Stop when `metric` hasn't improved for `patience` evals, but never
    before `min_epochs` epochs have run."""

    def __init__(self, metric: str, greater_is_better: bool,
                 patience: int = 5, min_epochs: float = 0.0):
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.patience = max(1, int(patience))
        self.min_epochs = max(0.0, float(min_epochs))
        self.best = None
        self.bad = 0

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.greater_is_better else value < self.best

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if self.metric not in metrics:
            if self.metric == "eval_loss":
                return
            self.metric, self.greater_is_better = "eval_loss", False
            self.best, self.bad = None, 0
        try:
            value = float(metrics[self.metric])
        except (TypeError, ValueError):
            return
        if self._better(value):
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        if self.bad >= self.patience and state.epoch >= self.min_epochs:
            print(f"  [early-stop] {self.metric} stagnant for {self.bad} evals "
                  f"(best={self.best:.4f}, epoch={state.epoch:.2f})", flush=True)
            control.should_training_stop = True


class BestAdapterTracker(TrainerCallback):
    """Copies adapter weights to best_dir whenever `metric` improves.

    Falls back to eval_loss (the only metric HF always computes in-loop);
    execution-based selection belongs to post-training run_eval.
    """

    def __init__(self, metric: str, greater_is_better: bool, best_dir: str):
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_dir = best_dir
        self.best_value = None
        self.trainer_ref = None  # set by build_trainer (HF does not pass it)
        self._fell_back = False

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.greater_is_better else value < self.best_value

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if self.metric not in metrics and not self._fell_back:
            self.metric, self.greater_is_better = "eval_loss", False
            self._fell_back = True
            print(f"  [best] execution metric unavailable in-loop; tracking eval_loss "
                  f"(use run_eval for execution-based selection)", flush=True)
        if self.metric not in metrics:
            return
        try:
            value = float(metrics[self.metric])
        except (TypeError, ValueError):
            return
        if self._is_better(value):
            self.best_value = value
            if self.trainer_ref is not None:
                os.makedirs(self.best_dir, exist_ok=True)
                self.trainer_ref.save_model(self.best_dir)
                print(f"  [best] {self.metric}={value:.4f} -> saved to {self.best_dir}", flush=True)


def _metric_direction(best_metric: str) -> tuple[str, bool]:
    loss_like = ("loss", "error_rate", "edit_distance")
    if any(k in best_metric for k in loss_like):
        return best_metric, False
    return best_metric, True


def build_trainer(model, tokenizer, train_dataset, val_dataset, collator,
                  out_dir: str, cfg, use_rl: bool = False, **rl_kwargs):
    from .batching import BucketDataloaderMixin
    from .rl import GRPOTrainer

    metric, greater = _metric_direction(cfg.best_metric)
    best_dir = os.path.join(out_dir, "best")
    tracker = BestAdapterTracker(metric, greater, best_dir)

    # token-budget batching: micro-batches packed to max_tokens AND
    # max_target_tokens; accumulation measured so each optimizer step sees
    # ~target_step_tokens (not guessed from the budget ceiling)
    bucket_cfg = None
    effective_grad_accum = cfg.grad_accum
    if cfg.dynamic_batch:
        from .batching import TokenBucketBatchSampler
        bucket_cfg = dict(max_batch_size=max(8, cfg.batch_size * 4),
                          max_tokens=cfg.max_tokens,
                          max_target_tokens=cfg.max_target_tokens,
                          num_workers=cfg.dataloader_workers,
                          seed=cfg.seed)
        probe = TokenBucketBatchSampler(
            train_dataset, max_batch_size=bucket_cfg["max_batch_size"],
            max_tokens=cfg.max_tokens, max_target_tokens=cfg.max_target_tokens,
            shuffle=False, seed=cfg.seed)
        probe_batches = probe._build_batches()
        mean_fill = (sum(sum(probe.lengths[i] for i in b) for b in probe_batches)
                     / max(len(probe_batches), 1))
        effective_grad_accum = max(1, int(round(cfg.target_step_tokens / max(mean_fill, 1))))
        print(f"  [batching] micro-budget={cfg.max_tokens}/{cfg.max_target_tokens}tgt "
              f"(mean fill {mean_fill:.0f}) -> grad_accum={effective_grad_accum} "
              f"(~{effective_grad_accum * mean_fill:.0f} tok/step)", flush=True)

    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size or cfg.batch_size,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        label_smoothing_factor=cfg.label_smoothing,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        load_best_model_at_end=False,   # PEFT-incompatible; BestAdapterTracker handles it
        # NOTE: HF requires this metric to exist in-loop; only eval_loss always
        # does (we compute no in-loop metrics). BestAdapterTracker falls back
        # to it independently; execution-based selection is a run_eval decision.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=cfg.eval_with_generate,
        generation_max_length=cfg.max_target_length,
        report_to="none",
        seed=cfg.seed,
        optim="adafactor",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )
    if cfg.max_steps > 0:
        kwargs["max_steps"] = int(cfg.max_steps)
    if "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = "steps" if val_dataset is not None else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if val_dataset is not None else "no"
    try:
        targs = Seq2SeqTrainingArguments(**kwargs, group_by_length=True)
    except TypeError:
        targs = Seq2SeqTrainingArguments(**kwargs)
    if "length_column_name" in ta_params:
        targs.length_column_name = "length"

    base = dict(model=model, args=targs, train_dataset=train_dataset,
                eval_dataset=val_dataset, data_collator=collator,
                callbacks=[tracker])
    if cfg.min_epochs > 0 and val_dataset is not None:
        base["callbacks"].append(MinEpochEarlyStopping(metric, greater, min_epochs=cfg.min_epochs))
    if use_rl:
        trainer = GRPOTrainer(tokenizer=tokenizer, cfg=cfg, **rl_kwargs, **base)
    else:
        from transformers import Seq2SeqTrainer as _T

        from .oom_safe import OOMSafeMixin

        class _SFT(BucketDataloaderMixin, OOMSafeMixin, _T):
            def _prepare_inputs(self, inputs):
                if isinstance(inputs, dict):
                    for k in ("gold_answer", "database", "input_text", "index"):
                        inputs.pop(k, None)
                return super()._prepare_inputs(inputs)

        trainer = _SFT(**base)
    tracker.trainer_ref = trainer
    trainer.bucket_cfg = bucket_cfg
    return trainer, tracker


def run_sft(model, tokenizer, train_dataset, val_dataset, cfg,
            resume_checkpoint: str = "", use_rl: bool = False,
            neo4j_client=None, kl_ref_model=None) -> str:
    collator = Text2CypherCollator(tokenizer)
    rl_kwargs = {}
    if use_rl:
        rl_kwargs = dict(
            neo4j_client=neo4j_client,
            kl_ref_model=kl_ref_model,
        )
    trainer, tracker = build_trainer(
        model, tokenizer, train_dataset, val_dataset, collator,
        cfg.output_dir, cfg, use_rl=use_rl, **rl_kwargs)
    trainer.train(resume_from_checkpoint=resume_checkpoint or None)
    gc.collect()
    if tracker.best_value is None:
        # no eval ever improved (or never ran): fall back to final state,
        # but say so explicitly instead of calling it "best"
        final_dir = os.path.join(cfg.output_dir, "final")
        trainer.save_model(final_dir)
        print(f"No improving eval seen; final model saved to {final_dir}")
        return final_dir
    return os.path.join(cfg.output_dir, "best")
