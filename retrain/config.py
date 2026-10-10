"""Run configuration: one dataclass, built from argparse. No globals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    # data (separate files per phase - what you point at is what trains)
    phase: int = 2
    train_parquet: str = ""
    val_parquet: str = ""
    test_parquet: str = ""
    schema_col: str = "schema_full"
    question_col: str = "question"
    target_col: str = "target"
    schema_variant: str = ""          # e.g. "small,mid,full" subset of augmented files; "" = all
    max_train_samples: int = 0        # 0 = all
    max_val_samples: int = 0
    max_test_samples: int = 0
    seed: int = 42

    # model
    model_path: str = "t5gemma-270m"
    output_dir: str = "./checkpoints/p2"
    lora_r: int = 64
    lora_dropout: float = 0.1
    use_quant: bool = True
    bf16: bool = True            # Ada (4050) runs bf16 as fast or faster than fp16, more stable
    fp16: bool = True
    compile_model: bool = False  # torch.compile: ~20-30% faster steps, costs warmup + RAM
    dataloader_workers: int = 2  # keep the GPU fed (0 = main-process loading, stalls)

    # optimization
    epochs: int = 3
    max_steps: int = 0
    lr: float = 1e-4
    batch_size: int = 2          # tuned for ~5 GB usable VRAM (RTX 4050 6GB)
    eval_batch_size: int = 0         # 0 = same as train
    grad_accum: int = 16         # effective batch 32 at batch_size 2
    # token-budget batching: pack micro-batches to max_tokens AND
    # max_target_tokens (targets OOM first: ~4 MB/token on 262K vocab),
    # accumulate to ~target_step_tokens per optimizer step
    dynamic_batch: bool = True
    max_tokens: int = 6000       # micro-batch total-token budget (SFT steps)
    max_target_tokens: int = 500  # micro-batch target-token budget (BINDING)
    rl_micro_tokens: int = 3000  # tighter total budget when use_rl (CE graph +
    rl_micro_target_tokens: int = 350  # target budget when use_rl; RL section
                                       # coexists with the CE graph, so RL
                                       # steps need smaller micros
    target_step_tokens: int = 55000
    max_input_length: int = 1536  # advisory only; inputs are never truncated
    max_target_length: int = 1024  # generation/eval cap == target p100 (1019); no waste
    rl_max_new_tokens: int = 320   # RL candidates stay short (VRAM); eval uses max_target_length
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    neftune_noise_alpha: float = 5.0
    save_steps: int = 200
    eval_steps: int = 200

    # eval / metrics
    eval_with_generate: bool = True
    eval_samples: int = 500
    exec_eval_samples: int = 100  # fixed val subset for execution-based best tracking (needs Neo4j)
    exec_jaccard_threshold: float = 0.85
    best_metric: str = "eval_execution_accuracy"  # or eval_loss / eval_exact_match

    # neo4j (optional; required for RL + execution metrics)
    neo4j_url: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_fallback_db: str = "movies"
    db_mapping_json: str = ""

    # RL (GRPO)
    use_rl: bool = False
    rl_coef: float = 0.3
    rl_candidates: int = 4          # GRPO group size G (>= 2)
    rl_chunk_size: int = 2          # candidates per forward slice (VRAM bound)
    rl_every_n_steps: int = 2
    rl_warmup_steps: int = 50
    reward_ema_alpha: float = 0.1   # logging EMA only; baseline is group mean
    entropy_coef: float = 0.0
    entropy_min: float = 0.0
    train_temperature: float = 0.7
    train_temperature_min: float = 0.7
    temp_spread: float = 0.0        # hotter later candidates: T*(1 + c*spread)
    temp_spread_min: float = 0.0    # annealed floor when max_steps set
    train_top_k: int = 50
    train_noise_scale: float = 0.0  # Gaussian logit noise for exploration
    kl_coef: float = 0.0
    kl_ref_model_path: str = ""     # frozen reference; defaults to model_path
    exact_match_weight: float = 0.3
    sft_coef: float = 1.0
    diversity_coef: float = 0.1     # 1 - mean pairwise bigram overlap
    simplicity_coef: float = 0.15   # -(length error + 3x repetition + brackets)
    consistency_coef: float = 0.1   # 1 if result == group majority else 0

    # early stopping + diagnostics
    min_epochs: float = 0.0         # floor before early stopping may fire (0 = off)
    rolling_jsonl_path: str = ""    # per-sample pred/gold/reward trace; "" = off
    rolling_json_max_records: int = 1000

    # hub sync
    hf_repo_id: str = ""
    hf_token: str = ""
    hf_path_prefix: str = ""
    upload_checkpoints: bool = False

    # misc
    resume_checkpoint: str = ""
    eval_only: bool = False
    eval_model_dir: str = ""
    final_test_only: bool = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NL-to-Cypher retraining harness (phases 2/3)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--phase", type=int, default=2)
        sp.add_argument("--train_parquet", type=str, default="")
        sp.add_argument("--val_parquet", type=str, default="")
        sp.add_argument("--test_parquet", type=str, default="")
        sp.add_argument("--schema_col", type=str, default="schema_full")
        sp.add_argument("--question_col", type=str, default="question")
        sp.add_argument("--target_col", type=str, default="target")
        sp.add_argument("--schema_variant", type=str, default="")
        sp.add_argument("--max_train_samples", type=int, default=0)
        sp.add_argument("--max_val_samples", type=int, default=0)
        sp.add_argument("--max_test_samples", type=int, default=0)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--model_path", type=str, default="t5gemma-270m")
        sp.add_argument("--output_dir", type=str, default="./checkpoints/p2")
        sp.add_argument("--lora_r", type=int, default=64)
        sp.add_argument("--lora_dropout", type=float, default=0.1)
        sp.add_argument("--no_quant", action="store_true")
        sp.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--compile_model", action="store_true")
        sp.add_argument("--dataloader_workers", type=int, default=2)
        sp.add_argument("--epochs", type=int, default=3)
        sp.add_argument("--max_steps", type=int, default=0)
        sp.add_argument("--lr", type=float, default=1e-4)
        sp.add_argument("--batch_size", type=int, default=2)
        sp.add_argument("--eval_batch_size", type=int, default=0)
        sp.add_argument("--grad_accum", type=int, default=16)
        sp.add_argument("--dynamic_batch", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--max_tokens", type=int, default=6000)
        sp.add_argument("--max_target_tokens", type=int, default=500)
        sp.add_argument("--rl_micro_tokens", type=int, default=3000)
        sp.add_argument("--rl_micro_target_tokens", type=int, default=350)
        sp.add_argument("--target_step_tokens", type=int, default=55000)
        sp.add_argument("--max_input_length", type=int, default=1536)
        sp.add_argument("--max_target_length", type=int, default=1024,
                        help="Generation/eval output cap == target p100 (1019). "
                             "Training inputs and targets are NEVER truncated.")
        sp.add_argument("--weight_decay", type=float, default=0.01)
        sp.add_argument("--label_smoothing", type=float, default=0.0)
        sp.add_argument("--neftune_noise_alpha", type=float, default=5.0)
        sp.add_argument("--save_steps", type=int, default=200)
        sp.add_argument("--eval_steps", type=int, default=200)
        sp.add_argument("--eval_with_generate", action=argparse.BooleanOptionalAction, default=True)
        sp.add_argument("--eval_samples", type=int, default=500)
        sp.add_argument("--exec_eval_samples", type=int, default=100)
        sp.add_argument("--exec_jaccard_threshold", type=float, default=0.85)
        sp.add_argument("--best_metric", type=str, default="eval_execution_accuracy")
        sp.add_argument("--neo4j_url", type=str, default="")
        sp.add_argument("--neo4j_user", type=str, default="neo4j")
        sp.add_argument("--neo4j_password", type=str, default="")
        sp.add_argument("--neo4j_fallback_db", type=str, default="movies")
        sp.add_argument("--db_mapping_json", type=str, default="")

    t = sub.add_parser("train", help="validate data, then run SFT (+ optional RL)")
    add_common(t)
    t.add_argument("--use_rl", action="store_true")
    t.add_argument("--rl_coef", type=float, default=0.3)
    t.add_argument("--rl_candidates", type=int, default=4)
    t.add_argument("--rl_chunk_size", type=int, default=2)
    t.add_argument("--rl_max_new_tokens", type=int, default=0)
    t.add_argument("--rl_every_n_steps", type=int, default=2)
    t.add_argument("--rl_warmup_steps", type=int, default=50)
    t.add_argument("--reward_ema_alpha", type=float, default=0.1)
    t.add_argument("--entropy_coef", type=float, default=0.0)
    t.add_argument("--entropy_beta", type=float, default=None)   # v1 alias
    t.add_argument("--entropy_min", type=float, default=0.0)
    t.add_argument("--train_temperature", type=float, default=0.7)
    t.add_argument("--train_temperature_min", type=float, default=0.7)
    t.add_argument("--temp_spread", type=float, default=0.0)
    t.add_argument("--temp_spread_min", type=float, default=0.0)
    t.add_argument("--train_top_k", type=int, default=50)
    t.add_argument("--top_k", type=int, default=None)           # v1 alias
    t.add_argument("--train_noise_scale", type=float, default=0.0)
    t.add_argument("--kl_coef", type=float, default=0.0)
    t.add_argument("--kl_beta", type=float, default=None)       # v1 alias
    t.add_argument("--kl_ref_model_path", type=str, default="")
    t.add_argument("--exact_match_weight", type=float, default=0.3)
    t.add_argument("--em_weight", type=float, default=None)     # v1 alias
    t.add_argument("--sft_coef", type=float, default=1.0)
    t.add_argument("--diversity_coef", type=float, default=0.1)
    t.add_argument("--simplicity_coef", type=float, default=0.05)
    t.add_argument("--consistency_coef", type=float, default=0.1)
    t.add_argument("--min_epochs", type=float, default=0.0)
    t.add_argument("--rolling_jsonl_path", type=str, default="")
    t.add_argument("--rolling_json_max_records", type=int, default=1000)
    t.add_argument("--hf_repo_id", type=str, default="")
    t.add_argument("--hf_token", type=str, default="")
    t.add_argument("--hf_path_prefix", type=str, default="")
    t.add_argument("--upload_checkpoints", action="store_true")
    t.add_argument("--resume_checkpoint", type=str, default="")

    e = sub.add_parser("eval", help="generation eval on val/test from a checkpoint dir")
    add_common(e)
    e.add_argument("--eval_model_dir", type=str, default="", required=True)
    e.add_argument("--final_test_only", action="store_true")

    m = sub.add_parser("merge-latest", help="merge a LoRA adapter into a base model")
    m.add_argument("--base_model", type=str, required=True)
    m.add_argument("--adapter_dir", type=str, required=True)
    m.add_argument("--out_dir", type=str, required=True)
    m.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    d = {k: v for k, v in vars(args).items() if k in RunConfig.__dataclass_fields__}
    d["use_quant"] = not getattr(args, "no_quant", False)
    d["fp16"] = bool(getattr(args, "fp16", True))
    cfg = RunConfig(**d)
    # precision precedence: explicit bf16 wins over fp16 (mutually exclusive in HF)
    if cfg.bf16:
        cfg.fp16 = False
    # v1 aliases (old train.py accepted both names)
    if getattr(args, "entropy_beta", None) is not None:
        cfg.entropy_coef = args.entropy_beta
    if getattr(args, "em_weight", None) is not None:
        cfg.exact_match_weight = args.em_weight
    if getattr(args, "top_k", None) is not None:
        cfg.train_top_k = args.top_k
    if getattr(args, "kl_beta", None) is not None:
        cfg.kl_coef = args.kl_beta
    return cfg
