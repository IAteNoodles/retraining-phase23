#!/usr/bin/env python3
"""CLI: `python cli.py train ...` or `python cli.py eval ...`.

Works identically on a local machine and on Kaggle (paths differ, flags don't).
Example local resume:
  python cli.py train --phase 2 --train_parquet .../phase2_train.parquet ... \
      --resume_checkpoint ./checkpoints/p2/checkpoint-500
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrain.config import build_parser, config_from_args


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "merge-latest":
        from retrain.merge import merge_latest
        merge_latest(args.base_model, args.adapter_dir, args.out_dir, dtype=args.dtype)
        return
    cfg = config_from_args(args)
    # hf:// resume pointers resolve before anything else touches checkpoints
    if getattr(args, "resume_checkpoint", ""):
        from retrain.sync import resolve_checkpoint
        cfg.resume_checkpoint = resolve_checkpoint(
            args.resume_checkpoint, token=cfg.hf_token)
    if args.command == "train":
        from retrain.flow import run_phase
        best = run_phase(cfg)
        # optional post-training eval on the best adapter
        from retrain.flow import run_eval
        run_eval(cfg, best)
        if cfg.upload_checkpoints and cfg.hf_repo_id:
            from retrain.sync import upload_dir
            prefix = (cfg.hf_path_prefix.strip("/") + "/") if cfg.hf_path_prefix else ""
            upload_dir(cfg.output_dir, cfg.hf_repo_id,
                       f"{prefix}phase_{cfg.phase}", token=cfg.hf_token)
    elif args.command == "eval":
        from retrain.flow import run_eval
        run_eval(cfg, args.eval_model_dir)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
