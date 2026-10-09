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
    cfg = config_from_args(args)
    if args.command == "train":
        from retrain.flow import run_phase
        best = run_phase(cfg)
        # optional post-training eval on the best adapter
        from retrain.flow import run_eval
        run_eval(cfg, best)
    elif args.command == "eval":
        from retrain.flow import run_eval
        run_eval(cfg, args.eval_model_dir)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
