"""Pre-training gates for phase-1 SFT. Fails fast with a clear message.

Checks: split files/columns/row-counts, no empty input/target, ZERO
target-in-input leaks, all targets valid Cypher-leading clauses, base model
files + loadable tokenizer, torch/CUDA present, train.py bundle importable.

Usage:
  python tools/assert_phase1.py --data_dir <dir with phase1_*.parquet>
      --base_dir <merged_base dir> [--min_train_rows 100000]
"""
import argparse
import os
import sys

LEADERS = ("MATCH", "WITH", "CREATE", "UNWIND", "OPTIONAL", "CALL", "MERGE",
           "FOREACH", "UNION", "LOAD", "RETURN")


def fail(msg):
    print(f"[ASSERT-FAIL] {msg}", flush=True)
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--min_train_rows", type=int, default=100000)
    ap.add_argument("--leak_scan", type=int, default=0,
                    help="rows to leak-scan per split (0 = all)")
    ap.add_argument("--hf_token", type=str, default="",
                    help="HF token for private repo access")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    splits = {}
    for split in ["train", "val", "test"]:
        p = os.path.join(args.data_dir, f"phase1_{split}.parquet")
        if not os.path.exists(p):
            fail(f"missing {p}")
        t = pq.read_table(p).to_pylist()
        if not t:
            fail(f"{p}: zero rows")
        splits[split] = t
        print(f"[ok] {split}: n={len(t)}", flush=True)
    if len(splits["train"]) < args.min_train_rows:
        fail(f"train rows {len(splits['train'])} < {args.min_train_rows}")

    need = {"input", "target", "database", "label"}
    for split, t in splits.items():
        cols = set(t[0].keys())
        if not need.issubset(cols):
            fail(f"{split}: missing columns {need - cols}")
        empty = sum(1 for r in t if not str(r.get("input", "")).strip()
                    or not str(r.get("target", "")).strip())
        if empty:
            fail(f"{split}: {empty} empty input/target rows")
        n = len(t)
        scan = t if not args.leak_scan else t[:args.leak_scan]
        leaks = sum(1 for r in scan
                    if len(str(r.get("target", "")).strip()) > 20
                    and str(r["target"]).strip() in str(r.get("input", "")))
        if leaks:
            fail(f"{split}: {leaks}/{len(scan)} target-in-input LEAKS")
        bad = sum(1 for r in t if not str(r.get("target", "")).strip().upper()
                  .startswith(LEADERS))
        if bad:
            fail(f"{split}: {bad} targets with non-Cypher leading clause")
        un = sum(1 for r in t if r.get("label") == "unanswerable")
        print(f"[ok] {split}: 0 leaks, 0 bad targets, unanswerable={un}", flush=True)

    for f in ["config.json", "model.safetensors", "tokenizer.json"]:
        if not os.path.exists(os.path.join(args.base_dir, f)):
            fail(f"base model missing {f} in {args.base_dir}")
    print("[ok] base model files present", flush=True)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True)
        assert tok.vocab_size > 1000
        print(f"[ok] tokenizer loads (vocab={tok.vocab_size})", flush=True)
    except Exception as e:
        fail(f"tokenizer load failed: {e}")

    try:
        import torch
        ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[ok] torch {torch.__version__}, cuda gpus={ngpu}", flush=True)
        if ngpu < 1:
            fail("no CUDA GPU visible")
    except ImportError as e:
        fail(f"torch import failed: {e}")

    import py_compile
    for mod in ["train.py", "dataset.py"]:
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "legacy", mod)
        try:
            py_compile.compile(mp, doraise=True)
            print(f"[ok] legacy/{mod} compiles", flush=True)
        except Exception as e:
            fail(f"legacy/{mod} broken: {e}")

    if not args.hf_token:
        print("[warn] HF_TOKEN empty: private base download + checkpoint "
              "upload will fail", flush=True)
    print("ALL GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
