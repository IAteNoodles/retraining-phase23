# NL-to-Cypher retraining harness (phases 2/3)

Modular rebuild of the old `train.py` monolith. Every run starts with data
validation (split overlap, input leakage, schema coverage); training is split
into SFT and RL stages sharing one eval module.

## Layout

```
_retraining_phase23/
  cli.py                 # entry point: train / eval-only, works locally + on Kaggle
  requirements.txt
  retrain/
    config.py            # RunConfig dataclass + argparse
    dataset.py           # Text2CypherDataset + collator (clean input building)
    data_checks.py       # asserts run BEFORE any training
    neo4j_client.py      # Bolt + HTTPS Query API client
    rewards.py           # execution-grounded reward (database always routed)
    sft.py               # supervised fine-tuning stage
    rl.py                # REINFORCE stage (fixed cadence, no dead knobs)
    evaluate.py          # generation eval shared by train + test
    flow.py              # phase runner: validate -> train -> eval
  notebooks/
    kaggle_phase23.ipynb # slim: install, clone repo, run cli.py
```

## Quick start (local)

```bash
pip install -r requirements.txt
python cli.py train --phase 2 \
  --train_parquet /path/to/phase2_train.parquet \
  --val_parquet   /path/to/phase2_val.parquet \
  --test_parquet  /path/to/phase2_test.parquet \
  --model_path /path/to/t5gemma-270m \
  --output_dir ./checkpoints/p2 \
  --schema_col schema_full --question_col question
```

Resume locally:

```bash
python cli.py train --phase 2 ... --resume_checkpoint ./checkpoints/p2/checkpoint-500
```

Eval only:

```bash
python cli.py eval --eval_model_dir ./checkpoints/p2/best \
  --val_parquet ... --test_parquet ... --schema_col schema_full
```

## Key differences from the old train.py

1. **Separate files per phase** (`--train_parquet` is exactly what gets trained on).
   No silent `phase=N` column filter; a missing/empty file is a hard error.
2. **Data validation is mandatory** (`data_checks.py`): exact train/val/test
   overlap fails the run; target-in-input leakage fails the run; schema
   coverage (every label/rel/prop used by gold must be in the schema) is
   reported and fails above threshold.
3. **Eval execution metrics route per-sample databases** (the old
   `compute_metrics` ran everything against the default DB).
   Reported honestly: `execution_accuracy` = fraction with result-Jaccard
   >= 0.85, plus mean Jaccard — never a raw-reward average.
4. **Best checkpoint is real**: a tracker copies the best adapter by the
   chosen metric (PEFT-safe), instead of mislabeling the final model "best".
5. **RL knobs are only the implemented ones**; cadence is explicit
   (`--rl_every_n_steps`, default 2 = old behavior).
6. **Neo4j is optional and explicit**: pass `--neo4j_url` (bolt or
   `https://host/db/{database}/query/v2`) plus `--db_mapping_json`.

## Data contract

Each parquet row used for training must have:
- `question`, `target` (gold Cypher), `database`
- one schema column (e.g. `schema_full` / `schema_compact`), referenced via `--schema_col`
- optional `gold_answer` (execution results; skips gold-query execution)

Model input is always built as `schema: {schema}\nquestion: {question}`.
Never use a column that embeds the target (`input_full` in older exports
contains `\ntarget: ...` and would leak the answer).
