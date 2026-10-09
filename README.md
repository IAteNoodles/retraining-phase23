# NL-to-Cypher retraining harness (phases 2/3)

Modular rebuild of the old `train.py` monolith. Every run starts with data
validation (split overlap, input leakage, schema coverage); training is split
into SFT and RL stages sharing one eval module.

## Layout

```
_retraining_phase23/
  cli.py                 # train / eval / merge-latest; works locally + on Kaggle
  requirements.txt
  retrain/
    config.py            # RunConfig dataclass + argparse (+ v1 aliases)
    dataset.py           # Text2CypherDataset + collator (clean input building)
    data_checks.py       # asserts run BEFORE any training
    neo4j_client.py      # Bolt + HTTPS Query API client
    rewards.py           # execution-grounded reward (database always routed)
    sft.py               # supervised fine-tuning + best tracking + early stopping
    rl.py                # GRPO: group advantages, shaping rewards, 2-model KL
    evaluate.py          # generation eval shared by train + test (+ rolling logs)
    merge.py             # merge-latest: base + adapter -> merged base
    sync.py              # HF upload + hf:// resume pointers
    flow.py              # phase runner: validate -> train -> eval
  notebooks/
    kaggle_phase23.ipynb # slim: install, clone repo, run cli.py
```

## Hardware profile (defaults target this)

RTX 4050 6 GB (≈5 GB usable) + 16 GB RAM: `batch_size=2`,
`grad_accum=16` (effective 32), fp16 4-bit QLoRA, `max_input_length=1024`.
GRPO with G=4 candidates at batch 2 ≈ 8 concurrent generations — the heaviest
op; lower `--rl_candidates` to 2 if VRAM trips. On bigger GPUs raise
`--batch_size` and drop `--grad_accum` proportionally.

## Rolling logs

Every RL step appends one record **per candidate** (text, temperature,
reward decomposition, group mean/std, database) to `rolling_data.jsonl`.
After each eval the file is archived (`rolling_data.{label}.{timestamp}.jsonl`)
and reset to blank, so each archive covers exactly one interval.

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

GRPO fine-tuning (needs Neo4j for rewards):

```bash
python cli.py train --phase 2 ... \
  --use_rl --rl_candidates 4 \
  --diversity_coef 0.1 --simplicity_coef 0.05 --consistency_coef 0.1 \
  --kl_coef 0.05 --kl_ref_model_path /path/to/sft-base \
  --neo4j_url "https://host/db/{database}/query/v2" \
  --neo4j_user training --neo4j_password trainig_phase23 \
  --db_mapping_json /path/to/db_name_mapping.json
```

Resume locally (or from Hub):

```bash
python cli.py train --phase 2 ... --resume_checkpoint ./checkpoints/p2/checkpoint-500
python cli.py train --phase 2 ... --resume_checkpoint hf://user/repo/phase_2/checkpoint-500 \
  --hf_token $HF_TOKEN
```

Eval only, and merging:

```bash
python cli.py eval --eval_model_dir ./checkpoints/p2/best \
  --val_parquet ... --test_parquet ... --schema_col schema_full
python cli.py merge-latest --base_model /path/to/base \
  --adapter_dir ./checkpoints/p2/best --out_dir ./merged_p2
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
