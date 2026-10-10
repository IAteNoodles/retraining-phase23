"""Clean dataset: builds model inputs ONLY from schema+question columns.

Model input is always `schema: {schema}\\nquestion: {question}`.
The target NEVER appears in the input (columns like `input_full` in older
exports embed `\\ntarget: ...` and must not be used as model input).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class Text2CypherDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_input_length: int = 1024,
                 max_target_length: int = 320, schema_col: str = "schema_full",
                 question_col: str = "question", target_col: str = "target",
                 schema_variant: str = "", max_samples: int = 0, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        suffix = data_path.lower().rsplit(".", 1)[-1] if "." in data_path else ""
        if suffix == "parquet":
            ds = load_dataset("parquet", data_files=data_path, split="train")
        elif suffix == "jsonl":
            ds = load_dataset("json", data_files=data_path, split="train")
        else:
            raise ValueError(f"unsupported dataset format: {data_path}")

        if schema_variant and "schema_variant" in ds.column_names:
            keep = [v.strip() for v in schema_variant.split(",") if v.strip()]
            ds = ds.filter(lambda x: x["schema_variant"] in keep)
        for col in (schema_col, question_col, target_col):
            if col not in ds.column_names:
                raise ValueError(f"{data_path}: missing column {col!r}")
        if "database" not in ds.column_names:
            raise ValueError(f"{data_path}: missing 'database' column (needed for RL/eval routing)")

        if max_samples and max_samples > 0 and len(ds) > max_samples:
            rng = random.Random(seed)
            ds = ds.select(sorted(rng.sample(range(len(ds)), int(max_samples))))

        self.gold_answers = ds["gold_answer"] if "gold_answer" in ds.column_names else [None] * len(ds)
        self.databases = ds["database"]
        self.input_texts = [
            f"schema: {str(s).strip()}\nquestion: {str(q).strip()}"
            for s, q in zip(ds[schema_col], ds[question_col])
        ]
        targets = [str(t) for t in ds[target_col]]
        for inp, tgt in zip(self.input_texts, targets):
            if len(tgt.strip()) > 20 and tgt.strip() in inp:
                raise ValueError("target text found inside a model input - refusing to build dataset")

        def tokenize_fn(batch):
            rendered = [f"schema: {str(s).strip()}\nquestion: {str(q).strip()}"
                        for s, q in zip(batch[schema_col], batch[question_col])]
            model_inputs = tokenizer(rendered, max_length=max_input_length,
                                     padding=False, truncation=True, add_special_tokens=True)
            label_enc = tokenizer(batch[target_col], max_length=max_target_length - 1,
                                  padding=False, truncation=True, add_special_tokens=True)
            eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
            labels = []
            for ids in label_enc["input_ids"]:
                ids = list(ids)
                if eos is not None and (not ids or ids[-1] != eos):
                    ids = ids + [eos]
                if pad is not None:
                    ids = [-100 if t == pad else t for t in ids]
                labels.append(ids)
            model_inputs["labels"] = labels
            model_inputs["input_length"] = [len(x) for x in model_inputs["input_ids"]]
            model_inputs["target_length"] = [len(x) for x in labels]
            model_inputs["length"] = [a + b for a, b in zip(model_inputs["input_length"],
                                                            model_inputs["target_length"])]
            return model_inputs

        self.dataset = ds.map(tokenize_fn, batched=True, batch_size=1000,
                              remove_columns=ds.column_names, desc="Tokenizing")
        self.dataset.set_format(type="torch",
                                columns=["input_ids", "attention_mask", "labels", "length",
                                         "input_length", "target_length"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["gold_answer"] = self.gold_answers[idx]
        item["database"] = self.databases[idx]
        item["input_text"] = self.input_texts[idx]
        item["index"] = idx
        return item


@dataclass
class Text2CypherCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "attention_mask": [f["attention_mask"] for f in features]},
            padding=True, return_tensors="pt")
        label_lists = [f["labels"] for f in features]
        max_len = max(x.size(0) for x in label_lists)
        labels = torch.full((len(label_lists), max_len), -100, dtype=torch.long)
        for i, lbl in enumerate(label_lists):
            labels[i, :lbl.size(0)] = lbl
        return {"input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels,
                "gold_answer": [f.get("gold_answer") for f in features],
                "database": [f.get("database") for f in features],
                "input_text": [f.get("input_text") for f in features],
                "index": [f.get("index") for f in features]}
