import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import Dataset
from transformers import AutoTokenizer


UNANSWERABLE_PHRASES = [
    "question not supported by this schema",
    "this question is not supported by the given schema",
    "cannot answer this question with the provided schema",
    "not enough information in the schema to answer this question",
    "unable to answer this question using the given schema",
    "this question is not answerable with the current schema",
    "the schema does not contain enough information to answer this question",
]


class Text2CypherDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        phase: Optional[int] = None,
        max_input_length: int = 512,
        max_target_length: int = 256,
        input_col: str = "input",
        target_col: str = "target",
        input_format: str = "plain",
        schema_col: str = "schema",
        question_col: str = "question",
        max_samples: int = 0,
        sample_seed: int = 42,
    ):
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if local_rank not in {"0", "-1"}:
            disable_progress_bar()

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        suffix = data_path.lower().rsplit(".", 1)[-1] if "." in data_path else ""
        if suffix == "parquet":
            ds = load_dataset("parquet", data_files=data_path, split="train")
        elif suffix == "jsonl":
            ds = load_dataset("json", data_files=data_path, split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {data_path}")

        if phase is not None and "phase" in ds.column_names:
            ds = ds.filter(lambda x: x["phase"] == phase)

        if max_samples and max_samples > 0 and len(ds) > max_samples:
            rng = random.Random(sample_seed)
            selected = sorted(rng.sample(range(len(ds)), int(max_samples)))
            ds = ds.select(selected)

        self.gold_answers = ds["gold_answer"] if "gold_answer" in ds.column_names else [None] * len(ds)
        self.databases = ds["database"] if "database" in ds.column_names else [None] * len(ds)
        if input_format == "schema_question" and schema_col in ds.column_names and question_col in ds.column_names:
            self.input_texts = [
                f"schema: {str(schema).strip()}\nquestion: {str(question).strip()}"
                for schema, question in zip(ds[schema_col], ds[question_col])
            ]
        elif input_col in ds.column_names:
            self.input_texts = ds[input_col]
        else:
            self.input_texts = [None] * len(ds)

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        def tokenize_fn(batch):
            if input_format == "schema_question":
                schemas = batch[schema_col]
                questions = batch[question_col]
                rendered_inputs = [
                    f"schema: {str(s).strip()}\nquestion: {str(q).strip()}"
                    for s, q in zip(schemas, questions)
                ]
            else:
                rendered_inputs = batch[input_col]

            model_inputs = tokenizer(
                rendered_inputs,
                max_length=max_input_length,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            label_enc = tokenizer(
                batch[target_col],
                max_length=max_target_length - 1,
                padding=False,
                truncation=True,
                add_special_tokens=True,
            )
            labels = []
            for ids in label_enc["input_ids"]:
                ids = list(ids)
                # Append EOS to target labels for teacher forcing (T5Gemma decoder)
                if eos_token_id is not None:
                    ids = ids + [eos_token_id]
                if pad_token_id is not None:
                    ids = [-100 if t == pad_token_id else t for t in ids]
                labels.append(ids)
            model_inputs["labels"] = labels
            input_lengths = [len(x) for x in model_inputs["input_ids"]]
            target_lengths = [len(x) for x in labels]
            model_inputs["input_length"] = input_lengths
            model_inputs["target_length"] = target_lengths
            model_inputs["length"] = [inp + tgt for inp, tgt in zip(input_lengths, target_lengths)]
            return model_inputs

        self.dataset = ds.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "length", "input_length", "target_length"],
        )

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
    tokenizer: Any

    def __call__(self, features):
        for f in features:
            f.pop("length", None)
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        label_lists = [f["labels"] for f in features]
        max_label_len = max(x.size(0) for x in label_lists)
        labels = torch.full((len(label_lists), max_label_len), -100, dtype=torch.long)
        for i, lbl in enumerate(label_lists):
            labels[i, : lbl.size(0)] = lbl
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "gold_answer": [f.get("gold_answer") for f in features],
            "database": [f.get("database") for f in features],
            "input_text": [f.get("input_text") for f in features],
            "index": [f.get("index") for f in features],
        }


if __name__ == "__main__":
    MODEL_PATH = "t5gemma-270m"
    DATA_PATH = "splits/train.parquet"

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    ds = Text2CypherDataset(DATA_PATH, tok)

    print(f"Dataset size: {len(ds)}")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\n--- Sample {i} ---")
        print(f"  input_ids len:      {sample['input_ids'].shape[0]}")
        print(f"  attention_mask len: {sample['attention_mask'].shape[0]}")
        print(f"  labels len:         {sample['labels'].shape[0]}")
        decoded_inp = tok.decode(sample["input_ids"], skip_special_tokens=True)
        decoded_lbl = tok.decode(
            [t if t != -100 else tok.pad_token_id for t in sample["labels"].tolist()],
            skip_special_tokens=True,
        )
        print(f"  input (truncated):  {decoded_inp[:120]}")
        print(f"  target (truncated): {decoded_lbl[:120]}")