"""Merge a LoRA adapter into its base model -> standalone merged base.

Usage:
    python cli.py merge-latest --base_model <dir> --adapter_dir <dir> --out_dir <dir>

Fails loudly if either side is missing or incompatible (no silent fallbacks).
"""

from __future__ import annotations

import os

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def merge_latest(base_model: str, adapter_dir: str, out_dir: str,
                 dtype: str = "fp16") -> str:
    for name, path in (("base_model", base_model), ("adapter_dir", adapter_dir)):
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"{name} not found: {path!r}")
    has_cfg = os.path.exists(os.path.join(base_model, "config.json"))
    has_adapter = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    if not has_cfg:
        raise FileNotFoundError(f"{base_model} has no config.json - not a model dir")
    if not has_adapter:
        raise FileNotFoundError(f"{adapter_dir} has no adapter_config.json - not an adapter dir")

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    print(f"loading base: {base_model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, local_files_only=True, torch_dtype=torch_dtype)
    if getattr(model.config, "decoder_start_token_id", None) is None:
        fallback = getattr(getattr(model.config, "decoder", None), "bos_token_id", None)
        if fallback is not None:
            model.config.decoder_start_token_id = fallback
    print(f"loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("merging...")
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        tok.save_pretrained(out_dir)
    except Exception as e:
        print(f"warning: tokenizer copy failed ({e}) - merge still valid")
    print(f"merged base written to {out_dir}")
    return out_dir
