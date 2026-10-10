"""Phase runner: validate data -> build model -> train (SFT, optional RL) -> eval.

One phase = one set of files. Nothing is silently filtered; anything
unexpected fails before a single gradient step.
"""

from __future__ import annotations

import gc
import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .config import RunConfig
from .data_checks import validate
from .dataset import Text2CypherDataset, Text2CypherCollator
from .evaluate import run_generation_eval
from .neo4j_client import Neo4jClient
from .sft import run_sft


def resolve_model(path: str, token: str = "") -> str:
    """Local dir wins; otherwise treated as a HuggingFace repo id (whole repo)
    or hf://repo/subpath (subdirectory only), snapshot-downloaded and cached
    under ./models/."""
    if path and os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    if path.startswith("hf://"):
        rest = path[len("hf://"):].strip("/")
        parts = rest.split("/")
        if len(parts) < 3:
            raise ValueError(f"bad model spec (want hf://owner/repo/subpath): {path!r}")
        repo_id = "/".join(parts[:2])
        subpath = "/".join(parts[2:])
        dest = os.path.join("models", (repo_id + "__" + subpath).replace("/", "__"))
        if not os.path.exists(os.path.join(dest, "config.json")):
            print(f"downloading {repo_id}/{subpath} -> {dest}")
            snapshot_download(repo_id=repo_id, repo_type="model",
                              allow_patterns=[subpath.rstrip("/") + "/*"],
                              local_dir=dest, local_dir_use_symlinks=False,
                              token=token or None)
        else:
            print(f"reusing cached model: {dest}")
        out = os.path.join(dest, subpath)
        if not os.path.isdir(out):
            raise FileNotFoundError(f"downloaded model missing: {out!r}")
        return out
    if not path or "/" not in path:
        raise FileNotFoundError(
            f"model not found locally: {path!r} - pass a local dir, a Hub id "
            f"like 'org/model', or hf://repo/subpath")
    dest = os.path.join("models", path.replace("/", "__"))
    if not os.path.exists(os.path.join(dest, "config.json")):
        print(f"downloading base model {path} -> {dest}")
        snapshot_download(repo_id=path, local_dir=dest,
                          local_dir_use_symlinks=False, token=token or None)
    else:
        print(f"reusing cached model: {dest}")
    return dest


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg: RunConfig, tokenizer):
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    if cfg.use_quant:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_path, quantization_config=bnb, local_files_only=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path, local_files_only=True)
        model = model.to(dtype)
    if getattr(model.config, "decoder_start_token_id", None) is None:
        fallback = getattr(getattr(model.config, "decoder", None), "bos_token_id", None)
        if fallback is not None:
            model.config.decoder_start_token_id = fallback
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=cfg.lora_r, lora_alpha=cfg.lora_r * 2,
        lora_dropout=cfg.lora_dropout,
        target_modules=r"model\.(encoder\.text_model|decoder)\.layers\.\d+\."
                       r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    )
    model = get_peft_model(model, lora)
    if not cfg.use_quant:
        model = model.to(dtype)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    if cfg.compile_model:
        try:
            import torch as _torch
            model = _torch.compile(model)
            print("torch.compile enabled for training")
        except Exception as e:
            print(f"torch.compile unavailable, continuing uncompiled: {e}")
    return model


def build_neo4j(cfg: RunConfig):
    if not cfg.neo4j_url:
        return None
    mapping = {}
    if cfg.db_mapping_json and os.path.exists(cfg.db_mapping_json):
        with open(cfg.db_mapping_json) as f:
            mapping = json.load(f).get("mapping", {})
    return Neo4jClient(url=cfg.neo4j_url, user=cfg.neo4j_user,
                       password=cfg.neo4j_password,
                       fallback_db=cfg.neo4j_fallback_db, database_mapping=mapping)


def run_phase(cfg: RunConfig) -> str:
    print(f"=== phase {cfg.phase}: validating data ===")
    rep = validate(cfg.train_parquet, cfg.val_parquet, cfg.test_parquet,
                   cfg.schema_col, cfg.question_col, cfg.target_col)
    print(f"  rows: {rep.split_counts} | ambiguous: {rep.ambiguous_counts} | "
          f"schema-miss: {rep.schema_miss}")

    print("=== resolving base model (local dir or Hub download) ===")
    cfg.model_path = resolve_model(cfg.model_path, cfg.hf_token)
    if cfg.kl_ref_model_path:
        cfg.kl_ref_model_path = resolve_model(cfg.kl_ref_model_path, cfg.hf_token)

    print("=== building model + data ===")
    tokenizer = load_tokenizer(cfg.model_path)
    model = build_model(cfg, tokenizer)

    def _mk(path, cap):
        return Text2CypherDataset(
            path, tokenizer, max_input_length=cfg.max_input_length,
            max_target_length=cfg.max_target_length, schema_col=cfg.schema_col,
            question_col=cfg.question_col, target_col=cfg.target_col,
            schema_variant=cfg.schema_variant,
            max_samples=cap, seed=cfg.seed) if path else None

    train_ds = _mk(cfg.train_parquet, cfg.max_train_samples)
    val_ds = _mk(cfg.val_parquet, cfg.max_val_samples)
    print(f"  train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0}")
    if val_ds is None:
        print("  WARNING: no val set - no in-training eval, no best tracking; "
              "final adapter will be saved as 'final'")

    os.makedirs(cfg.output_dir, exist_ok=True)
    if not cfg.rolling_jsonl_path:
        cfg.rolling_jsonl_path = os.path.join(cfg.output_dir, "rolling_data.jsonl")
    neo4j = build_neo4j(cfg) if cfg.neo4j_url else None
    if cfg.use_rl and neo4j is None:
        raise ValueError("--use_rl requires --neo4j_url (reward backend)")
    if neo4j is not None and not cfg.neo4j_password:
        print("  [WARN] Neo4j password is empty - Bolt auth will fail and every "
              "reward will be gold_exec_fail. Pass --neo4j_password.", flush=True)
    kl_ref = None
    if cfg.use_rl and cfg.kl_coef > 0:
        from transformers import AutoModelForSeq2SeqLM as _AM, BitsAndBytesConfig as _BNB
        kl_path = resolve_model(cfg.kl_ref_model_path or cfg.model_path, cfg.hf_token)
        print(f"loading frozen KL reference (4-bit, same as policy): {kl_path}")
        _bnb = _BNB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True)
        kl_ref = _AM.from_pretrained(kl_path, local_files_only=True,
                                     quantization_config=_bnb)
        kl_ref.eval()
        for p in kl_ref.parameters():
            p.requires_grad_(False)
        # sanity: policy base weights and KL ref weights must match when they
        # are supposed to be the same checkpoint - otherwise KL is meaningless
        try:
            base_sd = model.base_model.model.state_dict()
            ref_sd = kl_ref.state_dict()
            common = [k for k in base_sd if k in ref_sd][:8]
            diffs = [(k, float((base_sd[k].float() - ref_sd[k].float()).abs().max()))
                     for k in common]
            worst = max(diffs, key=lambda x: x[1]) if diffs else ("?", float("nan"))
            print(f"  [kl-check] compared {len(common)} tensors vs ref; "
                  f"max abs diff {worst[1]:.4f} on {worst[0]}", flush=True)
            if worst[1] > 0.05:
                print("  [kl-check] WARNING: policy base and KL ref DIFFER - "
                      "KL penalty is measuring checkpoint mismatch, not drift. "
                      "Check --model_path vs --kl_ref_model_path.", flush=True)
        except Exception as e:
            print(f"  [kl-check] skipped ({e})", flush=True)
    best_dir = run_sft(model, tokenizer, train_ds, val_ds, cfg,
                       resume_checkpoint=cfg.resume_checkpoint,
                       use_rl=cfg.use_rl, neo4j_client=neo4j,
                       kl_ref_model=kl_ref)
    if neo4j is not None:
        neo4j.close()
    del kl_ref
    if cfg.upload_checkpoints and cfg.hf_repo_id:
        from .sync import upload_dir
        prefix = (cfg.hf_path_prefix.strip("/") + "/") if cfg.hf_path_prefix else ""
        upload_dir(cfg.output_dir, cfg.hf_repo_id,
                   f"{prefix}phase_{cfg.phase}", token=cfg.hf_token)
    print(f"=== phase {cfg.phase} done; best adapter: {best_dir} ===")
    return best_dir


def run_eval(cfg: RunConfig, model_dir: str):
    from peft import PeftModel
    cfg.model_path = resolve_model(cfg.model_path, cfg.hf_token)
    tokenizer = load_tokenizer(cfg.model_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16)
    model = PeftModel.from_pretrained(base, model_dir).merge_and_unload()
    model.to("cuda").eval()
    collator = Text2CypherCollator(tokenizer)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(model_dir)), "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    neo4j = build_neo4j(cfg)

    def _mk(path, cap):
        return Text2CypherDataset(
            path, tokenizer, max_input_length=cfg.max_input_length,
            max_target_length=cfg.max_target_length, schema_col=cfg.schema_col,
            question_col=cfg.question_col, target_col=cfg.target_col,
            schema_variant=cfg.schema_variant,
            max_samples=cap, seed=cfg.seed) if path and os.path.exists(path) else None

    if not cfg.final_test_only:
        val_ds = _mk(cfg.val_parquet, cfg.max_val_samples)
        if val_ds is not None:
            run_generation_eval(model, val_ds, tokenizer, collator, cfg.max_target_length,
                                os.path.join(out_dir, "val_metrics.json"), label="val",
                                eval_batch_size=cfg.eval_batch_size or cfg.batch_size,
                                neo4j_client=neo4j,
                                exec_jaccard_threshold=cfg.exec_jaccard_threshold,
                                max_samples=cfg.eval_samples,
                                rolling_jsonl_path=cfg.rolling_jsonl_path or os.path.join(out_dir, "rolling_data.jsonl"),
                                rolling_json_max_records=cfg.rolling_json_max_records,
                                phase_name=f"phase{cfg.phase}")
    test_ds = _mk(cfg.test_parquet, cfg.max_test_samples)
    if test_ds is not None:
        run_generation_eval(model, test_ds, tokenizer, collator, cfg.max_target_length,
                            os.path.join(out_dir, "test_metrics.json"), label="test",
                            eval_batch_size=cfg.eval_batch_size or cfg.batch_size,
                            neo4j_client=neo4j,
                            exec_jaccard_threshold=cfg.exec_jaccard_threshold,
                            max_samples=cfg.eval_samples,
                            rolling_jsonl_path=cfg.rolling_jsonl_path or os.path.join(out_dir, "rolling_data.jsonl"),
                            rolling_json_max_records=cfg.rolling_json_max_records,
                            phase_name=f"phase{cfg.phase}")
    if neo4j is not None:
        neo4j.close()
    if cfg.upload_checkpoints and cfg.hf_repo_id:
        from .sync import upload_dir
        prefix = (cfg.hf_path_prefix.strip("/") + "/") if cfg.hf_path_prefix else ""
        upload_dir(out_dir, cfg.hf_repo_id,
                   f"{prefix}phase_{cfg.phase}_eval", token=cfg.hf_token)
