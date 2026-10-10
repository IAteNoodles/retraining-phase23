"""GRPO trainer (Group Relative Policy Optimization, Shao et al. 2024).

Per prompt, sample a GROUP of G candidates from the current policy, then:
  reward_i  = exec_jaccard + em_bonus + div_coef*diversity_i
              + simp_coef*simplicity_i + cons_coef*consistency_i
  advantage = (reward_i - mean(rewards)) / (std(rewards) + eps)   # no critic
  loss      = -mean(advantage * mean_token_logprob) + kl_coef * KL(policy || ref)

Shaping terms:
- diversity_i    = 1 - mean pairwise bigram-Jaccard vs other candidates
                   (rewards the group for covering different phrasings)
- simplicity_i   = -(length drift vs gold) - bracket mismatch
                   (rewards concise, balanced queries)
- consistency_i  = 1 if candidate's execution result == group majority result
                   else 0 (rewards stable answers; exec failures score 0)

KL uses a frozen reference model (two-model setup). No PPO clipping, no
adaptive controller, no critic - canonical GRPO.
"""

from __future__ import annotations

import gc
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, Seq2SeqTrainer

from .batching import BucketDataloaderMixin
from .oom_safe import OOMSafeMixin


def _bigrams(text: str) -> set:
    toks = text.split()
    return set(zip(toks, toks[1:])) if len(toks) > 1 else set(toks)


def pairwise_dissimilarity(texts: list[str]) -> list[float]:
    """1 - mean bigram-Jaccard vs every other candidate. All-identical -> 0."""
    bags = [_bigrams(t) for t in texts]
    out = []
    for i, bi in enumerate(bags):
        sims = []
        for j, bj in enumerate(bags):
            if i == j:
                continue
            union = bi | bj
            sims.append(len(bi & bj) / len(union) if union else 1.0)
        out.append(1.0 - (sum(sims) / len(sims) if sims else 0.0))
    return out


def simplicity(text: str, gold: str) -> float:
    """<= 0. Compares length and bracket balance against gold."""
    lp, lg = len(text.split()), max(len(gold.split()), 1)
    drift = abs(lp / lg - 1.0)
    brackets = abs(text.count("(") - text.count(")")) + abs(text.count("{") - text.count("}"))
    return -(drift + 0.5 * brackets)


class _ClampLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        return torch.where(torch.isfinite(scores), scores,
                           torch.tensor(0.0, device=scores.device, dtype=scores.dtype))


class _GaussianNoise(LogitsProcessor):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, input_ids, scores):
        if self.scale > 0:
            return scores + torch.randn_like(scores) * self.scale
        return scores


class GRPOTrainer(BucketDataloaderMixin, OOMSafeMixin, Seq2SeqTrainer):
    def __init__(self, *args, tokenizer=None, cfg=None, neo4j_client=None,
                 kl_ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cfg is None:
            raise ValueError("GRPOTrainer requires cfg")
        if cfg.rl_candidates < 2:
            raise ValueError(f"GRPO needs rl_candidates >= 2 (got {cfg.rl_candidates}): "
                             f"advantages are computed within the group")
        if cfg.kl_coef > 0 and kl_ref_model is None:
            raise ValueError("kl_coef > 0 requires a frozen reference model "
                             "(--kl_ref_model_path)")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.neo4j_client = neo4j_client
        self.kl_ref_model = kl_ref_model
        if kl_ref_model is not None:
            kl_ref_model.eval()
            for p in kl_ref_model.parameters():
                p.requires_grad_(False)
        self.reward_ema = 0.5  # logging only; GRPO baseline is the group mean
        self._rl_counter = 0
        self._ctx = {"gold_answer": None, "database": None, "input_text": None}

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, dict):
            self._ctx = {k: inputs.pop(k, None)
                         for k in ("gold_answer", "database", "input_text")}
        return super()._prepare_inputs(inputs)

    def _reward_one(self, cfg, task):
        from .rewards import compute_reward
        i, c, cand, gold, db = task
        r = compute_reward(cand, gold, self.neo4j_client, database=db)
        em = 1.0 if cand == gold else 0.0
        return r, em, simplicity(cand, gold)

    def _candidate_temperature(self, c: int, total_steps: int) -> float:
        cfg = self.cfg
        if total_steps > 0:
            progress = min(1.0, max(0.0, self.state.global_step / max(total_steps, 1)))
        else:
            progress = 0.0
        spread = cfg.temp_spread_min + (cfg.temp_spread - cfg.temp_spread_min) * (1.0 - progress)
        t = cfg.train_temperature * (1.0 + c * spread)
        return max(cfg.train_temperature_min, t)

    def _log_generations(self, cfg, inputs, pred_texts, gold_texts, databases,
                           base_r, divs, simps, cons, rewards, temps,
                           batch_size, G):
        """Append one record per candidate: text, reward decomposition, temps."""
        import datetime
        import json
        import os
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        in_texts = self._ctx.get("input_text") or [None] * batch_size
        os.makedirs(os.path.dirname(os.path.abspath(cfg.rolling_jsonl_path)), exist_ok=True)
        with open(cfg.rolling_jsonl_path, "a", encoding="utf-8") as f:
            for i in range(batch_size):
                for c in range(G):
                    f.write(json.dumps({
                        "time": now, "kind": "rl_candidate",
                        "step": int(self.state.global_step),
                        "temperature": temps[c] if c < len(temps) else None,
                        "input": in_texts[i] if i < len(in_texts) else None,
                        "output": pred_texts[c * batch_size + i].strip(),
                        "gold": gold_texts[i].strip(),
                        "reward_exec": round(float(base_r[i, c]), 4),
                        "reward_diversity": round(float(divs[i, c]), 4),
                        "reward_simplicity": round(float(simps[i * G + c]), 4),
                        "reward_consistency": round(float(cons[i, c]), 4),
                        "reward_total": round(float(rewards[i, c]), 4),
                        "group_mean": round(float(rewards[i].mean()), 4),
                        "group_std": round(float(rewards[i].std()), 4),
                        "database": databases[i] if i < len(databases) else None,
                    }) + "\n")
                    f.flush()  # visible to tail -Wait the moment it's written

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from .rewards import compute_reward

        cfg = self.cfg
        model_t = model.module if hasattr(model, "module") else model
        labels = inputs["labels"]
        self._rl_counter += 1
        do_rl = model.training and (self._rl_counter % max(1, cfg.rl_every_n_steps) == 0)

        outputs = model(**inputs)
        ce_loss = outputs.loss
        if not torch.isfinite(ce_loss):
            print(f"  [RL_WARN] non-finite CE at step={self.state.global_step}; using 0.0", flush=True)
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if not model.training:
            return (ce_loss, outputs) if return_outputs else ce_loss
        del outputs
        if not do_rl or self.state.global_step < cfg.rl_warmup_steps:
            return (ce_loss, None) if return_outputs else ce_loss

        pad_id = self.tokenizer.pad_token_id
        batch_size = inputs["input_ids"].shape[0]
        G = cfg.rl_candidates
        total_steps = getattr(self.args, "max_steps", 0) or 0
        gen_len = cfg.rl_max_new_tokens or self.args.generation_max_length
        import time as _time
        t_rl0 = _time.perf_counter()

        # defragment before the memory-heavy section (generation + G× forward)
        torch.cuda.empty_cache()
        gc.collect()

        # ── 1. sample a group of G candidates per prompt ──
        # NOTE: gradient checkpointing is disabled around generation: it only
        # saves memory for backward (irrelevant under no_grad) while forcing
        # use_cache=False, i.e. every decoded token recomputes the full prefix
        # (O(n^2) without KV cache). Disabling = pure speedup, zero downside.
        had_ckpt = False
        try:
            if hasattr(model_t, "gradient_checkpointing_disable"):
                model_t.gradient_checkpointing_disable()
                had_ckpt = True
        except Exception:
            had_ckpt = False
        with torch.no_grad():
            model_t.config.use_cache = True
            gen_batches, temps = [], []
            for c in range(G):
                temp = self._candidate_temperature(c, total_steps)
                temps.append(round(temp, 3))
            if G > 1 and len(set(temps)) == 1:
                # uniform temperature: one batched call, G sequences per prompt
                procs = LogitsProcessorList([_ClampLogits()])
                if cfg.train_noise_scale > 0:
                    procs.append(_GaussianNoise(cfg.train_noise_scale))
                g = model_t.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=gen_len,
                    do_sample=True, temperature=temps[0], top_k=cfg.train_top_k,
                    num_return_sequences=G, logits_processor=procs)
                B = inputs["input_ids"].shape[0]
                g = g.view(B, G, -1).transpose(0, 1).reshape(B * G, -1)
                gen_batches = [g[c * B:(c + 1) * B] for c in range(G)]
            else:
                for c in range(G):
                    procs = LogitsProcessorList([_ClampLogits()])
                    if cfg.train_noise_scale > 0:
                        procs.append(_GaussianNoise(cfg.train_noise_scale))
                    g = model_t.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=gen_len,
                        do_sample=True, temperature=temps[c], top_k=cfg.train_top_k,
                        logits_processor=procs)
                    gen_batches.append(g)
            model_t.config.use_cache = False
            if had_ckpt:
                try:
                    model_t.gradient_checkpointing_enable()
                except Exception:
                    pass
            if self.state.global_step % 20 == 0:
                batched = "batched" if (G > 1 and len(set(temps)) == 1) else "sequential"
                print(f"  [GRPO] step={self.state.global_step} G={G} temps={temps} "
                      f"top_k={cfg.train_top_k} noise={cfg.train_noise_scale} ({batched})", flush=True)
        gc.collect()
        max_len = max(t.shape[1] for t in gen_batches)
        gen_ids = torch.cat(
            [torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=pad_id)
             for t in gen_batches], dim=0)

        pred_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        labels_np = labels.cpu().numpy()
        labels_np = np.where(labels_np != -100, labels_np, pad_id)
        gold_texts = self.tokenizer.batch_decode(labels_np, skip_special_tokens=True)
        databases = self._ctx["database"] or [None] * batch_size
        t_gen_done = _time.perf_counter()

        # ── 2. reward every candidate (threaded: Neo4j calls are I/O-bound) ──
        tasks = []
        for i in range(batch_size):
            for c in range(G):
                tasks.append((i, c, pred_texts[c * batch_size + i].strip(),
                              gold_texts[i].strip(),
                              databases[i] if i < len(databases) else None))
        import concurrent.futures as _fut
        results = {}
        with _fut.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(self._reward_one, cfg, t): t for t in tasks}
            for fut in _fut.as_completed(futs):
                i, c, _, _, _ = futs[fut]
                results[(i, c)] = fut.result()
        t_reward_done = _time.perf_counter()

        base_r, exec_sets, simps = [], [], []
        for i in range(batch_size):
            for c in range(G):
                r, em, s_ = results[(i, c)]
                base_r.append(r["jaccard"] + cfg.exact_match_weight * em)
                try:
                    exec_sets.append(r.get("exec_set"))
                except Exception:
                    exec_sets.append(None)
                simps.append(s_)

        base_r = np.array(base_r, dtype=np.float64).reshape(batch_size, G)

        # diversity is group-relative by construction
        divs = []
        for i in range(batch_size):
            group_texts = [pred_texts[c * batch_size + i].strip() for c in range(G)]
            divs.extend(pairwise_dissimilarity(group_texts))
        divs = np.array(divs, dtype=np.float64).reshape(batch_size, G)

        # consistency vs group majority execution result
        cons = np.zeros((batch_size, G))
        for i in range(batch_size):
            votes: dict = {}
            for c in range(G):
                key = exec_sets[i * G + c]
                if key is not None:
                    votes[key] = votes.get(key, 0) + 1
            if votes:
                majority = max(votes, key=votes.get)
                for c in range(G):
                    cons[i, c] = 1.0 if exec_sets[i * G + c] == majority else 0.0

        rewards = (base_r + cfg.diversity_coef * divs
                   + cfg.simplicity_coef * np.array(simps, dtype=np.float64).reshape(batch_size, G)
                   + cfg.consistency_coef * cons)

        # ── 2b. log EVERY generation: candidate text, reward parts, group stats ──
        if cfg.rolling_jsonl_path:
            self._log_generations(cfg, inputs, pred_texts, gold_texts, databases,
                                  base_r, divs, simps, cons, rewards, temps,
                                  batch_size, G)

        # ── 3. group-relative advantages (no critic) ──
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True) + 1e-4
        adv = (rewards - mean) / std
        # clamp: near-identical groups have ~0 std which would explode advantages
        adv = np.clip(adv, -5.0, 5.0)
        adv_t = torch.tensor(adv.reshape(-1), device=inputs["input_ids"].device,
                             dtype=torch.float32)
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * float(rewards.mean())

        # ── 4. policy gradient on group candidates, in chunks ──
        # A full G×B stacked forward blows VRAM (logits over 262K vocab);
        # process `rl_chunk_size` candidates at a time, accumulating the loss.
        dec_in = gen_ids[:, :-1].contiguous()
        tgt_ids = gen_ids[:, 1:].contiguous()
        n_all = batch_size * G
        chunk = max(1, min(cfg.rl_chunk_size, n_all))
        order = torch.randperm(n_all, device=gen_ids.device)
        acc_rl, acc_ent_num, acc_ent_den, acc_kl_num, acc_kl_den = 0.0, 0.0, 0.0, 0.0, 0.0
        need_ent = bool(cfg.entropy_coef)
        need_kl = bool(cfg.kl_coef and self.kl_ref_model is not None)
        for s in range(0, n_all, chunk):
            sl = order[s:s + chunk]
            dec_slice = dec_in[sl]
            tgt_slice = tgt_ids[sl]
            adv_slice = adv_t[sl]
            in_slice = inputs["input_ids"].repeat(G, 1)[sl]
            mask_slice = inputs["attention_mask"].repeat(G, 1)[sl]
            outputs2 = model(input_ids=in_slice, attention_mask=mask_slice,
                             decoder_input_ids=dec_slice)
            log_probs = F.log_softmax(outputs2.logits, dim=-1)
            token_lp = log_probs.gather(2, tgt_slice.unsqueeze(-1)).squeeze(-1)
            pmask = (tgt_slice != pad_id).float()
            seq_lp = (token_lp * pmask).sum(dim=-1)
            acc_rl = acc_rl + (seq_lp * adv_slice).sum() / n_all
            if need_ent or need_kl:
                probs = log_probs.detach().exp()
                if need_ent:
                    acc_ent_num = acc_ent_num + (-(probs * log_probs.detach()).sum(-1) * pmask).sum()
                    acc_ent_den = acc_ent_den + pmask.sum()
                if need_kl:
                    with torch.no_grad():
                        ref = self.kl_ref_model(input_ids=in_slice,
                                                attention_mask=mask_slice,
                                                decoder_input_ids=dec_slice)
                        ref_lp = F.log_softmax(ref.logits, dim=-1)
                    acc_kl_num = acc_kl_num + ((probs * (log_probs - ref_lp)).sum(-1) * pmask).sum()
                    acc_kl_den = acc_kl_den + pmask.sum()
                    del ref, ref_lp
                del probs
            del outputs2, log_probs
        grpo_loss = -acc_rl

        if cfg.entropy_coef:
            ent = acc_ent_num / acc_ent_den.clamp_min(1.0)
            ent_bonus = torch.clamp(ent - cfg.entropy_min, min=0.0)
            grpo_loss = grpo_loss - cfg.entropy_coef * ent_bonus
        else:
            ent = torch.tensor(0.0, device=grpo_loss.device)

        if need_kl:
            kl = acc_kl_num / acc_kl_den.clamp_min(1.0)
            kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
            grpo_loss = grpo_loss + cfg.kl_coef * kl
        else:
            kl = torch.tensor(0.0, device=grpo_loss.device)

        total = cfg.sft_coef * ce_loss + cfg.rl_coef * grpo_loss
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        import time as _t2
        t_end = _t2.perf_counter()
        print(f"  [GRPO_LOSS] grpo={grpo_loss.item():.4f} ent={ent.item():.4f} "
              f"kl={kl.item():.4f} ce={ce_loss.item():.4f} total={total.item():.4f} "
              f"mean_r={rewards.mean():.3f}", flush=True)
        print(f"  [GRPO_TIME] step={self.state.global_step} "
              f"gen={t_gen_done - t_rl0:.1f}s reward={t_reward_done - t_gen_done:.1f}s "
              f"backward={(t_end - t_reward_done):.1f}s", flush=True)
        return (total, None) if return_outputs else total
