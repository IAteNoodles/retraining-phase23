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
from transformers import (LogitsProcessor, LogitsProcessorList,
                            NoRepeatNGramLogitsProcessor,
                            RepetitionPenaltyLogitsProcessor, Seq2SeqTrainer)

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


def repetition(text: str) -> float:
    """0 = no repeat .. ~1 = degenerate loop. 1 - unique/total bigram ratio."""
    toks = text.split()
    if len(toks) < 8:
        return 0.0
    bi = list(zip(toks, toks[1:]))
    return 1.0 - len(set(bi)) / len(bi)


def simplicity(text: str, gold: str) -> float:
    """<= 0. Punishes hard: relative length error vs gold, bigram repetition
    loops, and bracket imbalance."""
    lp, lg = len(text.split()), max(len(gold.split()), 1)
    drift = abs(lp - lg) / lg
    rep = repetition(text)
    brackets = abs(text.count("(") - text.count(")")) + abs(text.count("{") - text.count("}"))
    return -(drift + 3.0 * rep + 0.5 * brackets)


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


def _gen_processors(cfg) -> LogitsProcessorList:
    """Shared sampling guards for RL generation. Repetition controls are
    opt-in (defaults = off) so the base config is untouched unless asked."""
    procs: list = [_ClampLogits()]
    if cfg.train_noise_scale > 0:
        procs.append(_GaussianNoise(cfg.train_noise_scale))
    if getattr(cfg, "train_no_repeat_ngram_size", 0) and cfg.train_no_repeat_ngram_size > 0:
        procs.append(NoRepeatNGramLogitsProcessor(cfg.train_no_repeat_ngram_size))
    if getattr(cfg, "train_repetition_penalty", 1.0) not in (0.0, 1.0):
        procs.append(RepetitionPenaltyLogitsProcessor(cfg.train_repetition_penalty))
    return LogitsProcessorList(procs)


class _PerRowTemperature(LogitsProcessor):
    """Per-sequence temperature inside ONE batched generate call.

    Sampling with temperature T is exactly `sample(top_k(scores / T))`,
    so tiling prompts G-wide and scaling each row by its own T reproduces
    G separate per-temperature calls with a single forward pass per step.
    Must run AFTER noise (mirrors HF order: processors -> warpers) and
    the generate call itself uses temperature=1.0 (neutral).
    """

    def __init__(self, temps):
        self.temps = list(temps)

    def __call__(self, input_ids, scores):
        t = torch.as_tensor(self.temps, device=scores.device,
                            dtype=scores.dtype).unsqueeze(1)
        return scores / t.clamp_min(1e-3)


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
        i, c, cand, gold, db, inp = task
        r = compute_reward(cand, gold, self.neo4j_client, database=db,
                           input_text=inp)
        em = 1.0 if cand == gold else 0.0
        base = (r["jaccard"] + cfg.exact_match_weight * em
                + cfg.structure_coef * r.get("structure", 0.0)
                + cfg.schema_coef * r.get("schema", 0.0)
                + cfg.entity_coef * r.get("entity", 0.0)
                - cfg.trivial_coef * r.get("trivial", 0.0)
                - cfg.dbcost_coef * r.get("dbcost", 0.0)
                - cfg.complexity_coef * r.get("excess", 0)
                + r.get("cross", 0.0))
        return r, em, simplicity(cand, gold), base
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
                           structs, schemas, entities, branches,
                           crosses, excesses,
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
                        "reward_structure": round(float(structs[i, c]), 4),
                        "reward_schema": round(float(schemas[i, c]), 4),
                        "reward_entity": round(float(entities[i, c]), 4),
                        "reward_cross": round(float(crosses[i, c]), 4),
                        "reward_excess": int(excesses[i, c]),
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
                procs = _gen_processors(cfg)
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
                # spread temperatures: tile prompts G-wide, one call, per-row T.
                # Row layout is prompt-major ([p0c0,p0c1,p0c2,p1c0,...]) so the
                # downstream view(B, G, -1) recovers per-candidate batches.
                B = inputs["input_ids"].shape[0]
                tiled_ids = inputs["input_ids"].repeat_interleave(G, dim=0)
                tiled_mask = inputs["attention_mask"].repeat_interleave(G, dim=0)
                procs = _gen_processors(cfg)
                procs.append(_PerRowTemperature(temps * B))
                g = model_t.generate(
                    input_ids=tiled_ids,
                    attention_mask=tiled_mask,
                    max_new_tokens=gen_len,
                    do_sample=True, temperature=1.0, top_k=cfg.train_top_k,
                    logits_processor=procs)
                g = g.view(B, G, -1).transpose(0, 1).reshape(B * G, -1)
                gen_batches = [g[c * B:(c + 1) * B] for c in range(G)]
            model_t.config.use_cache = False
            if had_ckpt:
                try:
                    model_t.gradient_checkpointing_enable()
                except Exception:
                    pass
            if self.state.global_step % 20 == 0:
                batched = "batched" if (G > 1 and len(set(temps)) == 1) else "tiled"
                print(f"  [GRPO] step={self.state.global_step} G={G} temps={temps} "
                      f"top_k={cfg.train_top_k} noise={cfg.train_noise_scale} "
                      f"norepeat={cfg.train_no_repeat_ngram_size} "
                      f"reppen={cfg.train_repetition_penalty} ({batched})", flush=True)
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
        in_texts = self._ctx.get("input_text") or [None] * batch_size
        tasks = []
        for i in range(batch_size):
            for c in range(G):
                tasks.append((i, c, pred_texts[c * batch_size + i].strip(),
                              gold_texts[i].strip(),
                              databases[i] if i < len(databases) else None,
                              in_texts[i] if i < len(in_texts) else None))
        import concurrent.futures as _fut
        results = {}
        with _fut.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(self._reward_one, cfg, t): t for t in tasks}
            for fut in _fut.as_completed(futs):
                i, c, _, _, _, _ = futs[fut]
                results[(i, c)] = fut.result()
        t_reward_done = _time.perf_counter()

        base_r, exec_sets, simps = [], [], []
        branches, raw_jac = [], []
        structs, schemas, entities = [], [], []
        crosses, excesses = [], []
        for i in range(batch_size):
            for c in range(G):
                r, em, s_, base = results[(i, c)]
                base_r.append(base)
                raw_jac.append(r["jaccard"])
                structs.append(r.get("structure", 0.0))
                schemas.append(r.get("schema", 0.0))
                entities.append(r.get("entity", 0.0))
                crosses.append(r.get("cross", 0.0))
                excesses.append(r.get("excess", 0))
                branches.append(r.get("branch", "?"))
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

        # circuit breaker: if NO candidate has a valid gold execution, the
        # rewards are pure shaping noise - skip RL, keep SFT only
        if all(b == "gold_exec_fail" for b in branches):
            print(f"  [RL_SKIP] step={self.state.global_step}: all gold executions "
                  f"failed (check Neo4j URL/auth/mapping) - SFT loss only", flush=True)
            del gen_ids
            gc.collect()
            return (ce_loss, None) if return_outputs else ce_loss

        # ── 2b. log EVERY generation: candidate text, reward parts, group stats ──
        if cfg.rolling_jsonl_path:
            _stm = np.array(structs, dtype=np.float64).reshape(batch_size, G)
            _shm = np.array(schemas, dtype=np.float64).reshape(batch_size, G)
            _enm = np.array(entities, dtype=np.float64).reshape(batch_size, G)
            _xsm = np.array(crosses, dtype=np.float64).reshape(batch_size, G)
            _exm = np.array(excesses, dtype=np.float64).reshape(batch_size, G)
            self._log_generations(cfg, inputs, pred_texts, gold_texts, databases,
                                  base_r, divs, simps, cons, rewards, temps,
                                  _stm, _shm, _enm, branches,
                                  _xsm, _exm,
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

        # ── 3b. console scoreboard: every candidate, every score, every step ──
        _rj = np.array(raw_jac, dtype=np.float64).reshape(batch_size, G)
        _br = np.array(branches, dtype=object).reshape(batch_size, G)
        _sp = np.array(simps, dtype=np.float64).reshape(batch_size, G)
        _st = np.array(structs, dtype=np.float64).reshape(batch_size, G)
        _xs = np.array(crosses, dtype=np.float64).reshape(batch_size, G)
        print(f"  [GRPO_SCORES] step={self.state.global_step}", flush=True)
        for i in range(batch_size):
            xs = ''.join('!' if x < 0 else '.' for x in _xs[i])
            print(f"    prompt{i}: "
                  f"jac=[{','.join(f'{x:.2f}' for x in _rj[i])}] "
                  f"struct=[{','.join(f'{x:.2f}' for x in _st[i])}] "
                  f"div=[{','.join(f'{x:.2f}' for x in divs[i])}] "
                  f"simp=[{','.join(f'{x:.2f}' for x in _sp[i])}] "
                  f"cons=[{','.join(f'{x:.0f}' for x in cons[i])}] "
                  f"tot=[{','.join(f'{x:.2f}' for x in rewards[i])}] "
                  f"adv=[{','.join(f'{x:+.2f}' for x in adv[i])}] "
                  f"branch=[{','.join(str(x) for x in _br[i])}] xs={xs}", flush=True)

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
