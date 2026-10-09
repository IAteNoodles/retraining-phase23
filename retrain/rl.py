"""REINFORCE stage. Differences from the old ReinforceTrainer:

- No global `args` leak: everything comes from the passed config object.
- RL cadence is explicit (`rl_every_n_steps`, default 2 = old behavior).
- Dead knobs removed (ppo_clip_eps, kl_adaptive_*, eos_suppress_*, diversity /
  simplicity / consistency, dynamic_generation_length). Only the implemented
  terms exist: SFT + REINFORCE + optional entropy + optional KL + critic.
- Critic defaults to OFF: an encoder-mean baseline cannot rank multiple
  candidates for the same input (identical input states). EMA baseline stays.
- Rewards use per-sample database routing via the collator-provided context.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList, Seq2SeqTrainer


class CriticNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, encoder_hidden_states):
        return self.net(encoder_hidden_states.mean(dim=1)).squeeze(-1)


class ReinforceTrainer(Seq2SeqTrainer):
    def __init__(self, *args, tokenizer=None, cfg=None, neo4j_client=None,
                 kl_ref_model=None, **kwargs):
        model = kwargs.get("model")
        hidden = None
        if model is not None:
            hidden = getattr(model.config, "d_model", None) or getattr(model.config, "hidden_size", None)
        self.critic = CriticNet(hidden).to(model.device) if hidden else None
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.neo4j_client = neo4j_client
        self.kl_ref_model = kl_ref_model
        self.reward_ema = 0.5
        self._rl_counter = 0
        self._ctx = {"gold_answer": None, "database": None, "input_text": None}

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, dict):
            self._ctx = {k: inputs.pop(k, None)
                         for k in ("gold_answer", "database", "input_text")}
        return super()._prepare_inputs(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from .rewards import compute_reward

        cfg = self.cfg
        model_t = model.module if hasattr(model, "module") else model
        labels = inputs["labels"]
        self._rl_counter += 1
        do_rl = model.training and (self._rl_counter % max(1, cfg.rl_every_n_steps) == 0)

        model_t.config.output_hidden_states = do_rl
        outputs = model(**inputs)
        ce_loss = outputs.loss
        encoder_hidden = outputs.encoder_last_hidden_state if do_rl else None
        model_t.config.output_hidden_states = False

        if not torch.isfinite(ce_loss):
            print(f"  [RL_WARN] non-finite CE at step={self.state.global_step}; using 0.0", flush=True)
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if not model.training:
            return (ce_loss, outputs) if return_outputs else ce_loss
        del outputs
        if not do_rl:
            return (ce_loss, None) if return_outputs else ce_loss
        if self.state.global_step < cfg.rl_warmup_steps:
            return (ce_loss, None) if return_outputs else ce_loss

        pad_id = self.tokenizer.pad_token_id
        batch_size = inputs["input_ids"].shape[0]
        n_cand = max(1, cfg.rl_candidates)

        with torch.no_grad():
            model_t.config.use_cache = True
            gen_batches = []
            for _ in range(n_cand):
                g = model_t.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.args.generation_max_length,
                    do_sample=(n_cand > 1),
                    temperature=cfg.train_temperature if n_cand > 1 else 1.0,
                    top_k=cfg.train_top_k if n_cand > 1 else 50,
                )
                gen_batches.append(g)
            model_t.config.use_cache = False
        gc.collect()

        if n_cand > 1:
            max_len = max(t.shape[1] for t in gen_batches)
            gen_ids = torch.cat(
                [torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=pad_id)
                 for t in gen_batches], dim=0)
        else:
            gen_ids = gen_batches[0]

        pred_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        labels_np = labels.cpu().numpy()
        labels_np = np.where(labels_np != -100, labels_np, pad_id)
        gold_texts = self.tokenizer.batch_decode(labels_np, skip_special_tokens=True)

        ctx = self._ctx
        gold_answers = ctx["gold_answer"] or [None] * batch_size
        databases = ctx["database"] or [None] * batch_size

        rewards, jaccards, ems = [], [], []
        best_ids = []
        for i in range(batch_size):
            idxs = [c * batch_size + i for c in range(n_cand)]
            cands = [pred_texts[j].strip() for j in idxs]
            scored = []
            for pos, c in enumerate(cands):
                r = compute_reward(c, gold_texts[i].strip(), self.neo4j_client,
                                   database=databases[i] if i < len(databases) else None)
                em = 1.0 if c == gold_texts[i].strip() else 0.0
                scored.append((r["jaccard"] + cfg.exact_match_weight * em,
                               r["jaccard"], em, idxs[pos]))
            scored.sort(key=lambda x: x[0], reverse=True)
            rewards.append(scored[0][0])
            jaccards.append(scored[0][1])
            ems.append(scored[0][2])
            best_ids.append(gen_ids[scored[0][3]])

        rewards_t = torch.tensor(rewards, device=inputs["input_ids"].device, dtype=torch.float32)
        if self.critic is not None and not cfg.disable_critic:
            critic_pred = self.critic(encoder_hidden.detach())
            advantage = rewards_t - critic_pred
        else:
            advantage = rewards_t - self.reward_ema
        self.reward_ema = ((1 - cfg.reward_ema_alpha) * self.reward_ema
                           + cfg.reward_ema_alpha * rewards_t.mean().item())
        if self.state.global_step % 20 == 0:
            print(f"  [R] step={self.state.global_step} "
                  f"j=[{','.join(f'{j:.2f}' for j in jaccards)}] "
                  f"rwd=[{','.join(f'{r:.2f}' for r in rewards)}] "
                  f"ema={self.reward_ema:.3f} ce={ce_loss.item():.4f}", flush=True)

        best_ids = torch.stack(best_ids)
        dec_in = best_ids[:, :-1]
        tgt_ids = best_ids[:, 1:]
        outputs2 = model(input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"],
                         decoder_input_ids=dec_in)
        log_probs = F.log_softmax(outputs2.logits, dim=-1)
        token_lp = log_probs.gather(2, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = (tgt_ids != pad_id).float()
        seq_lp = (token_lp * mask).sum(dim=-1)

        rl_loss = -(seq_lp * advantage.detach()).mean()
        rl_loss = torch.nan_to_num(rl_loss, nan=0.0, posinf=0.0, neginf=0.0)
        total = cfg.sft_coef * ce_loss + cfg.rl_coef * rl_loss

        if cfg.entropy_coef:
            probs = log_probs.detach().exp()
            ent = (-(probs * log_probs.detach()).sum(-1) * mask).sum() / mask.sum().clamp_min(1.0)
            total = total - cfg.entropy_coef * torch.clamp(ent, min=0.0)
            del probs

        if cfg.kl_coef and self.kl_ref_model is not None:
            with torch.no_grad():
                ref = self.kl_ref_model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        decoder_input_ids=dec_in)
                ref_lp = F.log_softmax(ref.logits, dim=-1)
            kl = ((log_probs.exp() * (log_probs - ref_lp)).sum(-1) * mask).sum() / mask.sum().clamp_min(1.0)
            total = total + cfg.kl_coef * torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
            del ref, ref_lp

        if self.critic is not None and not cfg.disable_critic:
            cl = F.mse_loss(critic_pred, rewards_t)
            total = total + 0.1 * torch.nan_to_num(cl, nan=0.0, posinf=0.0, neginf=0.0)

        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        del outputs2, log_probs
        return (total, None) if return_outputs else total
