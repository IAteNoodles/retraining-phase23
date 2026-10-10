"""Token-budget bucket batching.

Micro-batches are packed by token count (not sample count): similar-length
rows are grouped, then indices are accumulated until the token budget is hit.
A second layer — dynamic gradient accumulation — sizes the accumulation steps
so each OPTIMIZER step sees ~target_step_tokens.

Why two layers: a single 60K-token forward does NOT fit on 5 GB VRAM. The
262K vocabulary makes LM-head logits ~1 MB per target token (x2 in backward),
so ~60K tokens in one forward needs ~10 GB+ just for logits. Instead:
micro-budget (~4K tokens/forward, fits) x grad_accum (~14) ~= 56K/step.
Same gradient math, fits in memory.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def sample_lengths(dataset, idx: int) -> tuple[int, int]:
    """(input_tokens, target_tokens) for one row."""
    try:
        item = dataset[idx]
        if isinstance(item, dict):
            il = item.get("input_length")
            tl = item.get("target_length")
            if il is not None and tl is not None:
                to_int = lambda v: int(v.item() if hasattr(v, "item") else v)
                return to_int(il), to_int(tl)
    except Exception:
        pass
    return sample_length(dataset, idx), 0


def sample_length(dataset, idx: int) -> int:
    try:
        item = dataset[idx]
        length = item.get("length") if isinstance(item, dict) else None
        if length is not None:
            return int(length.item() if hasattr(length, "item") else length)
        return int(len(item.get("input_ids") or []) + len(item.get("labels") or []))
    except Exception:
        return 1


class TokenBucketBatchSampler(Sampler):
    """Pack micro-batches by TWO budgets: total tokens AND target tokens.

    Target tokens are the binding constraint (LM-head logits over a 262K
    vocab cost ~4 MB/token in fwd+bwd on fp16; measured on RTX 4050 6 GB:
    ~800 target tokens/forward fits with headroom, batch=4/tgt=320 OOMs).
    Input length barely matters (sliding-window attention + checkpointing:
    batch=2 holds in_len 512..4096 for +0.7 GB).
    """

    def __init__(self, dataset, max_batch_size: int, max_tokens: int,
                 max_target_tokens: int = 800, bucket_size: int = 256,
                 shuffle: bool = True, seed: int = 42,
                 drop_last: bool = False):
        self.dataset = dataset
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_tokens = max(1, int(max_tokens))
        self.max_target_tokens = max(1, int(max_target_tokens))
        self.bucket_size = max(self.max_batch_size, int(bucket_size))
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.lengths = [sample_length(dataset, i) for i in range(len(dataset))]
        self.target_lengths = [sample_lengths(dataset, i)[1] for i in range(len(dataset))]
        self._iter_count = 0    # Trainer never calls set_epoch on a
        self._built_count = None  # batch_sampler; advance per __iter__ instead

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _build_batches(self, epoch: int | None = None):
        ep = self.epoch if epoch is None else epoch
        rng = np.random.default_rng(self.seed + ep)
        indices = sorted(range(len(self.dataset)), key=lambda i: self.lengths[i])
        if self.shuffle and len(indices) > 1:
            # random cyclic shift: pool MEMBERSHIP varies per epoch while
            # every pool stays a contiguous (similar-length) slice
            shift = int(rng.integers(0, len(indices)))
            indices = indices[shift:] + indices[:shift]
        pools = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle:
            rng.shuffle(pools)
        batches = []
        for pool in pools:
            # pack in SORTED order (no inner shuffle): minimal padding, and
            # the fattest rows never ambush a micro-batch mid-epoch
            batch, batch_tokens, batch_tgt = [], 0, 0
            for idx in sorted(pool, key=lambda i: self.lengths[i]):
                sample_tokens = max(1, self.lengths[idx])
                sample_tgt = max(0, self.target_lengths[idx])
                if batch and (len(batch) >= self.max_batch_size
                              or batch_tokens + sample_tokens > self.max_tokens
                              or batch_tgt + sample_tgt > self.max_target_tokens):
                    if len(batch) == self.max_batch_size or not self.drop_last:
                        batches.append(batch)
                    batch, batch_tokens, batch_tgt = [], 0, 0
                batch.append(idx)
                batch_tokens += sample_tokens
                batch_tgt += sample_tgt
            if batch and (len(batch) == self.max_batch_size or not self.drop_last):
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_batches(self.epoch + self._iter_count)
        self._built_count = len(batches)
        self._iter_count += 1
        return iter(batches)

    def __len__(self):
        if self._built_count is not None:
            return self._built_count
        return len(self._build_batches(self.epoch))


class BucketDataloaderMixin:
    """Override get_train_dataloader to use token-budget packing.

    Set `trainer.bucket_cfg = dict(max_batch_size, max_tokens, seed)` after
    construction; the mixin reads it when the dataloader is built.
    """
    bucket_cfg: dict | None = None

    def get_train_dataloader(self):
        if not self.bucket_cfg:
            return super().get_train_dataloader()
        cfg = self.bucket_cfg
        sampler = TokenBucketBatchSampler(
            self.train_dataset,
            max_batch_size=cfg.get("max_batch_size", 32),
            max_tokens=cfg.get("max_tokens", 6000),
            max_target_tokens=cfg.get("max_target_tokens", 500),
            bucket_size=cfg.get("bucket_size", 256),
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last)
        workers = int(cfg.get("num_workers", 0))
        loader = DataLoader(self.train_dataset, batch_sampler=sampler,
                            collate_fn=self.data_collator,
                            num_workers=workers,
                            pin_memory=workers > 0)
        try:
            return self.accelerator.prepare(loader)
        except Exception:
            return loader


def grad_accum_for_budget(target_step_tokens: int, micro_token_budget: int,
                          packing: float = 0.85) -> int:
    """Micro-batches per optimizer step. packing = measured fill ratio of the
    micro budget (bucketed packing typically hits ~85-93%)."""
    return max(1, int(round(target_step_tokens / max(1, micro_token_budget * packing))))
