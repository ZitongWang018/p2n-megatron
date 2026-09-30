"""Deterministic batches from a uint16 stream or ordered NumPy token shards."""

from __future__ import annotations

from pathlib import Path
from bisect import bisect_right

import numpy as np
import torch


class TokenStream:
    def __init__(self, path: str, *, vocab_size: int, allow_wrap: bool = False):
        file = Path(path)
        if file.is_file() and file.suffix == ".bin":
            if file.stat().st_size % np.dtype("uint16").itemsize:
                raise ValueError("uint16 token file has an odd number of bytes")
            self.shards = [np.memmap(file, dtype=np.uint16, mode="r")]
        elif file.is_dir():
            paths = sorted(file.glob("input_ids_2048_*.npy"))
            if not paths:
                raise ValueError(f"no input_ids_2048_*.npy shards in {file}")
            self.shards = []
            for shard_path in paths:
                shard = np.load(shard_path, mmap_mode="r")
                if shard.ndim != 2 or shard.shape[1] != 2048 or shard.dtype not in (
                    np.dtype("uint16"), np.dtype("int32"), np.dtype("int64")
                ):
                    raise ValueError(f"unexpected token shard layout: {shard_path}")
                self.shards.append(shard.reshape(-1))
        else:
            raise ValueError(f"expected a uint16 .bin file or NumPy shard directory: {file}")
        self.offsets = [0]
        for shard in self.shards:
            self.offsets.append(self.offsets[-1] + len(shard))
        self.vocab_size = vocab_size
        self.allow_wrap = allow_wrap
        if len(self) < 2:
            raise ValueError("token file is too short")

    def __len__(self):
        return self.offsets[-1]

    def _read(self, start: int, stop: int):
        if stop > len(self):
            if not self.allow_wrap:
                raise RuntimeError(
                    f"token stream exhausted at {stop:,}; file has {len(self):,} tokens. "
                    "Supply the full Pile stream or explicitly allow wrapping for a non-paper run."
                )
            indices = np.arange(start, stop, dtype=np.int64) % len(self)
            return np.array([self._read(int(i), int(i) + 1)[0] for i in indices], dtype=np.int64)
        chunks = []
        while start < stop:
            shard_index = bisect_right(self.offsets, start) - 1
            end = min(stop, self.offsets[shard_index + 1])
            chunks.append(np.asarray(
                self.shards[shard_index][start - self.offsets[shard_index]:end - self.offsets[shard_index]],
                dtype=np.int64,
            ))
            start = end
        return np.concatenate(chunks) if len(chunks) > 1 else np.array(chunks[0], copy=True)

    def batch(self, *, first_token: int, batch_size: int, seq_len: int, device: torch.device):
        # Each sample needs one label beyond its input tokens.
        rows = []
        for sample in range(batch_size):
            start = first_token + sample * seq_len
            stop = start + seq_len + 1
            rows.append(self._read(start, stop))
        tokens = torch.tensor(np.stack(rows), device=device, dtype=torch.long)
        if int(tokens.max()) >= self.vocab_size:
            raise ValueError(
                f"token id {int(tokens.max())} exceeds configured vocabulary {self.vocab_size}; "
                "check the tokenizer used to build this dataset"
            )
        return tokens[:, :-1], tokens[:, 1:]


def synthetic_batch(*, batch_size: int, seq_len: int, vocab_size: int, seed: int, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=generator)
    tokens = tokens.to(device)
    return tokens[:, :-1], tokens[:, 1:]
