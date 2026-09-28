"""Deterministic contiguous batches from Megatron's uint16 indexed .bin payload.

The paired .idx remains available for audit with Megatron IndexedDataset. This
reader intentionally uses the contiguous .bin token payload so every global
batch has an exact, non-overlapping token offset on four data-parallel ranks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, path: str, *, vocab_size: int, allow_wrap: bool = False):
        file = Path(path)
        if file.suffix != ".bin" or not file.is_file():
            raise ValueError(f"expected an existing Megatron .bin file: {file}")
        if file.stat().st_size % np.dtype("uint16").itemsize:
            raise ValueError("uint16 token file has an odd number of bytes")
        self.tokens = np.memmap(file, dtype=np.uint16, mode="r")
        self.vocab_size = vocab_size
        self.allow_wrap = allow_wrap
        if len(self.tokens) < 2:
            raise ValueError("token file is too short")

    def batch(self, *, first_token: int, batch_size: int, seq_len: int, device: torch.device):
        # Each sample needs one label beyond its input tokens.
        rows = []
        for sample in range(batch_size):
            start = first_token + sample * seq_len
            stop = start + seq_len + 1
            if stop <= len(self.tokens):
                row = np.array(self.tokens[start:stop], dtype=np.int64)
            elif self.allow_wrap:
                indices = np.arange(start, stop, dtype=np.int64) % len(self.tokens)
                row = np.asarray(self.tokens[indices], dtype=np.int64)
            else:
                raise RuntimeError(
                    f"token stream exhausted at {stop:,}; file has {len(self.tokens):,} tokens. "
                    "Supply the full Pile stream or explicitly allow wrapping for a non-paper run."
                )
            rows.append(row)
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
