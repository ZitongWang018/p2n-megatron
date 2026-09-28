"""Fetch a small public uint16 token sample for the four-GPU quickstart."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--eod-id", type=int, default=50256)
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        parser.error("rows must be between 1 and 100")
    if not 0 <= args.eod_id < 50304:
        parser.error("eod-id must be in the model vocabulary")

    query = urllib.parse.urlencode({
        "dataset": "hyq718/uint16smallpile", "config": "default",
        "split": "train", "offset": 0, "length": args.rows,
    })
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "p2n-megatron/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (TimeoutError, urllib.error.URLError) as error:
        raise RuntimeError(
            "Could not fetch the public token sample. Set P2N_TRAIN_BIN to a local uint16 .bin file."
        ) from error
    rows = payload.get("rows", [])
    if len(rows) != args.rows:
        raise RuntimeError(f"expected {args.rows} rows, received {len(rows)}")

    tokens = []
    for entry in rows:
        if entry.get("truncated_cells"):
            raise RuntimeError("dataset API truncated a token row")
        row = entry["row"]
        ids = row["input_ids"]
        mask = row.get("attention_mask")
        if mask is not None:
            ids = [token for token, keep in zip(ids, mask) if keep]
        if not ids or min(ids) < 0 or max(ids) >= 50304:
            raise ValueError("dataset contains empty or out-of-vocabulary token row")
        tokens.extend(ids)
        tokens.append(args.eod_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    np.asarray(tokens, dtype=np.uint16).tofile(temporary)
    os.replace(temporary, args.output)
    print(f"Prepared {len(tokens):,} tokens in {args.output}")


if __name__ == "__main__":
    main()
