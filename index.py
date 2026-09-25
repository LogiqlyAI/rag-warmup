"""Embed every chunk once and store the normalized vectors alongside the chunk file."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMB_PATH = Path("data/processed/embeddings.npy")
META_PATH = Path("data/processed/index_meta.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    # Unit-normalized, so cosine similarity at query time is a plain dot product
    emb = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(EMB_PATH, emb)
    META_PATH.write_text(json.dumps({"model": MODEL_NAME, "n_chunks": len(chunks), "dim": emb.shape[1]}))

    print(f"{len(chunks)} chunks -> {emb.shape} -> {EMB_PATH}")


if __name__ == "__main__":
    main()
