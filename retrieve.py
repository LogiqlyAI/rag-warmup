"""Embed a query and return the top-k chunks by cosine similarity."""

import json
import sys
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

from index import EMB_PATH, META_PATH, MODEL_NAME, load_chunks

K = 5

# bge models expect this instruction on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@cache
def load_index() -> tuple[list[dict], np.ndarray]:
    """Load chunks and their vectors, refusing a stale or mismatched index."""
    chunks = load_chunks()
    emb = np.load(EMB_PATH)
    meta = json.loads(META_PATH.read_text())
    if meta["model"] != MODEL_NAME or emb.shape[0] != len(chunks):
        raise RuntimeError(
            f"Index is stale ({meta['model']}, {emb.shape[0]} rows vs {len(chunks)} chunks); rerun index.py"
        )
    return chunks, emb


def retrieve(query: str, k: int = K) -> list[dict]:
    chunks, emb = load_index()
    q = _model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    scores = emb @ q  # both sides unit-normalized -> cosine similarity
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def main() -> None:
    query = " ".join(sys.argv[1:])
    if not query:
        sys.exit("usage: retrieve.py <query>")
    for rank, hit in enumerate(retrieve(query), start=1):
        preview = hit["text"][:200].replace("\n", " ")
        print(f"{rank}. {hit['score']:.3f}  {hit['chunk_id']}  (p{hit['page']})  {preview}")


if __name__ == "__main__":
    main()
