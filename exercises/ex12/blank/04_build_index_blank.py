"""
Goal:
You will complete a script that turns your text chunks into a searchable vector database.

You will implement/fill in the missing parts of:
- building the embeddings with a sentence transformer
- building the FAISS index
- running a test query

------------------------------------------------------------
How Sentence Embeddings Work
------------------------------------------------------------
A sentence embedding model converts a piece of text into a fixed-size vector like:

    "Demacia is a proud kingdom..."  -->  [0.12, -0.03, 0.88, ..., 0.04]

So:
- Similar texts produce vectors that point in a similar "direction" in vector space
- That makes it possible to search by meaning
- `normalize_embeddings=True` makes vectors have length 1, which makes cosine similarity easy to compute

------------------------------------------------------------
How FAISS Works
------------------------------------------------------------
FAISS is a library for fast similarity search between vectors.

We store all chunk embeddings in an index.

Then for a query:
- embed the query into a vector
- search the index for the nearest vectors
- retrieve the most similar chunks

Here we use:
    IndexFlatIP(dim)

Meaning:
- "Flat"  => brute force scan (exact search, not approximate)
- "IP"    => inner product similarity

If vectors are normalized:
    inner_product(a, b) == cosine_similarity(a, b)

So this gives cosine similarity search efficiently.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# config

CHUNKS_FILE = "lol_universe_chunks.json"
INDEX_FILE = "lol_universe.index"
METADATA_FILE = "lol_universe_chunks_meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# json utils

def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Your Tasks Start Here

def main():
    print("Loading chunks...")
    chunks = load_json(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # TODO:
    # Extract the raw text field from each chunk into a list of strings called `texts`.
    texts = []

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    # TODO:
    # Compute embeddings for all texts (hint: use model.encode())
    # Requirements:
    # - normalize_embeddings=True
    embeddings = None

    # TODO:
    # Convert the embeddings to a float32 numpy array.
    embeddings = None

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("\nBuilding FAISS index...")
    # TODO:
    # Create a FAISS index using IndexFlatIP(dim)
    index = None

    # TODO:
    # Add all embeddings into the index
    # After this, index.ntotal should equal len(chunks)
    pass

    print(f"FAISS index size: {index.ntotal}")

    print("\nSaving index and metadata...")
    # TODO:
    # Save the FAISS index to disk and also save the metadata JSON.
    pass

    print("\nBuilding Index complete")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

    # we test using a quary here

    query = "Who is the ruler of Demacia?"
    print(f"\nTest query: {query}")

    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.asarray(q_emb, dtype="float32"), k=3)

    print("\nTop results:")
    for rank, idx in enumerate(indices[0], start=1):
        chunk = chunks[idx]
        print(f"\n#{rank} | score={scores[0][rank-1]:.4f}")
        print(f"{chunk['chunk_id']} ({chunk['type']})")
        print(chunk["text"][:300], "...")


if __name__ == "__main__":
    main()
