import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = "lol_universe_chunks.json"
INDEX_FILE = "lol_universe.index"
METADATA_FILE = "lol_universe_chunks_meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("Loading chunks...")
    chunks = load_json(CHUNKS_FILE)
    print(f"✔ Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal}")

    print("\nSaving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    save_json(chunks, METADATA_FILE)

    print("\nBuilding Index complete")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

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
