import json
import faiss
import numpy as np
import torch
import threading
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

INDEX_FILE = "lol_universe.index"
METADATA_FILE = "lol_universe_chunks_meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

TOP_K = 5
MAX_NEW_TOKENS = 512


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(chunks, question):
    context = "\n\n".join(
        f"[{c['chunk_id']} | {c['type']} | {c['name']}]\n{c['text']}"
        for c in chunks
    )

    return f"""SYSTEM (STRICT, RAG, ULTRA-SHORT, HARD-STOP):

You are a chronicler of Runeterra, writing from within the world.
You recount only recorded history, testimony, or legend.

OUTPUT RULES (ABSOLUTE):
- Output ONLY the answer text.
- No headers, no labels, no extra formatting.
- Max 5 sentences total.
- Never repeat a fact.
- Never restate uncertainty twice.

GROUNDING (RAG ENFORCED):
- Every sentence MUST be explicitly supported by the context.
- Do NOT infer motives, goals, or hierarchy unless explicitly stated.

HARD EXIT CONDITIONS:
- If the context does not contain the answer, output exactly:
Not in the records.
- If the context states uncertainty OR lacks detail, output exactly one final sentence:
Records are uncertain.
and STOP immediately after it (no extra words).

STOPPING:
- Stop immediately after the final sentence.
- Do not continue writing for any reason.

CONTEXT:
{context}

QUESTION:
{question}
"""

def main():
    print("Loading index and metadata...")
    index = faiss.read_index(INDEX_FILE)
    chunks = load_json(METADATA_FILE)
    print(f"Loaded {len(chunks)} chunks")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("\nReady. Type a question (or 'quit').\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break

        # retrieval
        q_emb = embedder.encode([question], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, indices = index.search(q_emb, TOP_K)
        retrieved_chunks = [chunks[i] for i in indices[0]]

        print("\nRetrieved chunks:")
        for i, c in enumerate(retrieved_chunks, 1):
            print(
                f"{i}. {c['chunk_id']} "
                f"({c['type']} | {c['name']} | score: {scores[0][i-1]:.4f})"
            )

        prompt = build_prompt(retrieved_chunks, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            streamer=streamer
        )

        print("\nChronicle:\n")

        thread = threading.Thread(
            target=model.generate,
            kwargs=generation_kwargs
        )
        thread.start()

        for token in streamer:
            print(token, end="", flush=True)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
