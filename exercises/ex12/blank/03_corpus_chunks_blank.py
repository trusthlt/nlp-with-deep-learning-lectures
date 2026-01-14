"""
3. Chunking the data

Goal:
You will implement 3 key functions to semantically chunk documents into coherent blocks:
1) sentence_signatures
2) jaccard
3) chunk_document_semantic

You are given the skeleton code. Your job is to fill in the missing parts marked with TODO.

Expected output:
- The script loads `lol_universe_corpus.json`
- It produces chunked documents in `lol_universe_chunks.json`

Focus:
- sentence-level feature extraction (lemmas, entities, subject)
- Jaccard similarity for overlap
- chunking based on cohesion + size constraints
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Optional

import spacy

# config

INPUT_FILE = "lol_universe_corpus.json"
OUTPUT_FILE = "lol_universe_chunks.json"

# chunk size constraints
MIN_TOKENS = 120
MAX_TOKENS = 420

# similarity thresholds
LEMMA_SIM_THRESHOLD = 0.22   # content-lemma overlap
ENTITY_SIM_THRESHOLD = 0.10  # entity overlap

SPACY_MODEL = "en_core_web_sm"


# json utils

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def approx_tokens(text: str) -> int:
    # simple token estimation (good enough for chunk sizing)
    return len(text.split())


# sentence signature extraction

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


@dataclass
class SentSig:
    text: str
    token_count: int
    lemmas: Set[str]
    entities: Set[str]
    subject: Optional[str]


def sentence_signatures(nlp, text: str) -> List[SentSig]:
    """
    TODO: Implement sentence-level signature extraction.

    For each sentence in the text:
    - collect the cleaned sentence text
    - estimate its token count
    - extract a set of content-word lemmas (lowercased)
      * keep only tokens where:
        - token.pos_ is in CONTENT_POS
        - token is alphabetic
        - token is NOT a stopword
    - extract a set of named entities in the sentence (lowercased)
    - attempt to extract the main grammatical subject:
      * first token where:
        - token.dep_ is 'nsubj' or 'nsubjpass'
        - token.head.pos_ == 'VERB'
      * store token.lemma_.lower()
      * if none exists, subject = None

    Return a list[SentSig].
    """

    # TODO: parse the full text with spacy
    doc = None 

    sigs: List[SentSig] = []

    # TODO: iterate through doc.sents
    for sent in []:
        # TODO: create lemma set for content tokens
        lemmas: Set[str] = set()

        # TODO: create entity set from sent.ents
        entities: Set[str] = set()

        # TODO: find the main subject lemma (nsubj / nsubjpass)
        subj: Optional[str] = None

        # TODO: get sentence text and skip empty ones
        s_text = ""
        if not s_text:
            continue

        # TODO: append SentSig object
        sigs.append(
            SentSig(
                text=s_text,
                token_count=approx_tokens(s_text),
                lemmas=lemmas,
                entities=entities,
                subject=subj,
            )
        )

    return sigs


# similarity

def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    TODO: Implement Jaccard similarity.

    Jaccard(a, b) = |a ∩ b| / |a ∪ b|

    Requirements:
    - if either set is empty, return 0.0
    - return a float in [0, 1]
    """

    # TODO: edge case handling
    if False:
        return 0.0

    # TODO: compute intersection and union sizes
    intersection_size = 0
    union_size = 1

    return intersection_size / union_size


# =========================
# chunker
# =========================

def chunk_document_semantic(nlp, doc: dict) -> List[dict]:
    """
    TODO: Implement semantic chunking.

    You receive one document in this schema:
      {
        "doc_id": ...,
        "type": ...,
        "name": ...,
        "source": ...,
        "url": ...,
        "text": ...
      }

    Steps:
    1) Build sentence signatures from doc["text"]
    2) Iterate sentence-by-sentence and accumulate them into chunks.
    3) Decide whether a sentence "belongs" to the current chunk using:
         - lemma overlap (Jaccard >= LEMMA_SIM_THRESHOLD)
           OR
         - entity overlap (Jaccard >= ENTITY_SIM_THRESHOLD)
    4) Split (flush) when:
         - chunk is already big enough (>= MIN_TOKENS) AND the new sentence does NOT belong
         OR
         - adding the sentence would exceed MAX_TOKENS
    5) Every chunk must contain:
         chunk_id, doc_id, type, name, source, url, text
       where:
         chunk_id = f"{doc['doc_id']}_{chunk_index:03d}"
    """

    # TODO: generate sentence signatures
    sigs = []

    chunks: List[dict] = []

    current_sents: List[str] = []
    current_tokens = 0

    chunk_lemmas: Set[str] = set()
    chunk_entities: Set[str] = set()
    chunk_subject: Optional[str] = None

    chunk_index = 0

    def flush():
        """
        Writes the current chunk into `chunks` and resets the buffer.

        TODO: Keep this behavior:
        - if current_sents is empty -> do nothing
        - join sentences with spaces
        - reset all chunk accumulators
        """
        nonlocal current_sents, current_tokens, chunk_lemmas, chunk_entities, chunk_subject, chunk_index

        # TODO: do nothing if there is nothing to flush
        if False:
            return

        chunk_text = " ".join(current_sents).strip()

        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_{chunk_index:03d}",
                "doc_id": doc["doc_id"],
                "type": doc["type"],
                "name": doc["name"],
                "source": doc["source"],
                "url": doc["url"],
                "text": chunk_text,
            }
        )

        chunk_index += 1
        current_sents = []
        current_tokens = 0
        chunk_lemmas = set()
        chunk_entities = set()
        chunk_subject = None

    # TODO: iterate over sentence signatures
    for s in sigs:
        # TODO: if this is the first sentence in a fresh chunk, initialize chunk state
        if False:
            pass
            continue

        # TODO: check if adding the sentence would exceed MAX_TOKENS
        would_exceed = False

        # TODO: compute "belongs" using lemma overlap OR entity overlap
        lemma_cohesion = 0.0
        entity_cohesion = 0.0

        belongs = False

        # TODO: split conditions:
        # - (not belongs AND current_tokens >= MIN_TOKENS) OR would_exceed
        if False:
            flush()

            # TODO: start new chunk with this sentence
            continue

        # TODO: otherwise keep accumulating sentence into current chunk
        # - append sentence text
        # - add token count
        # - update lemma/entity sets
        # - keep the first non-empty subject as topic anchor (optional)
        pass

    # TODO: flush final leftover chunk
    flush()

    return chunks


# main loop (already complete)

def main():
    print("Loading spacy...")
    nlp = spacy.load(SPACY_MODEL)

    print("Loading corpus...")
    corpus = load_json(INPUT_FILE)
    print(f"Loaded {len(corpus)} documents")

    all_chunks: List[dict] = []

    print("\nSemantic chunking...")
    for d in corpus:
        chunks = chunk_document_semantic(nlp, d)
        all_chunks.extend(chunks)
        print(f"{d['doc_id']} → {len(chunks)} chunks")

    # basic sanity checks
    ids = [c["chunk_id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id detected!")

    print("\nSaving chunks...")
    save_json(all_chunks, OUTPUT_FILE)

    print("\nDone")
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
