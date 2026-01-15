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
SUBJECT_SWITCH_PENALTY = 0.15  # penalize if main subject changes

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

    doc = nlp(text)
    sigs: List[SentSig] = []

    for sent in doc.sents:
        # lemmas of content words
        lemmas = set(
            t.lemma_.lower()
            for t in sent
            if t.pos_ in CONTENT_POS and not t.is_stop and t.is_alpha
        )

        # named entities
        entities = set(e.text.lower() for e in sent.ents)

        # main grammatical subject lemma
        subj = None
        for t in sent:
            if t.dep_ in {"nsubj", "nsubjpass"} and t.head.pos_ == "VERB":
                subj = t.lemma_.lower()
                break

        s_text = sent.text.strip()
        if not s_text:
            continue

        sigs.append(
            SentSig(
                text=s_text,
                token_count=approx_tokens(s_text),
                lemmas=lemmas,
                entities=entities,
                subject=subj
            )
        )

    return sigs


# similarity

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# chunker

def chunk_document_semantic(nlp, doc: dict) -> List[dict]:
    sigs = sentence_signatures(nlp, doc["text"])
    chunks: List[dict] = []

    current_sents: List[str] = []
    current_tokens = 0
    chunk_lemmas: Set[str] = set()
    chunk_entities: Set[str] = set()
    chunk_subject: Optional[str] = None
    chunk_index = 0

    def flush():
        nonlocal current_sents, current_tokens, chunk_lemmas, chunk_entities, chunk_subject, chunk_index
        if not current_sents:
            return
        chunk_text = " ".join(current_sents).strip()
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_{chunk_index:03d}",
            "doc_id": doc["doc_id"],
            "type": doc["type"],
            "name": doc["name"],
            "source": doc["source"],
            "url": doc["url"],
            "text": chunk_text
        })
        chunk_index += 1
        current_sents = []
        current_tokens = 0
        chunk_lemmas = set()
        chunk_entities = set()
        chunk_subject = None

    for s in sigs:
        # first sentence in a new chunk
        if not current_sents:
            current_sents.append(s.text)
            current_tokens += s.token_count
            chunk_lemmas |= s.lemmas
            chunk_entities |= s.entities
            chunk_subject = s.subject
            continue

        would_exceed = current_tokens + s.token_count > MAX_TOKENS

        # "Belongs" check: needs either enough lemma cohesion OR entity cohesion
        belongs = (jaccard(chunk_lemmas, s.lemmas) >= LEMMA_SIM_THRESHOLD) or (
            jaccard(chunk_entities, s.entities) >= ENTITY_SIM_THRESHOLD
        )

        # split conditions: low cohesion AND chunk is already big enough or hard max size
        if (not belongs and current_tokens >= MIN_TOKENS) or would_exceed:
            flush()

            # start new chunk with this sentence
            current_sents.append(s.text)
            current_tokens += s.token_count
            chunk_lemmas |= s.lemmas
            chunk_entities |= s.entities
            chunk_subject = s.subject
            continue

        # otherwise, keep accumulating
        current_sents.append(s.text)
        current_tokens += s.token_count
        chunk_lemmas |= s.lemmas
        chunk_entities |= s.entities

        # keep the first found subject as "topic anchor"
        if chunk_subject is None and s.subject is not None:
            chunk_subject = s.subject

    flush()
    return chunks


# main loop

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
