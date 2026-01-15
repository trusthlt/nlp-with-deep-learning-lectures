import json
from pathlib import Path

CHAMPIONS_FILE = "lol_universe_champion_lore.json"
REGIONS_FILE = "lol_universe_region_lore.json"
OUTPUT_FILE = "lol_universe_corpus.json"


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_champion(doc):
    return {
        "doc_id": f"champion_{doc['slug']}",
        "type": "champion",
        "name": doc["name"],
        "slug": doc["slug"],
        "url": doc["url"],
        "source": "universe",
        "text": doc["text"]
    }


def normalize_region(doc):
    return {
        "doc_id": f"region_{doc['slug']}",
        "type": "region",
        "name": doc["name"],
        "slug": doc["slug"],
        "url": doc["url"],
        "source": "universe",
        "text": doc["text"]
    }


def main():
    print("Loading input files...")

    champions = load_json(CHAMPIONS_FILE)
    regions = load_json(REGIONS_FILE)

    print(f"Loaded {len(champions)} champions")
    print(f"Loaded {len(regions)} regions")

    corpus = []

    print("\nNormalizing champions...")
    for doc in champions:
        corpus.append(normalize_champion(doc))

    print("\nNormalizing regions...")
    for doc in regions:
        corpus.append(normalize_region(doc))

    print("\nRunning sanity checks...")

    doc_ids = [d["doc_id"] for d in corpus]

    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate doc_id detected!")

    required_keys = {
        "doc_id", "type", "name", "slug", "url", "source", "text"
    }

    for d in corpus:
        if set(d.keys()) != required_keys:
            raise ValueError(f"Schema mismatch in document: {d['doc_id']}")

        if not d["text"] or len(d["text"]) < 50:
            print(f"Warning: very short text in {d['doc_id']}")

    print("All checks passed")


    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("\nNormalization complete")
    print(f"Saved {len(corpus)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
