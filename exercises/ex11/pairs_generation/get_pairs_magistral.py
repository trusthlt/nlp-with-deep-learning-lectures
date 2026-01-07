import os
import json
import random
import time
import torch

from transformers import (
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
)

# to approximate determinism, this will rarely work in data generation tasks that require creative outputs tho

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False


# usually commercial LLMs have different loading modes, we took this from the magistral small documentation in HF

MODEL_ID = "mistralai/Magistral-Small-2509"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    tokenizer_type="mistral",
)

model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# a json parser, we will need later

def try_parse_json(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        return None


# a generation helper for magistral

def magistral_generate(messages, max_new_tokens=512, temperature=0.0):
    tokenized = tokenizer.apply_chat_template(
        messages,
        return_dict=True,
    )

    input_ids = torch.tensor(
        tokenized["input_ids"], device=model.device
    ).unsqueeze(0)

    attention_mask = torch.tensor(
        tokenized["attention_mask"], device=model.device
    ).unsqueeze(0)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
        )

    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# output path

output_path = "genz_pairs_magistral_diverse_5k.jsonl"
existing_originals = set()

if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                existing_originals.add(obj.get("original", "").strip().lower())
            except Exception:
                continue

print(f"Loaded {len(existing_originals)} existing pairs")


# this is our seed pair: we use this to kinda guide the model into the generation direction we want

seed_pair = {
    "original": "I am really tired after studying all night.",
    "gen_z": "Been grinding all night and I’m lowkey exhausted fr."
}

if seed_pair["original"].lower() not in existing_originals:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(seed_pair, ensure_ascii=False) + "\n")
    existing_originals.add(seed_pair["original"].lower())
    print("Seed pair written to dataset")


# the model will probably generate some duplicates, even when we feed already generated outputs, so we will have to clean those duplicates

def normalize(text):
    return (
        text.strip()
            .lower()
            .replace(",", "")
            .replace(".", "")
            .replace("!", "")
            .replace("?", "")
    )

def is_duplicate(text, seen):
    n = normalize(text)
    if n in seen:
        return True

    for s in seen:
        if n[:60] in s or s[:60] in n:
            return True

    return False


# these are some rules and modes we will be feeding magistral to get some diverse pairs

LENGTH_MODES = [
    "a short informal sentence",
    "a normal conversational sentence",
    "a long detailed sentence",
    "two sentences describing a situation",
    "a multi-sentence narrative paragraph",
]

CONTEXT_MODES = [
    "university or studying",
    "work, career or productivity",
    "friendships or relationships",
    "mental health or emotions",
    "gaming and online culture",
    "travel, commuting or daily life",
    "social plans or weekends",
    "technology and internet habits",
]

GEN_Z_STYLE_RULES = """
The Gen-Z rewrite must:
- keep the SAME meaning
- sound natural (not exaggerated)
- use slang only where it fits
- never change facts or intent
"""

BASE_SYSTEM_PROMPT = """
You generate synthetic style-transfer training data.

Return ONLY valid JSON in the form:
{
  "original": "...",
  "gen_z": "..."
}

The Gen-Z version must preserve meaning.
Do not explain. Do not add commentary.
No markdown. No emojis.
""".strip()


# prompt builder

def build_genz_prompt(avoid_examples):

    length_mode = random.choice(LENGTH_MODES)
    context_mode = random.choice(CONTEXT_MODES)

    avoid_block = ""
    if avoid_examples:
        avoid_block = (
            "\nAvoid producing anything similar to:\n"
            + "\n".join(f"- {x}" for x in avoid_examples)
        )

    user_prompt = (
        f"Generate one training pair.\n"
        f"The ORIGINAL text must be {length_mode}.\n"
        f"The topic should relate to {context_mode}.\n"
        f"{GEN_Z_STYLE_RULES}\n"
        f"{avoid_block}\n\n"
        f"Example format:\n"
        f"{json.dumps(seed_pair, ensure_ascii=False)}\n\n"
        f"Now produce one NEW unique pair:"
    )

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# generation config

TARGET_PAIRS = 5000
saved_count = 0

TEMPERATURE = 0.4
MAX_TOKENS = 400


# main generation loop

with open(output_path, "a", encoding="utf-8") as f:

    while saved_count < TARGET_PAIRS:

        avoid_subset = random.sample(
            list(existing_originals),
            k=min(20, len(existing_originals))
        ) if existing_originals else []

        messages = build_genz_prompt(avoid_subset)

        raw = magistral_generate(
            messages,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        parsed = try_parse_json(raw)

        if not parsed or "original" not in parsed or "gen_z" not in parsed:
            print(f"[{saved_count}] skipped (invalid JSON)")
            print(">>", raw[:120])
            continue

        original = parsed["original"].strip()

        if len(original) < 6:
            print(f"[{saved_count}] skipped (too short)")
            continue

        if is_duplicate(original, existing_originals):
            print(f"[{saved_count}] duplicate:", original[:70])
            continue

        f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

        existing_originals.add(normalize(original))
        saved_count += 1

        print(f"[{saved_count}/{TARGET_PAIRS}] saved:", original[:90])
        time.sleep(0.1)

print("\nDone. Added", saved_count, "new Gen-Z style pairs.")
