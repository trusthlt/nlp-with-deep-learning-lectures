# Code adapted from https://vgel.me/posts/handmade-transformer/

import numpy as np


def softmax(x):
    # Subtract max for numerical stability
    # see https://datascience.stackexchange.com/questions/107767/how-to-prove-softmax-numerical-stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# [m, in], [in, out], [out] -> [m, out]
def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


# [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


# [n_seq, n_embd] -> [n_seq, n_embd]
def causal_self_attention(x: np.ndarray, c_attn: dict, c_proj: dict):
    # qkv projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform causal self attention
    x = attention(q, k, v, causal_mask)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x


# [n_seq, n_embd] -> [n_seq, n_embd]
def transformer_block(x: np.ndarray, attn: dict):
    x = x + causal_self_attention(x, **attn)
    # NOTE: removed ffn
    return x


# [n_seq] -> [n_seq, n_vocab]
def gpt(inputs: np.ndarray, wte: np.ndarray, wpe: np.ndarray, blocks: list[dict[str, dict]]):
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        # x = transformer_block(x, **block)  # [n_seq, n_embd] -> [n_seq, n_embd]
        x = transformer_block(x, block['attn'])  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


N_CTX = 5
N_VOCAB = 2
N_EMBED = 8

Lg = 1024  # Large

MODEL = {
    # EMBEDDING USAGE
    #  P = Position embeddings (one-hot)
    #  T = Token embeddings (one-hot, first is `a`, second is `b`)
    #  V = Prediction scratch space
    #
    #       [P, P, P, P, P, T, T, V]
    "wte": np.array(
        # one-hot token embeddings
        [
            [0, 0, 0, 0, 0, 1, 0, 0],  # token `a` (id 0)
            [0, 0, 0, 0, 0, 0, 1, 0],  # token `b` (id 1)
        ]
    ),
    "wpe": np.array(
        # one-hot position embeddings
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # position 0
            [0, 1, 0, 0, 0, 0, 0, 0],  # position 1
            [0, 0, 1, 0, 0, 0, 0, 0],  # position 2
            [0, 0, 0, 1, 0, 0, 0, 0],  # position 3
            [0, 0, 0, 0, 1, 0, 0, 0],  # position 4
        ]
    ),
    "blocks": [
        {
            "attn": {
                "c_attn": {  # generates qkv matrix
                    "b": np.zeros(N_EMBED * 3),
                    "w": np.array(
                        # this is where the magic happens
                        [
                            [Lg, 0., 0., 0., 0., 0., 0., 0.,  # q
                             1., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [Lg, Lg, 0., 0., 0., 0., 0., 0.,  # q
                             0., 1., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., Lg, Lg, 0., 0., 0., 0., 0.,  # q
                             0., 0., 1., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., Lg, Lg, 0., 0., 0., 0.,  # q
                             0., 0., 0., 1., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., 0., Lg, Lg, 0., 0., 0.,  # q
                             0., 0., 0., 0., 1., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 1.],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., -1],  # v
                            [0., 0., 0., 0., 0., 0., 0., 0.,  # q
                             0., 0., 0., 0., 0., 0., 0., 0.,  # k
                             0., 0., 0., 0., 0., 0., 0., 0.],  # v
                        ]
                    ),
                },
                "c_proj": {  # weights to project attn result back to embedding space
                    "b": [0, 0, 0, 0, 0, Lg, 0, 0],
                    "w": np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, -Lg, Lg, 0],
                        ]
                    ),
                },
            },
        }
    ],
}

CHARS = ["a", "b"]


def tokenize(s: str) -> list[int]:
    return [CHARS.index(c) for c in s]


def untok(tok: int) -> str:
    return CHARS[tok]


def predict(s: str):
    tokens = tokenize(s)[-5:]
    logits = gpt(np.array(tokens), **MODEL)
    probs = softmax(logits)

    for idx, tok in enumerate(tokens):
        pred: int = np.argmax(probs[idx]).item()
        print(
            f"{untok(tok)} ({tok}): next={untok(pred)} ({pred}) probs={probs[idx]} logits={logits[idx]}"
        )

    return np.argmax(probs[-1])


def complete(s, max_new_tokens=10):
    tokens = tokenize(s)
    while len(tokens) < len(s) + max_new_tokens:
        logits = gpt(np.array(tokens[-5:]), **MODEL)
        probs = softmax(logits)
        pred = np.argmax(probs[-1])
        tokens.append(pred)
    return s + " :: " + "".join(untok(t) for t in tokens[len(s):])


if __name__ == '__main__1':
    s = "aabaa"
    tokens = tokenize(s)[-5:]
    x = MODEL['wte'][tokens] + MODEL['wpe'][range(len(tokens))]  # [n_seq] -> [n_seq, n_embd]
    print(x)

if __name__ == '__main__2':
    s = "aabaa"
    tokens = tokenize(s)[-5:]
    x = MODEL['wte'][tokens] + MODEL['wpe'][range(len(tokens))]  # [n_seq] -> [n_seq, n_embd]
    print(x)

    # qkv projections
    qkv_projections = linear(x, **MODEL['blocks'][0]['attn']['c_attn'])  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    with np.printoptions(linewidth=150):
        print(qkv_projections.astype(int))  # print as integers

    # split into qkv
    q, k, v = np.split(qkv_projections, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]
    with np.printoptions(linewidth=150):
        print("Q:")
        print(q.astype(int))
        print("K:")
        print(k.astype(int))
        print("V:")
        print(v.astype(int))

if __name__ == '__main__3':
    s = "aabaa"
    tokens = tokenize(s)[-5:]
    x = MODEL['wte'][tokens] + MODEL['wpe'][range(len(tokens))]  # [n_seq] -> [n_seq, n_embd]
    print(x)

    # qkv projections
    qkv_projections = linear(x, **MODEL['blocks'][0]['attn']['c_attn'])  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(qkv_projections, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    with np.printoptions(linewidth=150, suppress=True, precision=3):
        # causal mask to hide future inputs from being attended to
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
        print("Mask:")
        print(causal_mask)

        # perform causal self attention
        print("q @ k.T")
        print(q @ k.T)

        print("q @ k.T / np.sqrt(q.shape[-1])")
        print(q @ k.T / np.sqrt(q.shape[-1]))

        print("q @ k.T / np.sqrt(q.shape[-1]) + causal_mask")
        print(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask)

        print("softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask)")
        print(softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask))

        print("softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask) @ v")
        print(softmax(q @ k.T / np.sqrt(q.shape[-1]) + causal_mask) @ v)

if __name__ == '__main__':
    test = "aab" * 10
    total, correct = 0, 0
    for i in range(2, len(test) - 1):
        ctx = test[:i]
        expected = test[i]
        total += 1
        if untok(predict(ctx)) == expected:
            correct += 1
    print(f"ACCURACY: {correct / total * 100}% ({correct} / {total})")
