import os
from collections import Counter, defaultdict

import regex as re

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


def _pretokenize_chunk(text: str, counts: Counter) -> None:
    for m in _PAT_RE.finditer(text):
        counts[m.group()] += 1


def _pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunks: int,
) -> Counter:
    counts: Counter = Counter()
    split_re = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None
    with open(input_path, "rb") as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
        else:
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pieces = split_re.split(chunk) if split_re is not None else [chunk]
            for piece in pieces:
                _pretokenize_chunk(piece, counts)
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 32,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_counts = _pretokenize_file(input_path, special_tokens, num_chunks)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # Represent each unique word as a list of byte-tokens, paired with its count.
    word_tokens: list[list[bytes]] = []
    word_freq: list[int] = []
    for word, count in word_counts.items():
        word_bytes = word.encode("utf-8")
        if len(word_bytes) < 2:
            continue
        word_tokens.append([bytes([b]) for b in word_bytes])
        word_freq.append(count)

    pair_counts: Counter = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for i, tokens in enumerate(word_tokens):
        c = word_freq[i]
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pair_counts[pair] += c
            pair_to_words[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        next_id += 1
        merges.append(best_pair)

        affected = list(pair_to_words[best_pair])
        for i in affected:
            tokens = word_tokens[i]
            c = word_freq[i]

            for k in range(len(tokens) - 1):
                old_pair = (tokens[k], tokens[k + 1])
                pair_counts[old_pair] -= c
                pair_to_words[old_pair].discard(i)
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                    pair_to_words.pop(old_pair, None)

            new_tokens: list[bytes] = []
            k = 0
            n = len(tokens)
            while k < n:
                if k + 1 < n and tokens[k] == best_pair[0] and tokens[k + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    k += 2
                else:
                    new_tokens.append(tokens[k])
                    k += 1
            word_tokens[i] = new_tokens

            for k in range(len(new_tokens) - 1):
                new_pair = (new_tokens[k], new_tokens[k + 1])
                pair_counts[new_pair] += c
                pair_to_words[new_pair].add(i)

        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

    return vocab, merges
