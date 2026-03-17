from __future__ import annotations

import argparse
import gzip
import heapq
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization import parallel_file_processing
from cs336_basics.tokenizer import Tokenizer

# Keep the pretokenization regex compiled once: both the single-process fast path
# and the parallel chunk workers need the exact same GPT-2-style splitting rules.
PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class DescBytes:
    # heapq only knows how to pop the "smallest" item. We wrap bytes in a custom
    # comparator so that larger byte strings win ties, matching the assignment's
    # exact tie-break rule: count desc, then left_bytes desc, then right_bytes desc.
    value: bytes

    def __lt__(self, other: "DescBytes") -> bool:
        return self.value > other.value


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                          ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def save_with_pickle(data: Any, filename: str, compress: bool = False):
    """使用pickle保存数据"""
    mode = 'wb'  # 二进制写入

    if compress:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"数据已保存到 {filename} (压缩: {compress})")


def load_with_pickle(filename: str, compress: bool = False) -> Any:
    """使用pickle加载数据"""
    if compress:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def pretokenize_file(input_path: str, special_tokens: list[str] | None = None) -> dict[bytes, int]:
    """
    Read a file and pretokenize it, returning a dict of token bytes to counts.
    Uses GPT-2 style pretokenization regex.

    If special_tokens are provided, they are excluded from the corpus for training.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        # Split special tokens out before regex tokenization so their fragments
        # never enter the BPE training corpus.
        special_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
        pieces = re.split(special_pattern, text)
    else:
        pieces = [text]

    token_counts = {}
    for piece in pieces:
        for match in PRETOKEN_PATTERN.finditer(piece):
            token_str = match.group()
            token_bytes = token_str.encode('utf-8')
            token_counts[token_bytes] = token_counts.get(token_bytes, 0) + 1

    return token_counts


def _iter_adjacent_pairs(symbols: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    # Centralize adjacent-pair generation so counting and recounting use the exact
    # same definition of "neighboring pair".
    for index in range(len(symbols) - 1):
        yield (symbols[index], symbols[index + 1])


def _merge_pair_in_word(
    symbols: tuple[int, ...],
    pair: tuple[int, int],
    new_token_id: int,
) -> tuple[int, ...]:
    # Apply one merge exactly like the reference tuple-based implementation:
    # left-to-right and non-overlapping within each pretoken.
    left_id, right_id = pair
    merged: list[int] = []
    index = 0
    last_index = len(symbols) - 1

    while index < len(symbols):
        if index < last_index and symbols[index] == left_id and symbols[index + 1] == right_id:
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1

    return tuple(merged)


def _select_best_pair(
    pair_heap: list[tuple[int, DescBytes, DescBytes, tuple[int, int]]],
    pair_counts: dict[tuple[int, int], int],
):
    # The heap is "lazy": stale entries remain inside it after pair counts change.
    # We discard entries until we find one whose cached count still matches the
    # current global count.
    while pair_heap:
        neg_count, _, _, pair = heapq.heappop(pair_heap)
        current_count = pair_counts.get(pair)
        if current_count is None:
            continue
        if current_count == -neg_count:
            return pair
    raise ValueError("No valid BPE pairs remain.")


def _push_pair(
    pair_heap: list[tuple[int, DescBytes, DescBytes, tuple[int, int]]],
    pair: tuple[int, int],
    count: int,
    vocab: dict[int, bytes],
) -> None:
    # Store enough information in the heap node to reproduce the exact tie-break
    # behavior without rescanning the whole pair_counts dict every merge.
    heapq.heappush(
        pair_heap,
        (-count, DescBytes(vocab[pair[0]]), DescBytes(vocab[pair[1]]), pair),
    )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8,
    show_progress: bool = False,
):
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens

    Returns:
        vocab: dict mapping token_id -> bytes
        merges: list of (left_bytes, right_bytes) tuples in order of creation
    """
    # For small corpora, process startup dominates and the single-process path is
    # faster. Large corpora still benefit from chunked pretokenization.
    file_size = os.path.getsize(input_path)
    if num_processes <= 1 or file_size < 8 * 1024 * 1024:
        token_counts = pretokenize_file(input_path, special_tokens)
    else:
        token_counts = parallel_file_processing(
            input_path,
            special_tokens,
            num_processes,
            show_progress=show_progress,
        )
    num_special = len(special_tokens)
    base_vocab_size = 256 + num_special
    num_merges = vocab_size - base_vocab_size
    if num_merges <= 0:
        # This keeps the function well-defined even when the caller only wants the
        # base byte vocabulary plus special tokens.
        vocab = {i: special_token.encode(
            "utf-8") for i, special_token in enumerate(special_tokens)}
        for i in range(256):
            vocab[num_special + i] = bytes([i])
        return vocab, []

    # Initialize vocab
    vocab = {}

    # Add special tokens first
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode('utf-8')

    # Add byte tokens
    for i in range(256):
        vocab[num_special + i] = bytes([i])

    merges: list[tuple[bytes, bytes]] = []
    # word_symbols stores each unique pretoken once, weighted by word_counts.
    # The optimization comes from updating only the words touched by the chosen
    # merge instead of rescanning every word on every iteration.
    word_symbols: dict[int, tuple[int, ...]] = {}
    word_counts: dict[int, int] = {}
    word_pair_counters: dict[int, Counter[tuple[int, int]]] = {}
    # pair_counts is the global weighted frequency of each adjacent pair.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    # pair_to_words tells us exactly which unique words contain a given pair, so
    # a merge only revisits affected words.
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_id, (token_bytes, count) in enumerate(token_counts.items()):
        # Internal training uses token ids, not raw bytes. Byte tokens start at
        # num_special, so we must map each raw byte to its true vocab id to keep
        # tie-breaking and later merges aligned with the public vocab.
        symbols = tuple(num_special + byte_value for byte_value in token_bytes)
        word_symbols[word_id] = symbols
        word_counts[word_id] = count
        pair_counter = Counter(_iter_adjacent_pairs(symbols))
        word_pair_counters[word_id] = pair_counter
        for pair, occurrences in pair_counter.items():
            pair_counts[pair] += occurrences * count
            pair_to_words[pair].add(word_id)

    # The heap turns "find best pair" from an O(number_of_pairs) scan each round
    # into amortized O(log number_of_pairs) pushes/pops.
    pair_heap: list[tuple[int, DescBytes, DescBytes, tuple[int, int]]] = []
    for pair, count in pair_counts.items():
        _push_pair(pair_heap, pair, count, vocab)

    progress_bar = tqdm(
        total=num_merges,
        desc="BPE merges",
        unit="merge",
        disable=not show_progress,
        file=sys.stderr,
    )

    for merge_idx in range(num_merges):
        if not pair_counts:
            break

        best_pair = _select_best_pair(pair_heap, pair_counts)
        left_id, right_id = best_pair

        new_token_id = base_vocab_size + merge_idx
        new_token_bytes = vocab[left_id] + vocab[right_id]
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[left_id], vocab[right_id]))

        # Only words containing the winning pair can change after this merge.
        affected_word_ids = list(pair_to_words.pop(best_pair, set()))
        for word_id in affected_word_ids:
            old_symbols = word_symbols[word_id]
            old_pair_counter = word_pair_counters[word_id]
            new_symbols = _merge_pair_in_word(
                old_symbols, best_pair, new_token_id)
            if new_symbols == old_symbols:
                # This should be rare, but if a stale word slips through we keep
                # the index consistent and move on safely.
                pair_to_words[best_pair].add(word_id)
                continue

            new_pair_counter = Counter(_iter_adjacent_pairs(new_symbols))
            word_frequency = word_counts[word_id]

            # Reconcile this word's old/new local pair counts back into the
            # global pair_counts table. The delta is weighted by corpus frequency.
            for pair in set(old_pair_counter) | set(new_pair_counter):
                old_occurrences = old_pair_counter.get(pair, 0)
                new_occurrences = new_pair_counter.get(pair, 0)
                delta = (new_occurrences - old_occurrences) * word_frequency
                if delta == 0:
                    continue

                updated_count = pair_counts.get(pair, 0) + delta
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                    _push_pair(pair_heap, pair, updated_count, vocab)
                else:
                    pair_counts.pop(pair, None)

                # Keep the reverse index synchronized with the word-local pair
                # counter so the next merge can find affected words in O(1).
                if old_occurrences == 0 and new_occurrences > 0:
                    pair_to_words[pair].add(word_id)
                elif old_occurrences > 0 and new_occurrences == 0:
                    word_ids = pair_to_words.get(pair)
                    if word_ids is not None:
                        word_ids.discard(word_id)
                        if not word_ids:
                            pair_to_words.pop(pair, None)

            word_symbols[word_id] = new_symbols
            word_pair_counters[word_id] = new_pair_counter

        # The chosen pair has been fully consumed by this merge step.
        pair_counts.pop(best_pair, None)

        if show_progress:
            progress_bar.update(1)
            progress_bar.set_postfix(
                active_pairs=len(pair_counts),
                affected_words=len(affected_word_ids),
                refresh=False,
            )

    progress_bar.close()

    return vocab, merges


def save_tokenizer_artifacts(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> Path:
    # Keep tokenizer artifacts alongside the tokenized dataset so training can be
    # reproduced without rerunning BPE.
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    config_path = output_dir / "tokenizer_config.json"

    save_with_pickle(vocab, str(vocab_path))
    save_with_pickle(merges, str(merges_path))
    config_path.write_text(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": special_tokens,
                "vocab_path": vocab_path.name,
                "merges_path": merges_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def get_token_dtype(vocab_size: int, dtype_name: str) -> np.dtype:
    if dtype_name == "auto":
        # Use the smallest safe integer width to reduce token file size.
        if vocab_size <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    dtype = np.dtype(dtype_name)
    if vocab_size > np.iinfo(dtype).max:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds {dtype.name}. "
            "Use --dtype auto or a larger integer type."
        )
    return dtype


def iter_token_ids_from_file(
    tokenizer: Tokenizer,
    dataset_path: Path,
    encoding: str,
    append_eod: bool,
) -> Iterator[int]:
    # This generator is shared by the counting pass and the writing pass so the
    # corpus-to-token mapping stays identical across both scans.
    eod_token_id: int | None = None
    if append_eod:
        if not tokenizer.special_tokens:
            raise ValueError(
                "--append_eod requires at least one special token.")
        eod_token = tokenizer.special_tokens[0]
        eod_token_id = tokenizer.reverse_vocab[eod_token.encode("utf-8")]

    with dataset_path.open("r", encoding=encoding, errors="ignore") as source:
        for line in source:
            yield from tokenizer.encode(line)
            if append_eod and eod_token_id is not None:
                yield eod_token_id


def count_total_tokens(
    tokenizer: Tokenizer,
    dataset_path: Path,
    encoding: str,
    append_eod: bool,
) -> int:
    # A single .npy file needs its final shape up front, so we count once before
    # opening the memmap for writing.
    return sum(1 for _ in iter_token_ids_from_file(tokenizer, dataset_path, encoding, append_eod))


def write_tokens_to_npy(
    tokenizer: Tokenizer,
    dataset_path: Path,
    output_path: Path,
    token_dtype: np.dtype,
    total_tokens: int,
    encoding: str,
    append_eod: bool,
) -> None:
    # open_memmap writes a real .npy file incrementally, which keeps memory usage
    # flat even for very large tokenized corpora.
    token_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=token_dtype,
        shape=(total_tokens,),
    )

    write_index = 0
    for token_id in iter_token_ids_from_file(tokenizer, dataset_path, encoding, append_eod):
        token_array[write_index] = token_id
        write_index += 1

    del token_array


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and save the tokenized dataset as a single .npy file."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the text dataset.",
    )
    parser.add_argument(
        "vocabulary_size",
        type=int,
        help="Tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for tokenizer artifacts and tokens.npy.",
    )
    parser.add_argument(
        "--special_token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token to preserve. Can be passed multiple times.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes used during BPE training.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "uint16", "uint32"],
        default="uint16",
        help="Integer dtype used to save tokens.npy.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding used to read the dataset.",
    )
    parser.add_argument(
        "--append_eod",
        action="store_true",
        help="Append the first special token after every line.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if args.num_processes <= 0:
        raise ValueError("--num_processes must be a positive integer.")

    special_tokens = args.special_tokens or ["<|endoftext|>"]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_path.parent / f"{dataset_path.stem}_tokenized"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training tokenizer from {dataset_path} ...")
    vocab, merges = train_bpe(
        str(dataset_path),
        args.vocabulary_size,
        special_tokens,
        num_processes=args.num_processes,
        show_progress=True,
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    tokenizer_config_path = save_tokenizer_artifacts(
        output_dir, vocab, merges, special_tokens)

    token_dtype = get_token_dtype(len(vocab), args.dtype)
    print("Counting total token count ...")
    total_tokens = count_total_tokens(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        encoding=args.encoding,
        append_eod=args.append_eod,
    )

    tokens_path = output_dir / "tokens.npy"
    print(f"Writing {total_tokens} tokens to {tokens_path} ...")
    write_tokens_to_npy(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        output_path=tokens_path,
        token_dtype=token_dtype,
        total_tokens=total_tokens,
        encoding=args.encoding,
        append_eod=args.append_eod,
    )

    metadata_path = output_dir / "tokenized_dataset.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "tokens_path": tokens_path.name,
                "tokenizer_config_path": tokenizer_config_path.name,
                "total_tokens": total_tokens,
                "token_dtype": token_dtype.name,
                "special_tokens": special_tokens,
                "append_eod": args.append_eod,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Tokenization complete: {tokens_path}")
    print(f"Tokenizer config: {tokenizer_config_path}")
    print(f"Dataset metadata: {metadata_path}")


if __name__ == "__main__":
    main()
