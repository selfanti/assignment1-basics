from __future__ import annotations

from collections import Counter
import multiprocessing as mp
import os
import sys
from typing import BinaryIO

import regex as re
from tqdm import tqdm

# Shared pretokenization regex so every worker applies the same GPT-2-style
# splitting semantics as the single-process fallback.
PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def add_dicts(dicts: list[dict[bytes, int]]) -> dict[bytes, int]:
    # Merge per-chunk token histograms into one global histogram.
    result: dict[bytes, int] = {}
    for one_dict in dicts:
        for key, value in one_dict.items():
            result[key] = result.get(key, 0) + value
    return result


def find_chunk_boundaries_fixed(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    # Start from evenly spaced byte offsets, then move boundaries forward to the
    # next special-token boundary. This avoids splitting "<|endoftext|>" across
    # workers and keeps chunking deterministic.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [
        index * chunk_size for index in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(
    start_end: tuple[int, int],
    filename: str,
    special_tokens: list[str] | None = None,
) -> dict[bytes, int]:
    with open(filename, "rb") as file:
        start, end = start_end
        file_size = os.path.getsize(filename)

        # The initial byte offsets may land in the middle of a UTF-8 codepoint.
        # Shift them onto character boundaries before decoding the chunk.
        file.seek(start)
        while start < end:
            file.seek(start)
            first_byte = file.read(1)
            if not first_byte or (first_byte[0] & 0b11000000) != 0b10000000:
                break
            start += 1

        if end < file_size:
            file.seek(end)
            while end < file_size:
                file.seek(end)
                first_byte = file.read(1)
                if not first_byte or (first_byte[0] & 0b11000000) != 0b10000000:
                    break
                end += 1

        file.seek(start)
        chunk_data = file.read(end - start)
        text = chunk_data.decode("utf-8", errors="ignore")

    if special_tokens:
        # Split special tokens out before regex tokenization so their fragments do
        # not leak into the BPE training corpus.
        special_pattern = "|".join(re.escape(token)
                                   for token in special_tokens)
        pieces = re.split(special_pattern, text)
    else:
        pieces = [text]

    # Counter is substantially cheaper here than repeated dict.get updates in a
    # hot loop over every regex match in the chunk.
    token_counts: Counter[bytes] = Counter()
    for piece in pieces:
        for match in PRETOKEN_PATTERN.finditer(piece):
            token_counts[match.group().encode("utf-8")] += 1
    return dict(token_counts)


def _process_chunk_task(
    task: tuple[tuple[int, int], str, list[str] | None],
) -> dict[bytes, int]:
    # imap-style pool APIs only pass one argument per task, so unpack the tuple
    # here and reuse the main worker implementation unchanged.
    return process_chunk(*task)


def parallel_file_processing(
    filename: str,
    special_tokens: list[str] | None = None,
    num_processes: int = 4,
    show_progress: bool = False,
) -> dict[bytes, int]:
    # Build chunk ranges once in the parent process, then fan them out to worker
    # processes for parallel pretokenization.
    num_chunks = max(num_processes * 8, 1)
    with open(filename, "rb") as file:
        boundaries = find_chunk_boundaries_fixed(
            file, num_chunks, b"<|endoftext|>")

    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    tasks = [(chunk, filename, special_tokens) for chunk in chunks]

    with mp.Pool(num_processes) as pool:
        with tqdm(
            total=len(chunks),
            desc="Pretokenize",
            unit="chunk",
            disable=not show_progress,
            file=sys.stderr,
        ) as progress_bar:
            chunk_results = []
            # Update progress from actual completed worker results instead of the
            # private MapResult._number_left batch counter, which can stall at an
            # intermediate value and misreport chunk completion.
            try:
                for chunk_result in pool.imap_unordered(_process_chunk_task, tasks):
                    chunk_results.append(chunk_result)
                    progress_bar.update(1)
            except KeyboardInterrupt:
                # Terminate workers promptly so Ctrl-C stops the training run
                # cleanly instead of leaving the pool in an indeterminate state.
                pool.terminate()
                pool.join()
                raise

    return add_dicts(chunk_results)
