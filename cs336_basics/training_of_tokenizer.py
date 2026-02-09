from __future__ import annotations

import os
import pickle
import gzip
import regex as re
from typing import Any
from cs336_basics.pretokenization_example import parallel_file_processing

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
    
    # Convert special tokens to bytes for filtering
    special_token_bytes = set()
    if special_tokens:
        for token in special_tokens:
            special_token_bytes.add(token.encode('utf-8'))
    
    # GPT-2 pattern
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    token_counts = {}
    for match in re.finditer(pattern, text):
        token_str = match.group()
        token_bytes = token_str.encode('utf-8')
        
        # Skip special tokens - they should not be part of BPE training
        if token_bytes in special_token_bytes:
            continue
        
        token_counts[token_bytes] = token_counts.get(token_bytes, 0) + 1
    
    return token_counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
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
    # Get pretokenized corpus as dict of token_bytes -> count (excluding special tokens)
    #token_counts=pretokenize_file(input_path,special_tokens)
<<<<<<< HEAD
    token_counts=parallel_file_processing(input_path,special_tokens,4)
=======
    token_counts=parallel_file_processing(input_path,special_tokens,8)
>>>>>>> temp-branch
    num_special = len(special_tokens)
    base_vocab_size = 256 + num_special
    num_merges = vocab_size - base_vocab_size
    
    # Initialize vocab
    vocab = {}
    
    # Add special tokens first
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode('utf-8')
    
    # Add byte tokens
    for i in range(256):
        vocab[num_special + i] = bytes([i])
    
    # Track merges
    merges = []
    
    # Compute substrings of special tokens that should not be learned
    # We must not create any vocab entry that is a proper substring of a special token,
    # because during tokenization, the tokenizer might match this substring instead of
    # continuing to match the full special token.
    # However, we only filter when the special token pattern actually appears in the corpus.
    # For example, if "<|endoftext|>" is a special token and "<|" appears in the corpus,
    # we should prevent learning "|>" to avoid tokenizing "<|..." as "<" + "|>" instead of "<|..."
    forbidden_substrings = set()
    for special_token in special_tokens:
        token_bytes = special_token.encode('utf-8')
        # Check if this special token's prefix appears in the corpus
        # If not, we don't need to filter anything for this token
        for prefix_len in range(1, len(token_bytes)):
            prefix = token_bytes[:prefix_len]
            if prefix in token_counts:
                # The prefix appears, so we should filter proper suffixes
                # to prevent them from combining to form the full special token
                for i in range(len(token_bytes)):
                    for j in range(i + 1, len(token_bytes) + 1):
                        if i != 0 or j != len(token_bytes):  # Not the full token
                            substr = token_bytes[i:j]
                            # Only filter multi-char substrings that appear in corpus
                            if len(substr) > 1 and substr in token_counts:
                                forbidden_substrings.add(substr)
                break  # Found the prefix, no need to check longer ones
    
    # Track words as tuples of individual bytes with their counts
    # word_tokens: maps word tuple (of bytes) -> count
    word_tokens = {}
    for token_bytes, count in token_counts.items():
        # Split each pretokenized word into individual bytes
        # Each byte becomes a single-byte bytes object
        word_as_tuple = tuple(bytes([b]) for b in token_bytes)
        word_tokens[word_as_tuple] = count
    
    for merge_idx in range(num_merges):
        # Count all adjacent pairs across all words
        pair_counts = {}
        
        for word_tuple, count in word_tokens.items():
            for j in range(len(word_tuple) - 1):
                left = word_tuple[j]
                right = word_tuple[j + 1]
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        if not pair_counts:
            break
        
        # Filter out pairs that would create a forbidden substring
        valid_pairs = {}
        for pair, count in pair_counts.items():
            merged = pair[0] + pair[1]
            if merged not in forbidden_substrings:
                valid_pairs[pair] = count
        
        if not valid_pairs:
            break
        
        # Find the most frequent pair, using tuple order as tie-breaker
        # This ensures deterministic results matching the reference
        best_pair = max(valid_pairs.items(), key=lambda x: (x[1], x[0][0], x[0][1]))[0]
        #best_pair = max(valid_pairs, key=lambda x: (valid_pairs[x], pair_strings.get(x, ('b', 'b'))))
        left_bytes, right_bytes = best_pair
        
        # Add to vocab
        new_token_id = base_vocab_size + merge_idx
        new_token_bytes = left_bytes + right_bytes
        vocab[new_token_id] = new_token_bytes
        
        # Record the merge
        merges.append(best_pair)
        
        # Apply merge to all words
        new_word_tokens = {}
        for word_tuple, count in word_tokens.items():
            new_tokens = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == left_bytes and word_tuple[i + 1] == right_bytes:
                    new_tokens.append(new_token_bytes)
                    i += 2
                else:
                    new_tokens.append(word_tuple[i])
                    i += 1
            new_word = tuple(new_tokens)
            new_word_tokens[new_word] = count
        
        word_tokens = new_word_tokens
    
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("test.txt", 1000, ["<|endoftext|>"])
    print(vocab,merges)
    save_with_pickle(vocab, "test_vocab.pkl")
    save_with_pickle(merges, "test_merges.pkl")
