from __future__ import annotations
from collections.abc import Iterable, Iterator
from cs336_basics.training_of_tokenizer import load_with_pickle
import json
from tests.common import gpt2_bytes_to_unicode
import regex as re


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def longest_match(self, text: str, start: int) -> str:
        """Find the longest matching token from start position"""
        node = self.root
        longest = ""
        current = ""

        for i in range(start, len(text)):
            char = text[i]
            if char not in node.children:
                break

            node = node.children[char]
            current = current + char

            if node.is_end:
                longest = current

        return longest


class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Build byte_to_unicode mapping
        self.byte_to_unicode = gpt2_bytes_to_unicode()
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        
        # Ensure special tokens are in vocab with correct bytes
        for special_token in self.special_tokens:
            byte_encoded = special_token.encode("utf-8")
            if byte_encoded not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = byte_encoded
        
        # Build reverse_vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build special token to id mapping
        self._special_token_to_id = {}
        for special_token in self.special_tokens:
            byte_encoded = special_token.encode("utf-8")
            if byte_encoded in self.reverse_vocab:
                self._special_token_to_id[special_token] = self.reverse_vocab[byte_encoded]
        
        # Pre-compute merge order for O(1) lookup during encoding
        self._merge_order: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(self.merges):
            merge_gpt2 = (''.join([self.byte_to_unicode[b] for b in merge[0]]).encode('utf-8'),
                         ''.join([self.byte_to_unicode[b] for b in merge[1]]).encode('utf-8'))
            self._merge_order[merge_gpt2] = i
        
        # Build Trie tree for tokenization
        self.trie = Trie()
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                gpt2_unicode_str = ''.join([self.byte_to_unicode[b] for b in byte_seq])
                self.trie.insert(gpt2_unicode_str)

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        self.vocab = load_with_pickle(vocab_filepath)
        self.merges = load_with_pickle(merges_filepath)
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build byte_to_unicode mapping
        self.byte_to_unicode = gpt2_bytes_to_unicode()
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        
        # Build special token to id mapping
        self._special_token_to_id = {}
        for special_token in self.special_tokens:
            byte_encoded = special_token.encode("utf-8")
            if byte_encoded in self.reverse_vocab:
                self._special_token_to_id[special_token] = self.reverse_vocab[byte_encoded]
        
        # Pre-compute merge order for O(1) lookup during encoding
        self._merge_order: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(self.merges):
            merge_gpt2 = (''.join([self.byte_to_unicode[b] for b in merge[0]]).encode('utf-8'),
                         ''.join([self.byte_to_unicode[b] for b in merge[1]]).encode('utf-8'))
            self._merge_order[merge]=i
        
        # Build Trie tree for tokenization
        self.trie = Trie()
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                gpt2_unicode_str = ''.join([self.byte_to_unicode[b] for b in byte_seq])
                self.trie.insert(gpt2_unicode_str)

    def encode_one_token(self, token: str) -> list[int]:
        # Convert input token to UTF-8 bytes, then to GPT2 unicode
        utf8_bytes = token.encode("utf-8")
        gpt2_unicode = ''.join([self.byte_to_unicode[b] for b in utf8_bytes])
        
        # First, split into individual characters (base tokens)
        base_tokens = list(gpt2_unicode)
        
        # Convert base tokens to their token IDs
        token_ids = []
        for char in base_tokens:
            char_byte = self.unicode_to_byte[char]
        
            if char_bytes in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[char_bytes])
            else:
                raise ValueError(f"Cannot tokenize character: {repr(char)}")
        
        # Apply BPE merges in order (like tiktoken does)
        # Keep merging until no more merges are possible
        while len(token_ids) >= 2:
            # Find all adjacent pairs that can be merged
            mergeable_pairs = []
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                
                # Get the byte sequences for these tokens
                byte1 = self.vocab[pair[0]]
                byte2 = self.vocab[pair[1]]
                
                # Check if this pair has a merge in the merge order
                gpt2_1 = ''.join([self.byte_to_unicode[b] for b in byte1])
                gpt2_2 = ''.join([self.byte_to_unicode[b] for b in byte2])
                
                if (gpt2_1, gpt2_2) in self._merge_order:
                    mergeable_pairs.append((i, pair, self._merge_order[(gpt2_1, gpt2_2)]))
            
            if not mergeable_pairs:
                # No more merges possible
                break
            
            # Find the pair with the lowest merge order (earliest merge)
            mergeable_pairs.sort(key=lambda x: x[2])
            best_idx, best_pair, best_order = mergeable_pairs[0]
            
            # Create the merged token
            merged_byte1 = self.vocab[best_pair[0]]
            merged_byte2 = self.vocab[best_pair[1]]
            merged_bytes = merged_byte1 + merged_byte2
            
            # Check if the merged bytes exist in vocabulary
            if merged_bytes not in self.reverse_vocab:
                # This shouldn't happen if merges are valid
                break
            
            # Replace the pair with the merged token
            new_token_id = self.reverse_vocab[merged_bytes]
            token_ids = token_ids[:best_idx] + [new_token_id] + token_ids[best_idx + 2:]
        
        return token_ids

    def pretokenize(self, text: str) -> list[str]:
        """Pre-tokenize text into tokens"""
        if self.special_tokens:
            # Sort special tokens by length (longest first) for proper matching
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special_tokens = [re.escape(token) for token in sorted_special_tokens]
            special_token_pattern = '|'.join(escaped_special_tokens)
            
            # Split while preserving special tokens
            parts = re.split(f'({special_token_pattern})', text)
            
            final_parts = []
            for part in parts:
                if part in self.special_tokens:
                    final_parts.append(part)
                elif part:
                    sub_parts = self._split_by_whitespace(part)
                    final_parts.extend(sub_parts)
        else:
            final_parts = self._split_by_whitespace(text)
        
        return final_parts
    
    def _split_by_whitespace(self, text: str) -> list[str]:
        """Split text by whitespace - GPT-2 style pre-tokenization
        
        GPT-2 tokenizer regex pattern:
        - Contractions like 's, 't, 're, etc. are kept together
        - Optional space followed by word characters -> single token (e.g., " hello" stays as " hello")
        - Optional space followed by numbers -> single token
        - Optional space followed by non-whitespace/non-alphanumeric -> single token
        - Consecutive whitespace -> separate token (e.g., "\n\n" stays as "\n\n")
        - Trailing whitespace -> separate token
        """
        import re
        
        # GPT-2 pattern from tiktoken: handles contractions, leading spaces, and whitespace
        # Order matters - more specific patterns first
        pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s0-9a-zA-Z]+|\s+"
        
        tokens = []
        for match in re.finditer(pattern, text):
            tokens.append(match.group())
        
        return tokens

    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """Apply BPE merges to a list of token IDs within a single pretoken."""
        while len(token_ids) >= 2:
            # Find all adjacent pairs that can be merged
            mergeable_pairs = []
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                
                # Get the byte sequences for these tokens
                byte1 = self.vocab[pair[0]]
                byte2 = self.vocab[pair[1]]
                
                # Check if this pair has a merge in the merge order
                gpt2_1 = ''.join([self.byte_to_unicode[b] for b in byte1])
                gpt2_2 = ''.join([self.byte_to_unicode[b] for b in byte2])
                
                if (gpt2_1, gpt2_2) in self._merge_order:
                    mergeable_pairs.append((i, pair, self._merge_order[(gpt2_1, gpt2_2)]))
            
            if not mergeable_pairs:
                # No more merges possible
                break
            
            # Find the pair with the lowest merge order (earliest merge)
            mergeable_pairs.sort(key=lambda x: x[2])
            best_idx, best_pair, best_order = mergeable_pairs[0]
            
            # Create the merged token
            merged_byte1 = self.vocab[best_pair[0]]
            merged_byte2 = self.vocab[best_pair[1]]
            merged_bytes = merged_byte1 + merged_byte2
            
            # Check if the merged bytes exist in vocabulary
            if merged_bytes not in self.reverse_vocab:
                # This shouldn't happen if merges are valid
                break
            
            # Replace the pair with the merged token
            new_token_id = self.reverse_vocab[merged_bytes]
            token_ids = token_ids[:best_idx] + [new_token_id] + token_ids[best_idx + 2:]
        
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pretokenize(text)
        
        # Process each pretoken independently and apply BPE within each
        results = []
        for token in pre_tokens:
            if token:
                if token in self._special_token_to_id:
                    results.append(self._special_token_to_id[token])
                else:
                    # Convert to base token IDs (single characters)
                    utf8_bytes = token.encode("utf-8")
                    gpt2_unicode = ''.join([self.byte_to_unicode[b] for b in utf8_bytes])
                    
                    # Convert to token IDs
                    token_ids = []
                    for char in gpt2_unicode:
                        char_byte = self.unicode_to_byte[char]
                        char_bytes = bytes([char_byte])
                        token_ids.append(self.reverse_vocab[char_bytes])
                    
                    # Apply BPE merges WITHIN this pretoken only
                    token_ids = self._apply_bpe_merges(token_ids)
                    results.extend(token_ids)
        
        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings, yielding token IDs one at a time.
        
        This is memory-efficient for large files that cannot be loaded into memory.
        """
        for item in iterable:
            for token_id in self.encode(item):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        result_bytes = bytearray()
        
        for id in ids:
            if id in self.vocab:
                byte_seq = self.vocab[id]
                if isinstance(byte_seq, bytes):
                    result_bytes.extend(byte_seq)
                else:
                    result_bytes.extend(str(byte_seq).encode('utf-8'))
            else:
                raise ValueError(f"ID {id} not in vocabulary")
        
        # Decode the concatenated bytes as UTF-8, replacing invalid sequences
        return result_bytes.decode('utf-8', errors='replace')

    def decode_with_spaces(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.vocab:
                byte_seq = self.vocab[id]
                if isinstance(byte_seq, bytes):
                    token = byte_seq.decode("utf-8", errors="replace")
                else:
                    token = str(byte_seq)
                tokens.append(token)
            else:
                raise ValueError(f"ID {id} not in vocabulary")
        
        return " ".join(tokens)


if __name__ == "__main__":
    try:
        import json
        
        def load_json_file1(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        vocab = load_json_file1("/home/tao/assignment1-basics/tests/fixtures/gpt2_vocab.json")
        vocab = {v: k.encode('utf-8') for k, v in vocab.items()}
        
        # Read merges
        merges = []
        with open("/home/tao/assignment1-basics/tests/fixtures/gpt2_merges.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split(" ")) == 2:
                    merges.append(tuple(line.split(" ")))
        
        print("Vocabulary loaded successfully")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Merges size: {len(merges)}")
        
        # Convert merges to bytes
        byte_to_unicode = gpt2_bytes_to_unicode()
        unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
        
        byte_merges = []
        for merge in merges:
            token1_bytes = bytes([unicode_to_byte[c] for c in merge[0]])
            token2_bytes = bytes([unicode_to_byte[c] for c in merge[1]])
            byte_merges.append((token1_bytes, token2_bytes))
        
        tokenizer_instance = tokenizer(vocab, byte_merges, [])
        print("Tokenizer initialized successfully")
        
        test_token = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
        print('preo token:', tokenizer_instance.pretokenize(test_token))
        encoded = tokenizer_instance.encode(test_token)
        print(f"Encoded '{test_token}': {encoded}")
        decoded = tokenizer_instance.decode(encoded)
        print(f"Decoded back: '{decoded}'")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
