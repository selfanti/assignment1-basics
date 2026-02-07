from __future__ import annotations
from collections.abc import Iterable, Iterator
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
        
        # Ensure special tokens are in vocab with correct bytes
        for special_token in self.special_tokens:
            byte_encoded = special_token.encode("utf-8")
            if byte_encoded not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = byte_encoded
        
        # Build reverse_vocab: maps bytes -> token_id
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build special token to id mapping (string -> token_id)
        self._special_token_to_id = {}
        for special_token in self.special_tokens:
            byte_encoded = special_token.encode("utf-8")
            if byte_encoded in self.reverse_vocab:
                self._special_token_to_id[special_token] = self.reverse_vocab[byte_encoded]
        
        # Build byte_to_unicode and unicode_to_byte mappings
        self.byte_to_unicode = gpt2_bytes_to_unicode()
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        
        # Pre-compute merge order for O(1) lookup during encoding
        # merge_order maps (left_bytes, right_bytes) -> order
        self._merge_order: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(self.merges):
            self._merge_order[merge] = i
        
        # Build Trie tree for pre-tokenization lookup
        self.trie = Trie()
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                gpt2_unicode_str = ''.join([self.byte_to_unicode[b] for b in byte_seq])
                self.trie.insert(gpt2_unicode_str)


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
        
        # GPT-2 pattern from tiktoken: handles contractions, leading spaces, and whitespace
        # Order matters - more specific patterns first
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        tokens = []
        for match in re.finditer(pattern, text):
            tokens.append(match.group())
        
        return tokens

    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """Apply BPE merges to a list of token IDs within a single pretoken.
        
        Uses O(1) merge lookup via _merge_order dictionary.
        """
        while len(token_ids) >= 2:
            # Find the best pair to merge using _merge_order for O(1) lookup
            best_idx = -1
            best_merge_order = float('inf')
            
            for i in range(len(token_ids) - 1):
                left_id = token_ids[i]
                right_id = token_ids[i + 1]
                
                # Get the byte sequences for these tokens
                left_bytes = self.vocab[left_id]
                right_bytes = self.vocab[right_id]
                
                # Check if this pair has a merge using O(1) lookup
                if (left_bytes, right_bytes) in self._merge_order:
                    merge_order = self._merge_order[(left_bytes, right_bytes)]
                    if merge_order < best_merge_order:
                        best_merge_order = merge_order
                        best_idx = i
            
            if best_idx == -1:
                # No more merges possible
                break
            
            # Get the token IDs to merge
            left_id = token_ids[best_idx]
            right_id = token_ids[best_idx + 1]
            
            # Get bytes and merge them
            left_bytes = self.vocab[left_id]
            right_bytes = self.vocab[right_id]
            merged_bytes = left_bytes + right_bytes
            
            # Get the new token ID from reverse_vocab
            new_token_id = self.reverse_vocab[merged_bytes]
            
            # Replace the pair with the merged token
            token_ids = (
                token_ids[:best_idx] + 
                [new_token_id] + 
                token_ids[best_idx + 2:]
            )
        
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
                    # Convert text to UTF-8 bytes
                    utf8_bytes = token.encode("utf-8")
                    
                    # Convert UTF-8 bytes directly to token IDs using reverse_vocab
                    # reverse_vocab maps raw bytes -> token_id
                    token_ids = []
                    for byte in utf8_bytes:
                        byte_tuple = bytes([byte])
                        token_ids.append(self.reverse_vocab[byte_tuple])
                    
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
        """Decode token IDs back to a string.
        
        The vocab contains raw bytes, so we join them directly and decode as UTF-8.
        """
        # Step 1: Get all byte sequences from vocab and concatenate
        result_bytes = b''.join(self.vocab[id] for id in ids)
        
        # Step 2: Decode as UTF-8
        return result_bytes.decode('utf-8', errors='replace')
