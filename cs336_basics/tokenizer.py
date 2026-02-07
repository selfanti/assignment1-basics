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
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None,remapping:bool=False):
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
            self._merge_order[merge] = i
        
        # Build Trie tree for tokenization
        self.trie = Trie()
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                gpt2_unicode_str = ''.join([self.byte_to_unicode[b] for b in byte_seq])
                self.trie.insert(gpt2_unicode_str)
        print('self.vocab[1]',self.vocab[1])
        print('self.merges[8326]',self.merges[8326])

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

    # def encode_one_token(self, token: str) -> list[int]:
    #     # Convert input token to UTF-8 bytes, then to GPT2 unicode
    #     utf8_bytes = token.encode("utf-8")
    #     gpt2_unicode = ''.join([self.byte_to_unicode[b] for b in utf8_bytes])
        
    #     # # First, split into individual characters (base tokens)
    #     # base_tokens = list(gpt2_unicode)
        
    #     # Convert base tokens to their token IDs
    #     token_ids = []
    #     for char in gpt2_unicode:
    #         char_byte = bytes(self.unicode_to_byte[char])
        
    #         if char_byte in self.reverse_vocab:
    #             token_ids.append(self.reverse_vocab[char_byte])
    #         else:
    #             raise ValueError(f"Cannot tokenize character: {repr(char)}")
        
    #     # Apply BPE merges in order (like tiktoken does)
    #     # Keep merging until no more merges are possible
    #     while len(token_ids) >= 2:
    #         # Find all adjacent pairs that can be merged
    #         mergeable_pairs = []
    #         for i in range(len(token_ids) - 1):
    #             pair = (token_ids[i], token_ids[i + 1])
                
    #             # Get the byte sequences for these tokens
    #             byte1 = self.vocab[pair[0]]
    #             byte2 = self.vocab[pair[1]]
                
    #             # Check if this pair has a merge in the merge order
    #             gpt2_1 = ''.join([self.byte_to_unicode[b] for b in byte1])
    #             gpt2_2 = ''.join([self.byte_to_unicode[b] for b in byte2])
                
    #             if (gpt2_1, gpt2_2) in self._merge_order:
    #                 mergeable_pairs.append((i, pair, self._merge_order[(byte1, byte2)]))
            
    #         if not mergeable_pairs:
    #             # No more merges possible
    #             break
            
    #         # Find the pair with the lowest merge order (earliest merge)
    #         mergeable_pairs.sort(key=lambda x: x[2])
    #         best_idx, best_pair, best_order = mergeable_pairs[0]
            
    #         # Create the merged token
    #         merged_byte1 = self.vocab[best_pair[0]]
    #         merged_byte2 = self.vocab[best_pair[1]]
    #         merged_bytes = merged_byte1 + merged_byte2
            
    #         # Check if the merged bytes exist in vocabulary
    #         if merged_bytes not in self.reverse_vocab:
    #             # This shouldn't happen if merges are valid
    #             break
            
    #         # Replace the pair with the merged token
    #         new_token_id = self.reverse_vocab[merged_bytes]
    #         token_ids = token_ids[:best_idx] + [new_token_id] + token_ids[best_idx + 2:]
        
    #     return token_ids

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
        """Apply BPE merges to a list of token IDs within a single pretoken."""
        
        
        while len(token_ids) >= 2:
            mergeable_pairs = []
            
            # 扫描所有相邻对，找出可合并的
            for i in range(len(token_ids) - 1):
                left_id, right_id = token_ids[i], token_ids[i + 1]
                pair = (left_id, right_id)
                
                # 获取两个token的字节序列
                left_bytes = self.vocab[left_id]
                right_bytes = self.vocab[right_id]
                for merge in self.merges:
                    if merge == (left_bytes, right_bytes):
                        mergeable_pairs.append((i, pair, self._merge_order[merge]))
            
            if not mergeable_pairs:
                # 没有可合并的对，终止算法
                break
            
            # 找到优先级最高（merge_rank最小）的合并对
            mergeable_pairs.sort(key=lambda x: x[2])  # 按merge_rank排序
            best_idx, (best_left_id, best_right_id), _ = mergeable_pairs[0]
            
            # 获取两个token的原始字节并合并
            left_bytes = self.vocab[best_left_id]
            right_bytes = self.vocab[best_right_id]
            merged_bytes = left_bytes + right_bytes
            
            # 查找合并后字节序列对应的新token_id
            if merged_bytes not in self.reverse_vocab:
                # 关键修复：抛出异常而非静默中断
                raise KeyError(
                    f"BPE合并失败：字节序列 {merged_bytes!r} 不在词汇表中。"
                    f"合并规则与词汇表可能不匹配。"
                )
            
            new_token_id = self.reverse_vocab[merged_bytes]
            
            # 执行合并：用新token替换原来的两个token
            token_ids = (
                token_ids[:best_idx] + 
                [new_token_id] + 
                token_ids[best_idx + 2:]
            )
            mergeable_pairs.clear()  # 清除当前轮次的合并对，准备下一轮扫描
        
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
                        char_bytes=char.encode('utf-8')
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
        # Step 1: Get all byte sequences from vocab
        result_bytes = []
        for id in ids:
            byte_seq = self.vocab[id]
            result_bytes.append(byte_seq)
        
        # Step 2: Concatenate all bytes
        concatenated = b''.join(result_bytes)
        
        # Step 3: Convert GPT2 unicode characters back to regular bytes
        # Each GPT2 unicode char corresponds to a byte in the original UTF-8 encoding
        regular_bytes = bytes(self.unicode_to_byte[char] for char in concatenated.decode('utf-8'))
        
        # Step 4: Decode as UTF-8
        return regular_bytes.decode('utf-8', errors='replace')



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
                    str1,str2=line.split(" ")
                    byte1=str1.encode('utf-8')
                    byte2=str2.encode('utf-8')
                    merges.append((byte1, byte2))
        print(type(merges[1][0]))
        print("Vocabulary loaded successfully")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Merges size: {len(merges)}")
        

        tokenizer_instance = tokenizer(vocab, merges, [])
        print("Tokenizer initialized successfully")
        
        test_token = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
        #"Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
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
# "🙃" 的编码过程分析：
# "🙃" 的 UTF-8 编码是 F0 9F 99 83
#按照字节进行预分词，实际上unicode码在0~255之间对应的每个字符不都是可见的，
#为了可见，进行了转化，对0~255之间的每个不可见字符进行了映射，映射到可见的字符，即将不可见字符的unicode编码加上256后得到一个新的unicode字符，这样就保证了每个字节都对应一个可见的unicode字符。
# 因此，"🙃"的utf编码会被转换为对应的 GPT-2 unicode 字符：ðŁĻĥ
#
#
#
#
