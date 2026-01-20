from __future__ import annotations
from collections.abc import Iterable
from cs336_basics.training_of_tokenizer import load_with_pickle

# Tokenization follows the training process closely, in the sense that new inputs are tokenized by applying the following steps:
#
# 1. Normalization 在训练过程中，我们通常会应用一些 normalization 步骤，比如将所有字母转换为小写。然而GPT等模型没有忽略大小写
# 2. Pre-tokenization 预分词，按照空格简单进行预分词
# 3. Splitting the words into individual characters  按照字符进行划分
# 4. Applying the merge rules learned in order on those splits  多次合并，每次合并后，词汇表都会增加一个新的词汇，合并表都会增加一个新的合并
#Trie树实现寻找最长匹配
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
        """从start位置开始的最长匹配"""
        node = self.root
        longest = ""
        current = ""

        for i in range(start, len(text)):
            char = text[i]
            if char not in node.children:
                break

            node = node.children[char]
            current += char

            if node.is_end:
                longest = current

        return longest


class tokenizer:
    def __init__(self,vocab: dict[int,bytes],merges: list[tuple[bytes,bytes]],special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.merges=merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab={}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def from_files(self,vocab_filepath: str,merges_filepath: str,special_tokens: list[str] | None=None):
        self.vocab=load_with_pickle(vocab_filepath)
        self.merges = load_with_pickle(merges_filepath)
        self.special_tokens = special_tokens or []
        self.reverse_vocab={v:k for k,v in self.vocab.items()}
    
    def encode_one_token(self,token: str)->list[int]:
        # 构建Trie树
        trie = Trie()
        
        # 遍历词汇表，将每个字节序列解码为字符串用于Trie树构建
        for token_id, byte_seq in self.vocab.items():
            # 尝试用UTF-8解码
            decoded_str = byte_seq.decode("utf-8",errors="replace")
            trie.insert(decoded_str)

        result = []
        i = 0
        n = len(token)

        while i < n:
            # 查找从i开始的最长匹配
            match = trie.longest_match(token, i)

            if match:
                # 找到匹配后，需要确定其在词汇表中对应的原始字节序列和ID
                matched_id = None
                for token_id, byte_seq in self.vocab.items():
                    if isinstance(byte_seq, bytes):
                        decoded_str = byte_seq.decode("utf-8",errors="replace")

                        if decoded_str == match:
                            matched_id = token_id
                            break
                    else:
                        if str(byte_seq) == match:
                            matched_id = token_id
                            break
                
                if matched_id is not None:
                    result.append(matched_id)
                    i += len(match)
                else:
                    # 如果找不到匹配项，按字符编码
                    char_to_encode = token[i].encode("utf-8")
                    for token_id, byte_seq in self.vocab.items():
                        if byte_seq == char_to_encode:
                            result.append(token_id)
                            break

                    
                    i += 1
            else:
                # 没有匹配，处理单个字符
                char_to_encode = token[i].encode("utf-8")
                char_found = False
                for token_id, byte_seq in self.vocab.items():
                    if byte_seq == char_to_encode:
                        result.append(token_id)
                        break

                
                i += 1

        return result

    def pretokenize(self, text: str) -> list[str]:
        """
        预分词处理，将文本分割为token，保留空格与后续非空字符的组合
        """
        import re
        # 使用正则表达式分割文本，保留空格与后续内容的组合
        parts = re.split(r'(\s*)(\S+)', text)
        
        # 清理parts数组，移除空字符串，但保留有意义的部分
        result_parts = []
        i = 0
        while i < len(parts):
            if i < len(parts) and parts[i]:  # 如果当前部分不为空
                if i + 2 < len(parts) and parts[i].isspace():  # 如果当前部分是空格
                    # 组合空格和下一个非空部分
                    combined_part = parts[i] + parts[i+1]
                    result_parts.append(combined_part)
                    i += 3  # 跳过空格、下一个部分和随后的空字符串
                elif not parts[i].isspace():  # 如果不是空格也不是空字符串
                    result_parts.append(parts[i])
                    i += 1
                else:  # 单独的空格部分
                    result_parts.append(parts[i])
                    i += 1
            else:
                i += 1
        
        # 过滤掉空字符串，只保留有意义的部分
        result_parts = [part for part in result_parts if part]
        return result_parts

    def encode(self, text: str) -> list[int]:
        # 使用预分词函数处理文本
        pre_tokens = self.pretokenize(text)
        
        results = []
        for token in pre_tokens:
            if token:  # 忽略空字符串
                token_ids = self.encode_one_token(token)
                results.extend(token_ids)
        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        results = []
        for item in iterable:
            results.extend(self.encode(item))
        return results

    def decode(self, ids: list[int]) -> str:
        str_result = ""
        for i, id in enumerate(ids):
            if id in self.vocab:
                byte_seq = self.vocab[id]
                if isinstance(byte_seq, bytes):
                    token_str = byte_seq.decode("utf-8", errors="replace")
                    str_result += token_str
                else:
                    str_result += str(byte_seq)
            else:
                # 如果ID不在词汇表中，抛出错误
                raise ValueError(f"ID {id} not in vocabulary")
        return str_result

    def decode_with_spaces(self, ids: list[int]) -> str:
        # 实现一个新方法，将ID列表转换为token，然后在它们之间添加空格
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
                # 如果ID不在词汇表中，抛出错误
                raise ValueError(f"ID {id} not in vocabulary")
        
        # 在tokens之间添加空格
        return " ".join(tokens)

if __name__ == "__main__":
    try:
        vocab=load_with_pickle("test_vocab.pkl")
        merges=load_with_pickle("test_merges.pkl")
        print("Vocabulary loaded successfully")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Merges size: {len(merges)}")
        tokenizer_instance=tokenizer(vocab,merges,[''])
        print("Tokenizer initialized successfully")
        test_token = "hello, I am Tom. How's your day? Make sure the .pkl files exist at the specified paths."
        encoded = tokenizer_instance.encode(test_token)
        print(f"Encoded '{test_token}': {encoded}")
        decoded = tokenizer_instance.decode(encoded)
        print(f"Decoded back: '{decoded}'")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure the .pkl files exist at the specified paths.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()