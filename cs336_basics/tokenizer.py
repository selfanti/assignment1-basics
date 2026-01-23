from __future__ import annotations
from collections.abc import Iterable
from cs336_basics.training_of_tokenizer import load_with_pickle
import json
from tests.common import gpt2_bytes_to_unicode
import pickle
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

    def insert(self, word: str):  # 修改为接受字符串而不是bytes
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def longest_match(self, text: str, start: int) -> str:  # 修改为处理字符串
        """从start位置开始的最长匹配"""
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
def read_bpe_merge_file(file_path: str) -> list[tuple[bytes, bytes]]:
    """
    读取GPT-2 BBPE算法的merge表文件
    
    参数:
        file_path: merge表文件路径
        
    返回:
        list[tuple[bytes, bytes]]: 合并规则列表，每个元组包含两个待合并的字节序列
    """
    merges = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 跳过空行和注释行
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 按空格分割，但要注意GPT-2的token可能包含空格字符本身
                parts = line.split()
                if len(parts) != 2:
                    print(f"警告: 第{line_num}行格式不正确: {line}")
                    continue
                
                # 将token字符串转换为bytes
                # GPT-2使用特殊的Unicode字符表示，需要正确编码
                try:
                    token1 = parts[0].encode('utf-8')
                    token2 = parts[1].encode('utf-8')
                    merges.append((token1, token2))
                except UnicodeEncodeError as e:
                    print(f"警告: 第{line_num}行编码错误: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件不存在: {file_path}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return []
    
    return merges

class tokenizer:
    def __init__(self,vocab: dict[int,bytes],merges: list[tuple[bytes,bytes]],special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.merges=merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        # 构建Trie树
        self.trie = Trie()
        # 需要从vocab构建GPT-2映射的Unicode字符序列
        self.byte_to_unicode = gpt2_bytes_to_unicode()
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        # 将字节序列转换为GPT-2映射的Unicode字符序列并插入Trie
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                # 将字节序列转换为GPT-2映射的Unicode字符序列
                token_str=byte_seq.decode('utf-8')
                self.trie.insert(token_str)

    def from_files(self,vocab_filepath: str,merges_filepath: str,special_tokens: list[str] | None=None):
        self.vocab=load_with_pickle(vocab_filepath)
        self.merges = load_with_pickle(merges_filepath)
        self.special_tokens = special_tokens or []
        self.reverse_vocab={v:k for k,v in self.vocab.items()}
        # 重建Trie树
        self.trie = Trie()
        
        byte_to_unicode = gpt2_bytes_to_unicode()
        
        # 将字节序列转换为GPT-2映射的Unicode字符序列并插入Trie
        for token_id, byte_seq in self.vocab.items():
            if isinstance(byte_seq, bytes):
                # 将字节序列转换为GPT-2映射的Unicode字符序列
                gpt2_unicode_str = ''.join([byte_to_unicode[b] for b in byte_seq])
                self.trie.insert(gpt2_unicode_str)
            else:
                # 如果不是bytes类型，直接插入
                self.trie.insert(str(byte_seq))
    def merge_pair(self,word: tuple, pair: tuple) -> tuple:
        """
        合并单词中的两个字节序列
        """
        for i in range(len(word) - 1):
            if word[i:i + 2] == pair:
                # 创建一个新的元组，包含合并后的字节序列
                new_byte = b''.join(pair)
                return word[:i] + (new_byte,) + word[i + 2:]
        return word
    def encode_one_token(self,token: str)->list[int]:

        # 首先将输入token转换为GPT-2映射的Unicode字符
        utf8_bytes = token.encode("utf-8")
        
        # 将UTF-8字节转换为GPT-2映射的Unicode字符序列
        gpt2_unicode_str = ''.join([self.byte_to_unicode[b] for b in utf8_bytes])
        gpt2_unicode_bytes = tuple(gpt2_unicode_str[i].encode('utf-8') for i in range(len(gpt2_unicode_str)))
        for merge in self.merges:
            gpt2_unicode_bytes=self.merge_pair(gpt2_unicode_bytes,merge)
        gpt2_unicode_str =[b.decode('utf-8') for b in gpt2_unicode_bytes]
        result = []
        
        
        for str in gpt2_unicode_str:
            i = 0
            n = len(str)
            while i < n:
                # 查找从i开始的最长匹配
                match = self.trie.longest_match(str, i)
                if match:
                    # 找到匹配，需要找到对应的token ID
                    # 将匹配的Unicode字符串转换回字节序列来查找reverse_vocab
                    # matched_bytes_list = []
                    # for uc in match:
                    #     matched_bytes_list.append(self.unicode_to_byte[uc])
                    # matched_bytes = bytes(matched_bytes_list)
                    # print(matched_bytes)
                    match_bytes=match.encode('utf-8')
                    # 通过字节序列查找token ID
                    if match_bytes in self.reverse_vocab.keys():
                        matched_id = self.reverse_vocab[match_bytes]
                        result.append(matched_id)
                        i += len(match)  # 移动索引，跳过已匹配的字符数
                    else:
                        # 如果找不到匹配的token ID，这不应该发生，抛出错误
                        raise ValueError(f"Matched sequence {match} not found in vocabulary")
                else:
                    # 如果没有找到匹配，处理单个字符
                    single_char = gpt2_unicode_str[i]
                    single_byte = self.unicode_to_byte[single_char]
                    
                    # 在词汇表中查找这个单字节token
                    single_byte_seq = bytes([single_byte])
                    if single_byte_seq in self.reverse_vocab:
                        result.append(self.reverse_vocab[single_byte_seq])
                    else:
                        # 如果单字节不在词汇表中，这是一个错误状态
                        raise ValueError(f"Single byte {single_byte} not found in vocabulary")
                    
                    i += 1

        return result

    def pretokenize(self, text: str) -> list[str]:
        """
        预分词处理，将文本分割为token，保留空格与后续非空字符的组合
        同时保留特殊token的完整性
        """
        import re
        
        # 如果有特殊token，将它们作为整体保留
        if self.special_tokens:
            # 创建一个正则表达式模式，匹配任何特殊token，按长度降序排列以确保长的特殊token优先匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special_tokens = [re.escape(token) for token in sorted_special_tokens]
            special_token_pattern = '|'.join(escaped_special_tokens)
            
            # 使用正则表达式分割，但保留特殊token
            parts = re.split(f'({special_token_pattern})', text)
            
            final_parts = []
            for part in parts:
                if part in self.special_tokens:
                    # 如果是特殊token，直接添加
                    final_parts.append(part)
                elif part:
                    # 如果不是特殊token但非空，按常规方式进行处理
                    sub_parts = self._split_by_space_and_word(part)
                    final_parts.extend(sub_parts)
        else:
            # 没有特殊token，进行常规处理
            final_parts = self._split_by_space_and_word(text)
        
        return final_parts
    
    def _split_by_space_and_word(self, text: str) -> list[str]:
        """
        按照空格与单词的组合进行分割
        """
        tokens = []
        i = 0
        while i < len(text):
            # 如果当前字符是空格
            if text[i].isspace():
                # 找到空格序列的结束
                start = i
                while i < len(text) and text[i].isspace():
                    i += 1
                # 找到非空字符序列的结束
                start_non_space = i
                while i < len(text) and not text[i].isspace():
                    i += 1
                # 组合空格和非空字符
                if start_non_space < len(text):
                    tokens.append(text[start:i])
                else:
                    # 如果文本末尾只有空格，也添加
                    tokens.append(text[start:start_non_space])
            else:
                # 非空格字符，找到序列的结束
                start = i
                while i < len(text) and not text[i].isspace():
                    i += 1
                tokens.append(text[start:i])
        
        return tokens

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
        result_bytes = bytearray()
        
        for id in ids:
            if id in self.vocab:
                byte_seq = self.vocab[id]
                if isinstance(byte_seq, bytes):
                    # byte_seq是原始UTF-8字节序列，可以直接使用
                    result_bytes.extend(byte_seq)
                else:
                    # 如果不是bytes类型，将其编码为UTF-8
                    result_bytes.extend(str(byte_seq).encode('utf-8'))
            else:
                # 如果ID不在词汇表中，抛出错误
                raise ValueError(f"ID {id} not in vocabulary")
                
        # 将收集的字节解码为字符串
        result=[]
        result_bytes=result_bytes.decode('utf-8')
        for i in range(len(result_bytes)):
            one_byte=self.unicode_to_byte[result_bytes[i]]
            result.append(one_byte)
            # 尝试解码当前字节序列
        return bytes(result).decode('utf-8')

    def _get_gpt2_char_for_byte(self, byte_val: int) -> str:
        # 返回GPT-2中特定字节值对应的Unicode字符
        # 这是gpt2_bytes_to_unicode函数的逆操作
        # 但我们无法直接访问，所以我们需要从reverse_vocab推断
        # 实际上，我们应该在初始化时构建这个映射
        import sys
        if not hasattr(self, 'byte_to_unicode'):
            # 导入并构建GPT-2的字节到Unicode映射
            from tests.common import gpt2_bytes_to_unicode
            self.byte_to_unicode = gpt2_bytes_to_unicode()
        return self.byte_to_unicode[byte_val]

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
def load_json_file1(filename):
    """从 JSON 文件加载数据（推荐方法）"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
if __name__ == "__main__":
    try:
        #vocab=load_with_pickle("test_vocab.pkl")
        vocab=load_json_file1("/home/tao/assignment1-basics/tests/fixtures/gpt2_vocab.json")
        vocab={v:k.encode('utf-8') for k,v in vocab.items()} # int : string
        with open('dict_output.txt', 'w', encoding='utf-8') as f:
            for key, value in vocab.items():
                f.write(f'{key}: {value}\n')
        merges=read_bpe_merge_file("/home/tao/assignment1-basics/tests/fixtures/gpt2_merges.txt")
        print("Vocabulary loaded successfully")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Merges size: {len(merges)}")
        tokenizer_instance=tokenizer(vocab,merges,[])
        print("Tokenizer initialized successfully")
        test_token = " "
        print('preo token:',tokenizer_instance.pretokenize(test_token))
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