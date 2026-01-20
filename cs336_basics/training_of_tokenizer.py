from __future__ import annotations  # 添加这行以支持类型注解中的前向引用

from cs336_basics.pretokenization_example import parallel_file_processing
import os
import pickle
import gzip
from dataclasses import dataclass
from typing import Any



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

def merge_pair(word: tuple, pair: tuple) -> tuple:
    """
    合并单词中的两个字节序列
    """
    for i in range(len(word) - 1):
        if word[i:i + 2] == pair:
            # 创建一个新的元组，包含合并后的字节序列
            new_byte = b''.join(pair)
            return word[:i] + (new_byte,) + word[i + 2:]
    return word

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    pre_tokens = parallel_file_processing(input_path, 8)
    num_merges = vocab_size - len(special_tokens) - 256
    merges = []
    vocab = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode('utf-8')
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    for i in range(num_merges):
        # 每次合并都重新统计所有配对的频率
        pair_counts = {}
        for word, count in pre_tokens.items():  # word为tuple, count为int
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

            
        best_pair = max(pair_counts, key=pair_counts.get)
        vocab[i + 256 + len(special_tokens)] = b''.join(best_pair)
        merges.append(best_pair)

        # 应用合并
        new_pretokens = {}
        for word, count in pre_tokens.items():
            new_word = merge_pair(word, best_pair)
            new_pretokens[new_word] = count
        pre_tokens = new_pretokens

    print(merges)
    print(vocab)
    return vocab, merges
    
if __name__ == "__main__":
    vocab,merges=train_bpe("test.txt", 1000, [''])
    save_with_pickle(vocab, "test_vocab.pkl")
    save_with_pickle(merges, "test_merges.pkl")