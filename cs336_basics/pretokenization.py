# import os
from typing import BinaryIO
import multiprocessing as mp
import time
import regex as re
import os


def add_dicts(dicts: list[dict]) -> dict:
    """合并字典"""
    result = {}
    for d in dicts:
        for key, value in d.items():
            result[key] = result.get(key, 0) + value
    return result

def find_chunk_boundaries_fixed(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    修复的边界查找：确保边界在UTF-8字符边界
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    mini_chunk_size = 4096
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        
        while True:
            mini_chunk = file.read(mini_chunk_size)
            
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break
            
            # 1. 先尝试在特殊标记处分割
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




def process_chunk(start_end: tuple, filename: str,special_tokens: list[str] | None = None):
    """处理单个块的函数（可在不同进程中运行）"""
    with open(filename, "rb") as f:
        start, end = start_end
        
        # 调整开始位置到字符边界
        f.seek(start)
        while start < end:
            f.seek(start)
            first_byte = f.read(1)
            if not first_byte:
                break
            # 检查是否是字符开始
            if (first_byte[0] & 0b11000000) != 0b10000000:
                break
            start += 1
        
        # 调整结束位置到字符边界
        if end < os.path.getsize(filename):
            f.seek(end)
            while end < os.path.getsize(filename):
                f.seek(end)
                first_byte = f.read(1)
                if not first_byte:
                    break
                if (first_byte[0] & 0b11000000) != 0b10000000:
                    break
                end += 1
        
        # 读取数据
        f.seek(start)
        chunk_data = f.read(end - start)
        
        # 解码文本
        text = chunk_data.decode("utf-8", errors="ignore")
    # Convert special tokens to bytes for filtering
    special_token_bytes = set()
    if special_tokens:
        for one_token in special_tokens:
            special_token_bytes.add(one_token.encode('utf-8'))
    # 进行实际的处理，如tokenization
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


def parallel_file_processing(filename: str,special_tokens: list[str] | None = None,num_processes: int = 4)-> dict[bytes, int]:
    """并行处理文件的完整示例"""
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries_fixed(f, num_processes, b"<|endoftext|>")
    
    # 创建块列表
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    print('\nchunks:',chunks)

    # 使用进程池并行处理
    with mp.Pool(num_processes) as pool:
        # 每个进程处理一个块
        results = pool.starmap(
            process_chunk,
            [(chunk, filename, special_tokens) for chunk in chunks]
        )
    
    # 合并结果
    total_tokens = add_dicts(results)
    print(f"预分词完成，找到 {len(total_tokens)} 种不同的token")
    
    return total_tokens

def single_process_for_comparison(
    filename: str, 
    special_tokens: list[str]
) -> dict[bytes, int]:
    """单进程版本用于对比"""
    with open(filename, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="ignore")
    
    special_token_bytes = set()
    if special_tokens:
        for one_token in special_tokens:
            special_token_bytes.add(one_token.encode('utf-8'))
    
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counts = {}
    
    for match in re.finditer(pattern, text):
        token_str = match.group()
        token_bytes = token_str.encode('utf-8')
        
        if token_bytes in special_token_bytes:
            continue
        
        token_counts[token_bytes] = token_counts.get(token_bytes, 0) + 1
    
    return token_counts

def compare_results(parallel_result, single_result):
    """比较并行和单进程结果"""
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    # 检查key数量
    parallel_keys = set(parallel_result.keys())
    single_keys = set(single_result.keys())
    
    print(f"并行结果token种类数: {len(parallel_keys)}")
    print(f"单进程结果token种类数: {len(single_keys)}")
    
    # 检查差异
    only_in_parallel = parallel_keys - single_keys
    only_in_single = single_keys - parallel_keys
    
    if only_in_parallel:
        print(f"\n只在并行结果中的token ({len(only_in_parallel)} 个):")
        for token in list(only_in_parallel)[:10]:  # 只显示前10个
            print(f"  {token} -> {parallel_result[token]}")
    
    if only_in_single:
        print(f"\n只在单进程结果中的token ({len(only_in_single)} 个):")
        for token in list(only_in_single)[:10]:  # 只显示前10个
            print(f"  {token} -> {single_result[token]}")
    
    # 检查共同token的计数差异
    common_keys = parallel_keys & single_keys
    differing_counts = []
    
    for key in common_keys:
        if parallel_result[key] != single_result[key]:
            differing_counts.append((
                key, 
                parallel_result[key], 
                single_result[key],
                parallel_result[key] - single_result[key]
            ))
    
    if differing_counts:
        print(f"\n计数不同的token ({len(differing_counts)} 个):")
        for token, p_count, s_count, diff in differing_counts[:10]:
            print(f"  {token}: 并行={p_count}, 单进程={s_count}, 差异={diff}")
    
    # 统计信息
    total_parallel = sum(parallel_result.values())
    total_single = sum(single_result.values())
    
    print(f"\n总计token数: 并行={total_parallel}, 单进程={total_single}")
    print(f"差异: {abs(total_parallel - total_single)} ({abs(total_parallel - total_single)/total_single*100:.2f}%)")
    
    return parallel_keys == single_keys and not differing_counts

# 使用示例
if __name__ == "__main__":
    start_time=time.perf_counter()
    total_tokens_multi = parallel_file_processing("/home/tao/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",["<|endoftext|>"],8)
    end_time = time.perf_counter()
    print(f"将大文件分块并且按照空格进行预分词耗时: {end_time-start_time:.9f} 秒")  # 纳秒精度
    print(f"token总数: {len(total_tokens_multi)}")
    print(f"前10个token及其计数: {list(total_tokens_multi.items())[:10]}")
    start_time=time.perf_counter()
    total_tokens_single = parallel_file_processing("/home/tao/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",["<|endoftext|>"],1)
    end_time = time.perf_counter()
    print(f"将大文件分块并且按照空格进行预分词耗时: {end_time-start_time:.9f} 秒")  # 纳秒精度
    print(f"token总数: {len(total_tokens_single)}")
    print(f"前10个token及其计数: {list(total_tokens_single.items())[:10]}")
    #assert total_tokens_multi == total_tokens_single
    for key,value in total_tokens_multi.items():
        if key not in total_tokens_single:
            print(f"Token {key} found in multi but not in single")
        if total_tokens_single[key] != value:
            print(f"Token {key} has count {value} in multi but {total_tokens_single[key]} in single")
    for key,value in total_tokens_single.items():
        if key not in total_tokens_multi:
            print(f"Token {key} found in single but not in multi")
        if total_tokens_multi[key] != value:
            print(f"Token {key} has count {value} in single but {total_tokens_multi[key]} in multi")


