import os
from typing import BinaryIO
import multiprocessing as mp
from typing import List
import time
import regex as re
def add_dicts(dicts: list[dict]) -> dict:
    result={}
    for dict in dicts:
        for key, value in dict.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))





def process_chunk(start_end: tuple, filename: str,special_tokens: list[str] | None = None):
    """处理单个块的函数（可在不同进程中运行）"""
    print(start_end, filename,special_tokens)
    with open(filename, "rb") as f:
        start, end = start_end
        f.seek(start)
        chunk_data = f.read(end - start)
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
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 创建(start, end)对列表
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    print('chunks:',chunks)

    # 使用进程池并行处理
    with mp.Pool(num_processes) as pool:
        # 每个进程处理一个块
        results = pool.starmap(
            process_chunk,
            [(chunk, filename, special_tokens) for chunk in chunks]
        )

    # 汇总结果
    total_tokens_dicts = add_dicts(results)
    print("预分词完毕")
    return total_tokens_dicts

## Usage
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


