import os
import math
import itertools

# 固定参数
TOTAL_TOKENS = 327_680_000
CONTEXT_LENGTH = 256

# 搜索空间
learning_rates = [1.5e-4, 3e-4, 6e-4, 1e-3]
batch_sizes = [16, 32, 64, 128]

dataset_path = "/home/vipuser/assignment1/data/tinystories/tokens.train/tokens.npy"  # 改成你的路径

def compute_epochs(batch_size):
    steps = TOTAL_TOKENS / (batch_size * CONTEXT_LENGTH)
    return int(steps)

def main():
    combinations = list(itertools.product(learning_rates, batch_sizes))

    for i, (lr, bs) in enumerate(combinations):
        epochs = compute_epochs(bs)

        run_name = f"exp_{i}_lr{lr}_bs{bs}_ep{epochs}"

        cmd = f"""
        python train.py {dataset_path} \
            --batch_size {bs} \
            --lr {lr} \
            --epochs {epochs} \
            --context_length {CONTEXT_LENGTH} \
            --device cuda:0
        """

        print(f"\n🚀 Running {run_name}")
        os.system(cmd)

if __name__ == "__main__":
    main()