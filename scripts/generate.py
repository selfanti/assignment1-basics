import argparse
import pickle
from pathlib import Path

import readline  # noqa: F401  # 保留 readline 只是为了让终端输入自动拥有历史编辑能力。
import torch

from cs336_basics.data import load_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.radix_attention import RadixAttentionCache, generate_with_radix_attention
from cs336_basics.tokenizer import Tokenizer


DEFAULT_CHECKPOINT = Path("/home/tao/assignment1-basics/data/datasets/tokens_train/checkpoint_epoch_30000.pt")
DEFAULT_VOCAB_PATH = Path("/home/tao/assignment1-basics/data/datasets/vocab.pkl")
DEFAULT_MERGES_PATH = Path("/home/tao/assignment1-basics/data/datasets/merges.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-user multi-turn interactive generation demo backed by radix-managed KV cache.",
    )
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT, help="checkpoint to load")
    parser.add_argument("--vocab_path", type=Path, default=DEFAULT_VOCAB_PATH, help="tokenizer vocab pickle")
    parser.add_argument("--merges_path", type=Path, default=DEFAULT_MERGES_PATH, help="tokenizer merges pickle")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p threshold")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="maximum assistant tokens per turn")
    parser.add_argument("--device", type=str, default="cuda:0", help="device used for generation")
    parser.add_argument("--vocab_size", type=int, default=10000, help="model vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="trained context length")
    parser.add_argument("--d_model", type=int, default=512, help="model hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="feed-forward hidden size")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="standard",
        choices=("standard", "flash_attention_v2"),
        help="attention implementation to use when initializing the model",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a concise assistant.",
        help="system prompt prepended to every turn",
    )
    return parser.parse_args()


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    # tokenizer 文件和 checkpoint 一样都是外部产物；显式检查路径能让脚本在启动阶段尽早失败，
    # 避免进入交互循环后才因为文件缺失中断。
    with vocab_path.open("rb") as vocab_file:
        vocabs = pickle.load(vocab_file)
    with merges_path.open("rb") as merges_file:
        merges = pickle.load(merges_file)
    return Tokenizer(vocabs, merges, ["<|endoftext|>"])


def build_model(args: argparse.Namespace) -> TransformerLM:
    # attention_backend 通过模型构造函数一路传到每个 attention 层，
    # 这样交互脚本和 benchmark.py 能复用同一个初始化开关。
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.d_ff,
        args.num_layers,
        args.num_heads,
        args.rope_theta,
        device=args.device,
        attention_backend=args.attention_backend,
    ).to(args.device)
    optimizer = AdamW(model.parameters())
    load_checkpoint(args.checkpoint_path, model, optimizer)
    model.eval()
    return model


def format_conversation(system_prompt: str, conversation: list[tuple[str, str]]) -> str:
    # 这里故意把多轮对话格式化成稳定、显式的纯文本模板:
    # 1. radix cache 的命中依赖“新 prompt 是旧 prompt 的前缀扩展”；
    #    稳定模板可以保证同一段历史在不同轮次不会因为分隔符飘动而丢失前缀复用机会。
    # 2. 这种模板也方便人工检查 tokenizer 编码后的总长度是否接近 context_length。
    turns: list[str] = []
    if system_prompt.strip():
        turns.append(f"System: {system_prompt.strip()}\n")
    for role, content in conversation:
        turns.append(f"{role}: {content.strip()}\n")
    turns.append("Assistant: ")
    return "".join(turns)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
    model = build_model(args)
    radix_cache = RadixAttentionCache(model.context_length)
    conversation: list[tuple[str, str]] = []

    print(f"Loaded model on {args.device} with attention_backend={model.attention_backend}.")
    print("Commands: /reset clears the conversation, /quit exits.")

    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            print("Exiting interactive session.")
            break
        if user_input == "/reset":
            conversation.clear()
            radix_cache.clear()
            print("Conversation and radix cache cleared.")
            continue

        conversation.append(("User", user_input))
        prompt_text = format_conversation(args.system_prompt, conversation)
        prompt_tokens = tokenizer.encode(prompt_text)

        # 用户已经给出前提: 上下文不会超过训练窗口。
        # 因此这里不做滑窗或截断，而是在检测到超限时直接提示用户 reset，
        # 避免“历史被静默裁掉”导致 radix cache 命中和对话语义同时失真。
        if len(prompt_tokens) > model.context_length:
            conversation.pop()
            print(
                "Prompt length exceeds the trained context length. "
                f"prompt_tokens={len(prompt_tokens)}, context_length={model.context_length}. "
                "Use /reset or shorten the conversation."
            )
            continue

        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=args.device)
        try:
            generation = generate_with_radix_attention(
                model,
                prompt_tensor,
                radix_cache,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except ValueError as exc:
            # 如果本轮生成让长度超限，撤回刚加入的 user turn，保持对话状态和 radix tree 一致。
            conversation.pop()
            print(f"Generation aborted: {exc}")
            continue

        assistant_text = tokenizer.decode(generation.generated_tokens.tolist()).strip()
        conversation.append(("Assistant", assistant_text))
        print(f"[radix reused {generation.reused_prefix_length} prompt tokens]")
        print(f"Assistant> {assistant_text}\n")


if __name__ == "__main__":
    main()
