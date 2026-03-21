import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import autocast

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


DEFAULT_RESULTS_FILE = Path(__file__).resolve().parent.parent / "benchmark_results.csv"
MEMORY_SNAPSHOT_FILE = Path.cwd() / "memory_snapshot.pickle"
RESULT_COLUMNS = [
    "timestamp",
    "mode",
    "model_size",
    "batch_size",
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
    "device",
    "autocast_bf16",
    "warmup_steps",
    "run_steps",
    "mean_seconds",
    "variance_seconds",
    "memory_snapshot",
]


@dataclass(frozen=True)
class ModelSize:
    size: Literal["small", "medium", "large", "xl", "2.7B", "default"]
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODELS = {
    "default": ModelSize("default", 512, 1344, 4, 16),
    "small": ModelSize("small", 768, 3072, 12, 12),
    "medium": ModelSize("medium", 1024, 4096, 24, 16),
    "large": ModelSize("large", 1280, 5120, 36, 20),
    "xl": ModelSize("xl", 1600, 6400, 48, 25),
    "2.7b": ModelSize("2.7B", 2560, 10240, 32, 32),
}


def stringify_result_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\n", " ")


def render_markdown_table(rows: list[dict[str, object]]) -> str:
    dataframe = pd.DataFrame(
        [{column: stringify_result_value(row.get(column, "")) for column in RESULT_COLUMNS} for row in rows],
        columns=RESULT_COLUMNS,
    )
    return dataframe.to_markdown(index=False)


def load_results_dataframe(results_file: Path) -> pd.DataFrame:
    if not results_file.exists() or results_file.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    dataframe = pd.read_csv(results_file)
    dataframe = dataframe.reindex(columns=RESULT_COLUMNS, fill_value="")
    for column in RESULT_COLUMNS:
        dataframe[column] = dataframe[column].map(stringify_result_value)
    return dataframe


def append_result_row(results_file: Path, row: dict[str, object]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    existing_dataframe = load_results_dataframe(results_file)
    new_row = pd.DataFrame(
        [{column: stringify_result_value(row.get(column, "")) for column in RESULT_COLUMNS}],
        columns=RESULT_COLUMNS,
    )
    results_dataframe = pd.concat([existing_dataframe, new_row], ignore_index=True).fillna("")
    results_dataframe.to_csv(results_file, index=False)


def maybe_synchronize(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def maybe_start_memory_history(enabled: bool, device_type: str) -> None:
    if enabled and device_type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, device_type: str) -> str | None:
    if not enabled or device_type != "cuda":
        return None
    torch.cuda.memory._dump_snapshot(str(MEMORY_SNAPSHOT_FILE))
    torch.cuda.memory._record_memory_history(enabled=None)
    return str(MEMORY_SNAPSHOT_FILE)


def build_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str,
) -> TransformerLM:
    return TransformerLM(
        vocab_size,
        context_length,
        d_model,
        d_ff,
        num_layers,
        num_heads,
        rope_theta,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, help="the size of the model")
    parser.add_argument("--autocast_bf16", action="store_true", help="auto cast the forward")
    parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocaulary size")
    parser.add_argument("--context_length", type=int, default=256, help="context size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer block")
    parser.add_argument("--num_heads", type=int, default=16, help="number of head")
    parser.add_argument("--d_ff", type=int, default=1344, help="dimension of the ffn")
    parser.add_argument("--memory_profiler", action="store_true", help="use the memory profiler")
    parser.add_argument("--rope_theta", type=float, default=10000, help="rope theta ")
    parser.add_argument("--device", type=str, default="cuda:0", help="device of the training")
    parser.add_argument("--warmup_steps", type=int, default=10, help="steps of warm up")
    parser.add_argument("--run_steps", type=int, default=20, help="steps of measurement")
    parser.add_argument("--forward_only", action="store_true", help="only for the time of running the forward")
    parser.add_argument(
        "--results_file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=f"CSV file used to persist benchmark results, default: {DEFAULT_RESULTS_FILE}",
    )

    args = parser.parse_args()

    model_size = MODELS[args.model_size.lower()] if args.model_size else None
    device = torch.device(args.device)
    device_type = device.type
    resolved_d_model = model_size.d_model if model_size else args.d_model
    resolved_num_layers = model_size.num_layers if model_size else args.num_layers
    resolved_num_heads = model_size.num_heads if model_size else args.num_heads
    resolved_d_ff = model_size.d_ff if model_size else args.d_ff
    model_label = model_size.size if model_size else "custom"
    mode = "forward_only" if args.forward_only else "forward_backward"

    if args.autocast_bf16 and device_type != "cuda":
        print("warning: --autocast_bf16 only applies to CUDA devices; running without autocast.")
    if args.memory_profiler and device_type != "cuda":
        print("warning: --memory_profiler only applies to CUDA devices; no snapshot will be generated.")

    data = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    autocast_context = (
        autocast(device_type="cuda", dtype=torch.bfloat16) if args.autocast_bf16 and device_type == "cuda" else nullcontext()
    )
    model = build_model(
        args.vocab_size,
        args.context_length,
        resolved_d_model,
        resolved_num_layers,
        resolved_num_heads,
        resolved_d_ff,
        args.rope_theta,
        args.device,
    )
    optimizer = AdamW(model.parameters())

    if args.forward_only:
        model.eval()
        times: list[float] = []
        with torch.no_grad():
            for _ in range(args.warmup_steps):
                with autocast_context:
                    model(data)

            maybe_synchronize(device_type)
            maybe_start_memory_history(args.memory_profiler, device_type)
            last_time = default_timer()
            for _ in range(args.run_steps):
                with autocast_context:
                    model(data)
                maybe_synchronize(device_type)
                time_point = default_timer()
                times.append(time_point - last_time)
                last_time = time_point
    else:
        model.train()
        times = []
        for _ in range(args.warmup_steps):
            with autocast_context:
                logits = model(data)
            loss = cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        maybe_synchronize(device_type)
        maybe_start_memory_history(args.memory_profiler, device_type)
        last_time = default_timer()
        for _ in range(args.run_steps):
            with autocast_context:
                logits = model(data)
            loss = cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            maybe_synchronize(device_type)
            time_point = default_timer()
            times.append(time_point - last_time)
            last_time = time_point

    mean_seconds = float(np.mean(times))
    variance_seconds = float(np.var(times))
    memory_snapshot = maybe_dump_memory_snapshot(args.memory_profiler, device_type)
    result_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "model_size": model_label,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": resolved_d_model,
        "num_layers": resolved_num_layers,
        "num_heads": resolved_num_heads,
        "d_ff": resolved_d_ff,
        "rope_theta": args.rope_theta,
        "device": args.device,
        "autocast_bf16": args.autocast_bf16 and device_type == "cuda",
        "warmup_steps": args.warmup_steps,
        "run_steps": args.run_steps,
        "mean_seconds": mean_seconds,
        "variance_seconds": variance_seconds,
        "memory_snapshot": memory_snapshot,
    }
    append_result_row(args.results_file, result_row)

    print(render_markdown_table([result_row]))
    print(f"results appended to: {args.results_file}")


if __name__ == "__main__":
    main()
