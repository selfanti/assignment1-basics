# CLAUDE.md

This file provides guidance to Claude ​Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_nn_utils.py

# Run a specific test
uv run pytest tests/test_nn_utils.py::test_linear -s

# Run any Python file in the repo (auto-activates the venv)
uv run <python_file_path>
```

The project uses `uv` for environment management. Do not use `pip` or activate the venv manually.

## Architecture

This is a CS336 assignment implementing a GPT-style transformer language model from scratch.

### Key files

- **`cs336_basics/model.py`** — Core model components: `Linear`, `Embedding`, `RMSNorm`, `PositionWise_FeedForward` (SwiGLU), `RotaryPositionalEmbedding`, `multihead_self_attention`, `softmax`, `scaled_dot_product_attention`. Missing: `TransformerBlock` and full `TransformerLM` (still `raise NotImplementedError` in adapters).
- **`cs336_basics/tokenizer.py`** — BPE tokenizer class with GPT-2-style pre-tokenization (via `regex` library), a Trie for vocabulary lookup, merge-order dict for O(1) BPE application, and `encode`/`decode`/`encode_iterable` methods.
- **`cs336_basics/train_bpe.py`** — `train_bpe()`: trains BPE from a corpus; uses `parallel_file_processing` from `pretokenization.py`.
- **`cs336_basics/pretokenization.py`** — Parallel corpus pre-tokenization using `multiprocessing`. Splits files at `<|endoftext|>` boundaries, processes chunks with worker processes.
- **`tests/adapters.py`** — Bridge between tests and implementations. Each `run_*` function instantiates the relevant class, loads weights via `load_state_dict`, and calls forward. **This is where you connect your implementations to the test suite** — complete the `raise NotImplementedError` stubs here.

### Test files

- `tests/test_nn_utils.py` — Tests Linear, Embedding, RMSNorm, SwiGLU, attention, RoPE, TransformerBlock, TransformerLM
- `tests/test_train_bpe.py` — Tests BPE training
- `tests/test_data.py` — Tests data loading (`get_batch`)
- `tests/test_optimizer.py` — Tests AdamW, gradient clipping, LR schedule
- `tests/test_serialization.py` — Tests checkpoint save/load

### Model architecture

The transformer follows a pre-norm design:
- Token embedding → N × TransformerBlock → RMSNorm → LM head (weight-tied with embedding)
- Each block: RMSNorm → MHA (with RoPE) → residual + RMSNorm → SwiGLU FFN → residual
- FFN uses SwiGLU: `W2(SiLU(W1x) ⊙ W3x)`, with `d_ff ≈ (8/3) * d_model` rounded to multiple of 64
- Linear layers initialized with truncated normal: σ = √(2/(d_in + d_out)), truncated at ±3σ
- No bias terms anywhere

### State dict key conventions

The adapter functions in `tests/adapters.py` show the expected weight key names for each module. For TransformerBlock: `attn.q_proj.weight`, `attn.k_proj.weight`, `attn.v_proj.weight`, `attn.output_proj.weight`, `ln1.weight`, `ln2.weight`, `ffn.w1.weight`, `ffn.w2.weight`, `ffn.w3.weight`.

### Still unimplemented (as of latest commit)

The following adapter stubs still raise `NotImplementedError`:
- `run_transformer_block` / `run_transformer_lm`
- `run_silu`, `run_get_batch`, `run_cross_entropy`
- `run_gradient_clipping`, `get_adamw_cls`, `run_get_lr_cosine_schedule`
- `run_save_checkpoint`, `run_load_checkpoint`
