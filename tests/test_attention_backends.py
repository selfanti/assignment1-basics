import pytest
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.radix_attention import RadixAttentionCache, generate_with_radix_attention


class _ToyRadixAwareModel(torch.nn.Module):
    def __init__(self, context_length: int = 16, vocab_size: int = 16):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.processed_positions: list[int] = []

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        kv_cache=None,
        use_kv_cache: bool = False,
    ):
        # 这条 toy model 不关心语言建模质量，只把“哪些绝对位置被重新前向了”记录下来。
        # 测试的目标是验证 radix cache 是否真的跳过了共享前缀，而不是验证某个具体采样分布。
        assert token_positions is not None
        self.processed_positions.extend(token_positions.squeeze(0).tolist())

        logits = torch.full(
            (x.shape[0], x.shape[1], self.vocab_size),
            -1e9,
            device=x.device,
        )
        logits[:, -1, 7] = 0.0

        if not use_kv_cache:
            return logits

        previous_length = 0 if kv_cache is None else kv_cache[0]["k"].shape[-2]
        total_length = previous_length + x.shape[1]
        next_cache = [{
            "k": torch.zeros(1, 1, total_length, 1, device=x.device),
            "v": torch.zeros(1, 1, total_length, 1, device=x.device),
        }]
        return logits, next_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="manual Triton flash attention requires CUDA")
def test_flash_attention_backend_matches_standard_attention():
    torch.manual_seed(0)
    device = torch.device("cuda")
    standard_model = TransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        rope_theta=10000.0,
        attention_backend="standard",
    ).to(device)
    flash_model = TransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        rope_theta=10000.0,
        attention_backend="flash_attention_v2",
    ).to(device)
    flash_model.load_state_dict(standard_model.state_dict())
    standard_model.eval()
    flash_model.eval()

    tokens = torch.tensor([list(range(16))], device=device)

    # 这条测试确保“后端切换”只影响执行路径和性能，不影响模型语义。
    # 同一组权重下，普通 attention 与 FlashAttention-v2 风格路径应该给出同样的 logits。
    standard_logits = standard_model(tokens)
    flash_logits = flash_model(tokens)

    torch.testing.assert_close(flash_logits, standard_logits, atol=1e-5, rtol=1e-5)


def test_generate_with_radix_attention_reuses_cached_prefix():
    model = _ToyRadixAwareModel(context_length=16)
    radix_cache = RadixAttentionCache(model.context_length)

    first_prompt = torch.tensor([11, 12, 13], dtype=torch.long)
    first_result = generate_with_radix_attention(model, first_prompt, radix_cache, max_tokens=1)

    assert first_result.generated_tokens.tolist() == [7]
    assert first_result.reused_prefix_length == 0
    assert model.processed_positions == [0, 1, 2, 3]

    second_prompt = torch.tensor([11, 12, 13, 20], dtype=torch.long)
    assert radix_cache.match(second_prompt).matched_length == 3

    previous_call_count = len(model.processed_positions)
    second_result = generate_with_radix_attention(model, second_prompt, radix_cache, max_tokens=1)

    # 第二轮 prompt 共享前缀 [11, 12, 13]。
    # 如果 radix cache 生效，本轮只需要重新计算位置 3 的新 user token，
    # 然后再计算位置 4 的新 assistant token；不会重新跑 0,1,2 这三个旧位置。
    assert second_result.generated_tokens.tolist() == [7]
    assert second_result.reused_prefix_length == 3
    assert model.processed_positions[previous_call_count:] == [3, 4]
