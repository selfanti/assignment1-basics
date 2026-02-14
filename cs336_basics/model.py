import torch.nn
from einops import rearrange, einsum
from jaxtyping import Float
from torch import Tensor
import math
class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int,device:torch.device|None=None,dtype:torch.dtype|None=None):
        """
        自定义线性层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            device: 设备 (CPU/GPU)
            dtype: 数据类型
        """
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        self.init_parameters()

    def init_parameters(self):
        '''
        初始化参数
        均值为0,方差为$2/(d_{in}+d_{out})$
        '''
        std=(2/(self.in_features+self.out_features))**0.5
        mean=0
        torch.nn.init.trunc_normal_(self.weight,mean=mean,std=std,a=-3*std,b=3*std)

    def forward(self,x:torch.Tensor):
        #assert len(x.shape)==2,"input should be 2D"
        return einsum(self.weight,x, " d_out d_in, ... d_in ->... d_out")
    
    def extra_repr(self) -> str:
        """用于打印模型信息的额外表示"""
        return f'in_features={self.in_features}, out_features={self.out_features}'
    

class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings:int, embedding_dim:int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weights=torch.nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))  
    def init_parameters(self):
        #Embedding: N (μ = 0, σ^2 = 1) truncated at [−3, 3]
        torch.nn.init.trunc_normal_(self.weights,0,1,-3,3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #batch sequence_length ->batch sequence_length d_model
        return self.weights[x]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device:torch.device | None=None, dtype:torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.gamma=torch.nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 缩放和平移
        return self.gamma * (x/rms)
class PositionWise_FeedForward(torch.nn.Module):
    def __init__(self,d_model:int,d_ff:int,device:torch.device|None=None,dtype:torch.dtype|None=None):
        '''
        $FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(SiLU(W_1x) ⊙ W_3x)$
        You should set dff to approximately 8/3 times dmodel in your implementation, 
        while ensuring that  the dimensionality of the inner feed-forward layer
        is a multiple of 64 to make good use of your hardware.
        '''
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1=torch.nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))
        self.w2=torch.nn.Parameter(torch.empty((d_model,d_ff),device=device,dtype=dtype))
        self.w3=torch.nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))
        self.init_parameters()
    def init_parameters(self):
        '''
        初始化参数
        均值为0,方差为$2/(d_{in}+d_{out})$
        '''
        std=(2/(self.d_ff+self.d_model))**0.5
        mean=0
        torch.nn.init.trunc_normal_(self.w1,mean=mean,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.w2,mean=mean,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.w3,mean=mean,std=std,a=-3*std,b=3*std)
    def forward(self,x):
        # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        # x: (..., d_model) -> w1@x: (..., d_ff), w3@x: (..., d_ff)
        w1_out = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_out = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        swiglu = w1_out*torch.sigmoid(w1_out) * w3_out
        return einsum(self.w2, swiglu, "d_model d_ff, ... d_ff -> ... d_model")

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        $q'^{(i)} = R^iq^{(i)} = R^iW_qx^{(i)}$ 
        and the same computation for k
        '''
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        
        # Create rotation matrices for all positions up to max_seq_len
        # Shape: (max_seq_len, d_k // 2)
        # Freqs: theta * base^(-2i/d_k) = theta^(-2i/d_k) for base=theta
        # For i in range(d_k // 2), freq = theta^(-2*i/d_k)
        freqs = torch.pow(theta, -2 * torch.arange(0, d_k // 2, device=device) / d_k)
        
        # Position indices: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device)
        
        # Outer product gives (max_seq_len, d_k // 2)
        # freq * position = theta * position / theta^(2i) = position / theta^(2i/d_k)
        freqs_expo = einsum(freqs, positions, "d2, seq -> seq d2")
        
        # cos and sin values: (max_seq_len, d_k // 2)
        self.register_buffer("cos_vals", torch.cos(freqs_expo), persistent=False)
        self.register_buffer("sin_vals", torch.sin(freqs_expo), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) or (seq_len,)
        
        # Get sequence length from x
        seq_len = x.shape[-2]
        
        # Expand token_positions to match x's leading dimensions if needed
        # Handle both (seq_len,) and (..., seq_len) cases
        if token_positions.dim() == 1:
            # (seq_len,) - expand to match x's leading dims
            expanded_shape = [1] * (x.dim() - 2) + list(token_positions.shape)
            pos_expanded = token_positions.view(expanded_shape)
        else:
            # Already has batch dims
            pos_expanded = token_positions
        
        # Get cos and sin for the specified positions
        # cos_vals, sin_vals: (max_seq_len, d_k // 2)
        # pos_expanded: (..., seq_len) -> after indexing: (..., seq_len, d_k // 2)
        cos_subset = self.cos_vals[pos_expanded]
        sin_subset = self.sin_vals[pos_expanded]
        
        # x_even: (..., seq_len, d_k // 2) - even indices 0, 2, 4, ...
        # x_odd: (..., seq_len, d_k // 2) - odd indices 1, 3, 5, ...
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation:
        # x_rotated_even = x_even * cos - x_odd * sin
        # x_rotated_odd = x_even * sin + x_odd * cos
        
        x_rotated_even = x_even * cos_subset - x_odd * sin_subset
        x_rotated_odd = x_even * sin_subset + x_odd * cos_subset
        
        # Interleave even and odd back together
        # x_rotated: (..., seq_len, d_k)
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to a tensor along a specified dimension with numerical stability.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
    
    Returns:
        Tensor with same shape as input, with softmax applied along dim
    """
    # Subtract max for numerical stability
    # keepdim=True maintains the dimension for correct broadcasting
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    # Compute exp(x - max(x))
    exp_x = torch.exp(x - max_vals)
    # Normalize by sum along the dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Tensor of shape (batch_size, num_heads, seq_len_q, d_v) containing the attention output
    """
    d_k = Q.shape[-1]
    
    # Compute scaled dot product: Q @ K.T
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / (d_k ** 0.5)
    # Apply mask if provided
    if mask is not None:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
        assert mask.shape == scores.shape, f"Mask shape {mask.shape} must match scores shape {scores.shape}"

        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)
    
    # Compute attention output: attn_weights @ V
    output = einsum(attn_weights, V,"... q k, ... k d -> ... q d")
    return output
class multihead_self_attention(torch.nn.Module):
    def __init__(self,d_model:int,num_heads:int,if_rope=False,theta=None,max_seq_len=None):
        super().__init__()
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.if_rope=if_rope
        self.theta=theta
        self.max_seq_len=max_seq_len
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj_weight=torch.nn.Parameter(torch.empty((d_model,d_model)))
        self.k_proj_weight=torch.nn.Parameter(torch.empty((d_model,d_model)))
        self.v_proj_weight=torch.nn.Parameter(torch.empty((d_model,d_model)))
        self.o_proj_weight=torch.nn.Parameter(torch.empty((d_model,d_model)))
        self.init_parameters()
    def init_parameters(self):
        '''
        初始化参数
        均值为0,方差为$2/(d_{in}+d_{out})$
        '''
        std=(2/(self.d_model+self.d_model))**0.5
        mean=0
        torch.nn.init.trunc_normal_(self.q_proj_weight,mean=mean,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.k_proj_weight,mean=mean,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.v_proj_weight,mean=mean,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.o_proj_weight,mean=mean,std=std,a=-3*std,b=3*std)
        if self.if_rope and self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_model//self.num_heads, self.max_seq_len)
    def forward(self,x,token_positions=None):
        
        Q = x @ self.q_proj_weight.T
        K = x @ self.k_proj_weight.T
        V = x @ self.v_proj_weight.T
        d_k=self.d_model//self.num_heads
        # Reshape Q, K, V for multi-head attention
        Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        K = rearrange(K, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        V = rearrange(V, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        if self.if_rope:
            assert self.theta is not None and self.max_seq_len is not None, "theta, max_seq_len, and token_positions must be provided for RoPE"
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        # Compute attention for each head
        attn_output = scaled_dot_product_attention(Q, K, V)
        print('attn_output.shape:',attn_output.shape)
        # Concatenate heads and project back to d_model
        attn_output = rearrange(attn_output, "... head seq d_k -> ... seq (head d_k)", head=self.num_heads, d_k=d_k)
        out = attn_output @ self.o_proj_weight.T
        return out

if __name__=='__main__':
    linear_layer=Linear(3,3)
    input=torch.randn(3,1)
    output=linear_layer(input)
    print(output)
    print(linear_layer.extra_repr())
    print(linear_layer.state_dict())
