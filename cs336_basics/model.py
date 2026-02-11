import torch.nn
from einops import rearrange, einsum
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
        均值为0,方差为2/(d_in+d_out)
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
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
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
        均值为0,方差为2/(d_in+d_out)
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

if __name__=='__main__':
    linear_layer=Linear(3,3)
    input=torch.randn(3,1)
    output=linear_layer(input)
    print(output)
    print(linear_layer.extra_repr())
    print(linear_layer.state_dict())