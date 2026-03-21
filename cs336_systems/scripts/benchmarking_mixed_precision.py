from torch import nn
import torch
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("After fc1:", x.dtype)   # 在 autocast 下应为 float16 或 bfloat16
        x = self.relu(x)
        print("After relu:", x.dtype)  # 激活函数通常保持输入类型
        x = self.ln(x)
        print("After layer norm:", x.dtype)  # LayerNorm 会被提升为 float32
        x = self.fc2(x)
        print("After fc2:", x.dtype)   # 线性层输出为低精度（float16/bfloat16）
        return x
model = ToyModel(in_features=20, out_features=5).cuda()
x = torch.randn(2, 20).cuda()

with torch.autocast(device_type="cuda",dtype=torch.bfloat16):   # 或 torch.bfloat16
    output = model(x)