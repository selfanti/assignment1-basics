import torch
x=torch.tensor([[1,3,4],[4,5,6]])
print(x.shape)
print(x.max(dim=-1,keepdim=False))