import torch

a=torch.ones(4,4,4)
a[0][:]=torch.zeros(4,4)
print(torch.zeros(4,4).dtype)
