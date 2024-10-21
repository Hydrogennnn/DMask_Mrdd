
import torch

a=torch.ones(4,4,4)

b=[[a for _ in range(3)] for j in range(3)]
print(torch.count_nonzero(a[1][1]))
a[1][1]=torch.zeros(a[1][1].shape)

print(torch.count_nonzero(a[1][1]))
