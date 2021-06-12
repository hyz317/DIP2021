import torch
import random

c = torch.zeros((10, 4096))
a = torch.zeros((1000, 4096))
b = torch.mean(a, dim=0)
c[0] = b

print(a)
print(b)
print(c)