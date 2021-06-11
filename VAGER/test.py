import torch
import torch.nn as nn

def main():
    loss = nn.MSELoss()
    a = torch.tensor([1, 2, 3, 4]).to(float)
    b = torch.tensor([3]).to(float)
    print(loss(a, b))

if __name__ == '__main__':
    main()