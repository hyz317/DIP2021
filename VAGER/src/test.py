import torch
import random

a = list(range(0, 100))
b = list(range(0, 100))
randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(a)
random.seed(randnum)
random.shuffle(b)

print(a)
print(b)