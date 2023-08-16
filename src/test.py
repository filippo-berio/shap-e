import torch
import time

###CPU
start_time = time.time()
a = torch.ones(4,4)
for _ in range(1000000):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(4,4).cuda()
for _ in range(1000000):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)