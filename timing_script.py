import torch
import time

in_row, in_f, out_f = 256, 1024, 2048
loop_times = 10000

# CPU
s = time.time()
tensor = torch.randn(in_row, in_f).to('cpu')
l_trans = torch.nn.Linear(in_f, out_f).to('cpu')
for _ in range(loop_times):
    l_trans(tensor)

print('CPU time:',time.time() - s)

# GPU (NVIDIA A10 24GB)
s = time.time()
tensor = torch.randn(in_row, in_f).cuda()
l_trans = torch.nn.Linear(in_f, out_f).cuda()
for _ in range(loop_times):
    l_trans(tensor)

torch.cuda.synchronize()
print('CUDA time:',time.time() - s)

# GPU with Tensor Cores
s = time.time()
tensor = torch.randn(in_row, in_f).cuda().half()
l_trans = torch.nn.Linear(in_f, out_f).cuda().half()
for _ in range(loop_times):
    l_trans(tensor)
    
torch.cuda.synchronize()
print('CUDA with Tensor Cores time:',time.time() - s)


# CPU time: 110.57620406150818
# CUDA time: 2.7178804874420166
# CUDA with Tensor Cores time: 0.217362642288208