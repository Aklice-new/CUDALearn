import torch

def time_pytorch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # warm up
    for i in range(10):
        func(input)
    
    start.record()
    func(input)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)

def square_2(input):
    return input ** 2

def square_3(input):
    return input * input

b = torch.randn((10000, 10000)).cuda()

print(time_pytorch_function(torch.square, b))
print(time_pytorch_function(square_2, b))
print(time_pytorch_function(square_3, b))

print("=============")
print("Profiling torch.square")
print("=============")

# profile functions by torch.auto.profiler

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

# profile functions by torch.auto.profiler

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

# profile functions by torch.auto.profiler

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
