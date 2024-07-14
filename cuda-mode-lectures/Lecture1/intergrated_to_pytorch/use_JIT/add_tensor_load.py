import torch
from torch.utils.cpp_extension import load




add_tensor_cpp_extension = load(
    name='add_tensor_function',
    sources=['./cuda/add_tensor.cpp',
             './cuda/add_tensor.cu']
)

a = torch.arange(20, dtype=torch.float32).reshape(4,5).cuda()
b = torch.randn((4, 5), dtype=torch.float32).cuda()

print("================")
print("torch implement tensor add")
print("================")
print(a + b)

print("================")
print("cpp/cuda implement tensor add")
print("================")
print(add_tensor_cpp_extension.forward(a, b))


print("================")
print("cpp/cuda implement add backward")
print("================")
grad = torch.ones_like(a)
grad_a, grad_b = add_tensor_cpp_extension.backward(grad)

print(grad_a, grad_b)