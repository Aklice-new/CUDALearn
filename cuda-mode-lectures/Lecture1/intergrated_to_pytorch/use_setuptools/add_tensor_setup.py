import torch
import add_cuda  # 这个就是通过setup安装的包的名字，其函数是通过pybind来绑定的


a = torch.arange(20, dtype=torch.float32).reshape(4,5).cuda()
b = torch.randn((4, 5), dtype=torch.float32).cuda()

print("================")
print("torch implement tensor add")
print("================")
print(a + b)

print("================")
print("cpp/cuda implement tensor add")
print("================")
print(add_cuda.forward(a, b))
