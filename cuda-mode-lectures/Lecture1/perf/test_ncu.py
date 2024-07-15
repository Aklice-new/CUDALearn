import torch



a = torch.randn((10000, 10000), dtype=torch.float32).cuda()
b = torch.randn((10000, 10000), dtype=torch.float32).cuda()


res = torch.matmul(a, b)

print(res)