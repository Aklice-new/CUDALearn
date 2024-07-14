from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name = "add_tensor_cpp_extension",
    ext_modules=[
        CUDAExtension('add_cuda', [   # 这个名字就是包的名字,cuda文件构建需要用CUDAExtension
            './cuda/add_tensor.cpp',
            './cuda/add_tensor_.cu' # 注意：setuptools无法处理具有相同名称但扩展名不同的文件
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })