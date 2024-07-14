#  Lecture 1

第一节课的目标是：
1. 如何将cuda程序内嵌到pytorch中
2. 对kernel进行profile



## 把CUDA集成到pytorch中

###  Pytorch

视频中提到的方法是通过Pytorch的load_inline方法可以直接将C/C++的源代码以函数的方式来加载到python程序中。

但实际上，pytorch提供了更多的方法将C++/CUDA的代码集成到python中。加载的方法主要分为两种:
- setuptools：通过提前构建的方式，将C++/CUDA的代码编译成动态链接库，然后通过pybind将C++的接口和python进行绑定，最后通过setuptools将其打包成python的包。
- JIT(just-in-time)即时编译：通过torch.utils.cpp_extension.load()(或者load_inline就是视频中提到的方法，直接加载源程序的str)。这种方法是在运行时动态编译C++/CUDA代码，同样完成的是setuptools的工作，但是它免去了维护setup.py的麻烦。在load()中，我们向函数提供与setuptools相同的信息。在后台，这将执行以下操作：
    - 创建临时目录/tmp/torch_extensions/op_name
    - 将一个Ninja构建文件输出到该临时目录，
    - 编译您的源文件为一个共享库，
    - 将这个共享库导入为一个Python模块。


注意：
使用setuptools时，C++需要使用CPPExtenstion,CUDA需要使用CUDAExtension。
而JIT的方法可以直接在load时指定cppsource,cudasource。

### Triton

Triton是openai针对gpu上的算子优化提出的一个programming language & compiler。以NVIDIA GPU为例，使用triton可以越过cuda的闭源生态，直接将自己的后端接入llvm IR，通过走NVPTX来生成在GPU上的代码。

可以直接通过编写python代码(triton kernel)，然后来让Triton进行编译，进而优化出性能表现高的CUDA kernel。

- Triton Debugger
    triton debugger可以支持在triton kernel中打断点来进行测试。

## Perf
### Pytorch Profiler

[pt_profiler.py](./pt_profiler.py)

通过torch.autograd.profiler.profile(use_cuda=True)对cuda进行相关性能分析。

### ncu Profiler

ncu(Nsight Compute)是cuda提供的一个用于性能分析的工具，用它来分析cuda kernel可以得到一些改进建议，重要的是能看到gpu上各个部分的使用率，从而从不同的角度对kernel进行优化，如L2 cache、shared memory等.


