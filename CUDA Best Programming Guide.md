# CUDA Best Programming Guide

## Chapter 5 Programming Model

## 5.2.1 Thread Block Clusters 线程块集群

在compute capability 9.0以上的设备提供了一种线程块集群的层次，与在SM上协同处理相似，集群中的线程块也能保证在GPU的GPU集群上保证协同。

集群中线程块的组织方式和block内部一样，也能被组织为一维、二维或者三维的，但是需要注意的是CUDA中最多支持8个线程块集群。

![5_2_1_figure1](assert\CUDA_BEST_PROGRAMMING\5_2_1_figure1.png)

启动线程块集群有两种方式：一种昂是在编译期间，通过指定kernel的属性，使用\__cluster__dims(X, Y, Z)来指定；另一种是通过CUDA提供的APIcudaLaunchKernelEx来使用。

下面是第一种在编译期间确定的

```cpp
__global__ void __cluster_dims(2, 1, 1) cluster_kernel(float * input, float * output)
{

}
int main()
{
    dim3 threadperblock(16, 16);
    dim3 numblocks(N / threadperblock.x, N / thread.perblock.y);
    // grid size并不会因为集群而受到影响，仍然正常按照block进行计算
    // 但是grid dimension必须能够整除cluster dims
    cluster_kernel<<<threadperblock, numblocks>>>(intput, output);
}
```

下面是通过调用cuda API来进行的：

```cpp
__global__ void cluster_kernel(float * intput , float* output)
{

}

int main()
{
    ∕∕ Kernel invocation with runtime cluster size 
    { 
        cudaLaunchConfig_t config = {0}; 
        ∕∕ The grid dimension is not affected by cluster launch, and is still enumerated 
        ∕∕ using number of blocks. 
        ∕∕ The grid dimension should be a multiple of cluster size. 
        config.gridDim = numBlocks; 
        config.blockDim = threadsPerBlock; 
        cudaLaunchAttribute attribute[1]; 
        attribute[0].id = cudaLaunchAttributeClusterDimension; 
        attribute[0].val.clusterDim.x = 2; 
        ∕∕ Cluster size in X-dimension 
        attribute[0].val.clusterDim.y = 1; 
        attribute[0].val.clusterDim.z = 1; 
        config.attrs = attribute; 
        config.numAttrs = 1; 
        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

集群中的线程可以在单个GPU Processing Cluster中协同调度，可以通过cluster.sync()进行硬件同步，同时还提供了一些查询的相关函数。

集群中的线程块还能访问分布式共享内存，后面在Distributed Shared Memory中介绍到。

## Chapter 10 C++ Language Extensions

### 10.1 函数执行的一些关键字

\__global__ ： host端启动， device端执行

\__device__ ： 在device端调用并执行

\__host__： 在 host端调用并执行

\__device__  \__host__: 表示在host和device都可以执行调用

\__CUDA_ARCH__   : 可以用于区分编译主机代码还是设备代码。当在设备代码中使用是，它表示一个计算能力，但是当在主机代码中使用时，它没有被定义，所以可以用来区分编译主机代码还是设备代码。

```cpp
#ifndef __CUDA_ARCH__
//host code here
#else
//device code here
#endif
```

编译器会自动在合适的情况下将\__device__函数进行内联。

\__noinline__ : 表示强制不将改函数内联

\__forceinline__: 表示强制将该函数内联

\__inline_hint__: 表示尽可能将函数内联，但不是一定内联********

### 10.2 变量的一些关键字

在设备端定义的一些变量，如果没有被\__devcice__、\__shared__、\__constant__这些关键字修饰的话，一般情况会保存到寄存器中，有些情况会被保存在local memory中。

\__device__:只被该词修饰的变量会存在于全局内存中，并且需要用cudaGetSymbolAddredd()等进行访问和操作。但是它之后可以再跟一个修饰词，更进一步表示它的具体位置。 

\____shared____:

\____constant__:

\__grid_constant__: 从compute capability 7.0开始，用于注释 global函数中的常量非引用类型。他表示该变量：

- 生命周期为这个grid；
- 是该grid私有的，不能被主机线程或者其他grid进行访问
- 每个网格中的该常量都是独有的，且是read-only

```cpp
__device__ void unknown_function(S const&); 
__global__ void kernel(const __grid_constant__ S s) {  //这里只能修饰常量，即被const修饰的变量
    s.x += threadIdx.x; ∕∕ Undefined Behavior: tried to modify read-only memory 
    ∕∕ Compiler will _not_ create a per-thread thread local copy of "s": 
    unknown_function(s); // 调用的函数类型必须与const类型一致
}            
```

\__managed__: 可以和device配合使用，表示该变量可以在设备和主机代码中都引用，例如，它的地址可以直接从主机或者设备中直接读写。

\__restrict__: 用于修饰指针，这个问题是因为在类C语言中，两个指针可能指的是同一片区域，也就是说两个指针都是同一片区域的别名，那么在一个函数中，在转换成汇编指令的过程中

​          为了保证正确性它可能就必须先拿到这个指针的所指的值，然后再进行运算，但是加上\__restrict__关键词之后，就表明这两个指针不是别名，转换为汇编的时候就可以进行指令优化。

```cpp
void foo(const float* a, const float* b, float* c) 
{ 
    c[0] = a[0] * b[0]; 
    c[1] = a[0] * b[0]; 
    c[2] = a[0] * b[0] * a[1]; 
    c[3] = a[0] * a[1]; 
    c[4] = a[0] * b[0]; 
    c[5] = b[0]; 
    ... 
}
void foo(const float* __restrict__ a, 
         const float* __restrict__ b, 
         float* __restrict__ c) 
{ 
    float t0 = a[0]; 
    float t1 = b[0]; 
    float t2 = t0 * t1; 
    float t3 = a[1]; 
    c[0] = t2; 
    c[1] = t2; 
    c[4] = t2; 
    c[2] = t2 * t3;
}
```

当然不同的情况下有不同的作用，这个例子的情况下对访存做出了优化。但是这也可能带来负面的影响，比如增加了寄存器的压力。

### 10.3 内置的vector 类型

char, short, int, long, longlong, float, double 都可以和1， 2， 3， 4来组合得到一个简单的vector类型，构造方法为make_\<type name>

```cpp
int2 make_int2(int x, int y);
```

dim3: uint3类型的变量

### 10.5 内存栅栏函数

内存栅栏函数可以用来对内存访问进行顺序一致的排序。内存栅栏函数在执行命令的范围上有所不同，但它们与访问的内存空间(共享内存,全局内存,页锁定内存,以及对等设备的内存)无关。

这与同步函数不一样，内存栅栏不能保证所有线程运行到统一位置，只保证执行memory fence函数的线程生产的数据能够安全的被其他线程消费。

### TODO:

\__threadfence_block()

\__threadfence()

\__threadfence_system()

### 10.6 同步函数

\__syncthreads()： 对一个block中的所有线程进行同步，等待所有线程到达这一点。为了避免block中的线程以不同的执行速度导致对数据的不正确读取，可以使用该函数来对线程进行同步。

\__syncthreads_count(int predicate):

\__syncthreads_and(int predicate):

\__syncthreads_or(int predicate): 

以上三种功能都和syncthreads是一致的，附加的时可以通过predicate来统计一下满足predicate的线程，count可以计数，and要求所有线程都满足，or只要有一个满足即可

\__syncwarp(unsigned mask = 0xffffffff): 可以用于同步一个warp内的线程，mask掩码中的每个bit都对应着一个线程的执行。

### 10.7 数学计算函数

### 10.8 Texture Function 纹理函数

### 10.9 Surface Function 函数

### 10.10 Read-Only Data Cache Load Function 只读数据缓存加载函数

### 10.11 Load Functions Using Cache Hints

### 10. 12  Store Functions Using Cache Hints

### 10.14 Atomic Functions 原子操作函数

原子函数可以在设备上执行一个32-bit或者64-bit的读-改-写的原子操作，原子函数只能在设备上使用。

原子操作的API可以以不同的后缀结尾，同时以不同后缀结尾的API也意味着在不同的作用域内保证原子性：

- _system结尾的表示在整个系统上都保证原子性，包括GPU和CPU，这里涉及到统一内存的东西。

- _device结尾的表示仅在设备端保证原子性。

- _block结尾的表示在该线程所在的block内保证原子性。

#### 关于原子操作的实现：

所有的原子操作都可以通过atomiCAS(compare and swap)来实现，下面是一个简单的atomicAdd()的实现：

```cpp
 #if __CUDA_ARCH__ < 600
 __device__ double atomicAdd(double* address, double val)
 {
     unsigned long long int* address_as_ull = (unsigned long long int *)address;   // 读
     unsigned long long int old = *address_as_ull, assumed;                        // old 为该地址的旧值
     do{
         assumed = old;                            // 通过CAS反复进行竞争写操作，最终将修改完的值加到
         old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));    
     }while(assumed != old);
     return __longlong_as_double(old);
 }
 #endif
```

### 10.15 Address Space Predicate Functions

```cpp
__device__ unsingned int __isGlobal(const void* ptr);// 如果该指针在全局内存中则返回1， 否则0
__device__ unsigned int __isShared(const void *ptr); // 同理
__device__ unsigned int __isConstant(const void *ptr);
__device__ unsigned int __isGridConstant(const void *ptr);
__device__ unsigned int __isLocal(const void *ptr);
```

### 10.16 Address Space Conversion Functions 地址空间转换函数

通过调用PTX的 地址转换指令，可以将地址空间的类型进行转换，如将local,shared,global, constant转换为generic，反之亦然。

```cpp
__device__ size_t __cvta_generic_to_shared(const void *ptr);
__device__ void * __cvta_shared_to_generic(size_t rawbits);   // 返回的是泛型的指针
```

### 10.17 Alloca Function

```cpp
__host__ __device__ void * alloca(size_t size);
```

alloca()函数在调用者的堆栈帧中分配size大小的内存，返回值是分配内存的指针。当从device中调用时，内存开始是和16字节对齐的。当调用alloca的函数返回时，申请的空间自动会释放。

### 10.18 编译器优化提示函数

用于向编译器优化器提供信息。

```cpp
// 允许编译器假设参数和align字节对其，并返回该参数的指针。
void * __builtin_assume_aligned(const void *exp, size_t align);
void * __builtin_assume_aligned(const void *exp, size_t align, <integral type>offset);
// example
void* res = __builtin_assume_aligned(ptr, 32);
void* res = __builtin_assume_aligned(ptr, 32, 8);


__assume(); // 
```

### 10.19
