#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
// cuda part

template <typename scalar_t>
__global__ void add_forward_kernel(scalar_t* a, scalar_t* b, scalar_t* res) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  res[idx] = a[idx] + b[idx];
}

template <typename scalar_t>
__global__ void add_backward_kernel(scalar_t* grad, scalar_t* d_a,
                                    scalar_t* d_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[idx] = grad[idx];
  d_b[idx] = grad[idx];
}

// 下面的接口为了检查和转发cuda的接口
at::Tensor add_cuda_forward(at::Tensor a, at::Tensor b) {
  auto a_shape = a.sizes();
  auto b_shape = b.sizes();
  assert(a_shape == b_shape);

  at::Tensor res = at::zeros_like(a);
  int num_elements = a.numel();
  // 这里使用1维的block
  // 每个block内的线程数为256
  dim3 thread_per_block(256);
  dim3 block_per_grid((num_elements + 256 - 1) / 256);
  // 这里这个宏会根据a的数据类型来选择对应的scalar_t
  /*
    AT_DISPATCH_FLOATING_TYPES：
        switch (tensor.type().scalarType()) {
        case at::ScalarType::Double:
            return function<double>(tensor.data<double>());
        case at::ScalarType::Float:
            return function<float>(tensor.data<float>());
        ...
        }
  */

  AT_DISPATCH_FLOATING_TYPES(a.type(), "add_forward_kernel", [&] {
    add_forward_kernel<scalar_t><<<block_per_grid, thread_per_block>>>(
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        res.data_ptr<scalar_t>());
  });
  // 完成cpu和gpu的同步
  cudaDeviceSynchronize();
  return res;
}

std::vector<at::Tensor> add_cuda_backward(at::Tensor grad) {
  auto d_a = at::zeros_like(grad);
  auto d_b = at::zeros_like(grad);
  int num_elements = grad.numel();
  dim3 thread_per_block(256);
  dim3 block_per_grid((num_elements + 256 - 1) / 256);
  std::cout << "begin to backward" << std::endl;
  AT_DISPATCH_FLOATING_TYPES(grad.type(), "add_backward_kernel", [&] {
    add_backward_kernel<scalar_t><<<block_per_grid, thread_per_block>>>(
        grad.data_ptr<scalar_t>(), d_a.data_ptr<scalar_t>(),
        d_b.data_ptr<scalar_t>());
  });
  cudaDeviceSynchronize();
  return {d_a, d_b};
}