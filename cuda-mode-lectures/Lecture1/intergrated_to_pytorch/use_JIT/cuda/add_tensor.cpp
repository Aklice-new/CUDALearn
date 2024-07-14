#include <torch/extension.h>

#include <vector>

// #include "add_tensor.cu"

at::Tensor add_cuda_forward(at::Tensor a, at::Tensor b);
std::vector<at::Tensor> add_cuda_backward(at::Tensor grad);

// 这两个接口是为了转发到cuda kernel
at::Tensor add_forward(at::Tensor a, at::Tensor b) {
  //   CHECK_INPUT(a);
  //   CHECK_INPUT(b);
  return add_cuda_forward(a, b);
}

std::vector<at::Tensor> add_backward(at::Tensor grad) {
  //   CHECK_INPUT(grad);
  return add_cuda_backward(grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward, "add tensor forward (CUDA)");
  m.def("backward", &add_backward, "add tensor backward (CUDA)");
}
// CUDA forward declarations