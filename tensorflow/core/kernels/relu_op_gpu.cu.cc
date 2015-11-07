#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/relu_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in relu_op.cc.
#define DEFINE_GPU_KERNELS(T)                      \
  template struct functor::Relu<GPUDevice, T>;     \
  template struct functor::ReluGrad<GPUDevice, T>; \
  template struct functor::Relu6<GPUDevice, T>;    \
  template struct functor::Relu6Grad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
