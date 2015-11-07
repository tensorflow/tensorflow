#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/softplus_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in softplus_op.cc.
#define DEFINE_GPU_KERNELS(T)                      \
  template struct functor::Softplus<GPUDevice, T>; \
  template struct functor::SoftplusGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
