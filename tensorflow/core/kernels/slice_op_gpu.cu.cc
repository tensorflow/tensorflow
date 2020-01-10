#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/slice_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                      \
  template struct functor::Slice<GPUDevice, T, 1>; \
  template struct functor::Slice<GPUDevice, T, 2>; \
  template struct functor::Slice<GPUDevice, T, 3>; \
  template struct functor::Slice<GPUDevice, T, 4>; \
  template struct functor::Slice<GPUDevice, T, 5>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
DEFINE_GPU_KERNELS(int32);

#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
