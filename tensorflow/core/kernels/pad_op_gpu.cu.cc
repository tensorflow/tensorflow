#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/pad_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_SPECS(T)                      \
  template struct functor::Pad<GPUDevice, T, 0>; \
  template struct functor::Pad<GPUDevice, T, 1>; \
  template struct functor::Pad<GPUDevice, T, 2>; \
  template struct functor::Pad<GPUDevice, T, 3>; \
  template struct functor::Pad<GPUDevice, T, 4>; \
  template struct functor::Pad<GPUDevice, T, 5>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
