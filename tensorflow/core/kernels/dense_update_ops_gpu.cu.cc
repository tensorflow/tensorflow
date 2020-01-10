#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/dense_update_ops.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                              \
  template struct functor::DenseUpdate<GPUDevice, T, ADD>; \
  template struct functor::DenseUpdate<GPUDevice, T, SUB>; \
  template struct functor::DenseUpdate<GPUDevice, T, ASSIGN>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
