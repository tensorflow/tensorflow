#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/argmax_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T)                       \
  template struct functor::ArgMax<GPUDevice, T>; \
  template struct functor::ArgMin<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
