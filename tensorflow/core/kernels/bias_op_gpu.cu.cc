#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bias_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in bias_op.cc.
#define DEFINE_GPU_SPECS(T)                       \
  template struct functor::Bias<GPUDevice, T, 2>; \
  template struct functor::Bias<GPUDevice, T, 3>; \
  template struct functor::Bias<GPUDevice, T, 4>; \
  template struct functor::Bias<GPUDevice, T, 5>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
