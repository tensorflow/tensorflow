#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/batch_norm_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::BatchNorm<GPUDevice, float>;
template struct functor::BatchNormGrad<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
