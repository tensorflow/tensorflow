#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/adjust_contrast_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::AdjustContrast<GPUDevice, uint8>;
template struct functor::AdjustContrast<GPUDevice, int8>;
template struct functor::AdjustContrast<GPUDevice, int16>;
template struct functor::AdjustContrast<GPUDevice, int32>;
template struct functor::AdjustContrast<GPUDevice, int64>;
template struct functor::AdjustContrast<GPUDevice, float>;
template struct functor::AdjustContrast<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
