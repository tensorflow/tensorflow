#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::InflatePadAndShuffle<GPUDevice, float, 4>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
