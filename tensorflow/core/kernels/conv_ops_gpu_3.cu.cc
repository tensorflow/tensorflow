#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::ShuffleAndReverse<GPUDevice, float, 4>;

template struct functor::TransformFilter<GPUDevice, float>;

template struct functor::PadInput<GPUDevice, float>;

template struct functor::TransformDepth<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
