#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::ShuffleAndReverse<GPUDevice, float, 4, int>;
template struct functor::ShuffleAndReverse<GPUDevice, float, 4,
                                           Eigen::DenseIndex>;

template struct functor::TransformFilter<GPUDevice, float, int>;

template struct functor::PadInput<GPUDevice, float, int>;

template struct functor::TransformDepth<GPUDevice, float, int>;
// TODO(jiayq): currently pooling ops still use DenseIndex, so I am keeping it
// here.
template struct functor::TransformDepth<GPUDevice, float, Eigen::DenseIndex>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
