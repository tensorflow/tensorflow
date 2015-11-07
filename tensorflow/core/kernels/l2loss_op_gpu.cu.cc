#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/l2loss_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::L2Loss<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
