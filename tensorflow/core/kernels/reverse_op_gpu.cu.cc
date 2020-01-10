#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/reverse_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_REVERSE(DIM)                                \
  template struct functor::Reverse<GPUDevice, uint8, DIM>; \
  template struct functor::Reverse<GPUDevice, int8, DIM>;  \
  template struct functor::Reverse<GPUDevice, int32, DIM>; \
  template struct functor::Reverse<GPUDevice, bool, DIM>;  \
  template struct functor::Reverse<GPUDevice, float, DIM>; \
  template struct functor::Reverse<GPUDevice, double, DIM>;
DEFINE_REVERSE(0)
DEFINE_REVERSE(1)
DEFINE_REVERSE(2)
DEFINE_REVERSE(3)
DEFINE_REVERSE(4)
DEFINE_REVERSE(5)
DEFINE_REVERSE(6)
DEFINE_REVERSE(7)
DEFINE_REVERSE(8)
#undef DEFINE_REVERSE

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
