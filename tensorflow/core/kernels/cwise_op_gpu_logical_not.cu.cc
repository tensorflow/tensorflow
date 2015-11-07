#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
template struct UnaryFunctor<GPUDevice, logical_not>;
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
