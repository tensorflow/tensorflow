#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
template struct BinaryFunctor<GPUDevice, logical_or, 1>;
template struct BinaryFunctor<GPUDevice, logical_or, 2>;
template struct BinaryFunctor<GPUDevice, logical_or, 3>;
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
