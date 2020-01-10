#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
template struct BinaryFunctor<GPUDevice, logical_and, 1>;
template struct BinaryFunctor<GPUDevice, logical_and, 2>;
template struct BinaryFunctor<GPUDevice, logical_and, 3>;
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
