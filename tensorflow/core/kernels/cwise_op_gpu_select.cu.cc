#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
template struct SelectFunctor<GPUDevice, float>;
template struct SelectFunctor<GPUDevice, double>;
template struct SelectFunctor<GPUDevice, int32>;
template struct SelectFunctor<GPUDevice, int64>;
template struct SelectFunctor<GPUDevice, complex64>;
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
