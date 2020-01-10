#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
DEFINE_BINARY4(not_equal_to, float, double, int64, complex64);
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
