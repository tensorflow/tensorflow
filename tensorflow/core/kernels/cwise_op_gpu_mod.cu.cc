#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
// No GPU ops for mod yet.
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
