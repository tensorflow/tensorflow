#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
DEFINE_BINARY1(make_complex, float);
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
