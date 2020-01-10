#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(BinaryOp, CPU, "Pow", functor::pow, float, double, int32, int64,
          complex64);
#if GOOGLE_CUDA
REGISTER3(BinaryOp, GPU, "Pow", functor::pow, float, double, int64);
#endif
}  // namespace tensorflow
