#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(BinaryOp, CPU, "NotEqual", functor::not_equal_to, float, double,
          int32, int64, complex64);
#if GOOGLE_CUDA
REGISTER3(BinaryOp, GPU, "NotEqual", functor::not_equal_to, float, double,
          int64);
#endif
}  // namespace tensorflow
