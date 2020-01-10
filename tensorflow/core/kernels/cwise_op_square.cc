#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Square", functor::square, float, double, int32,
          complex64, int64);
#if GOOGLE_CUDA
REGISTER3(UnaryOp, GPU, "Square", functor::square, float, double, int64);
#endif
}  // namespace tensorflow
