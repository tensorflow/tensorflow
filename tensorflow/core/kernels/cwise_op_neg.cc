#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Neg", functor::neg, float, double, int32, complex64,
          int64);
#if GOOGLE_CUDA
REGISTER4(UnaryOp, GPU, "Neg", functor::neg, float, double, int32, int64);
#endif
}  // namespace tensorflow
