#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Rsqrt", functor::rsqrt, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Rsqrt", functor::rsqrt, float, double);
#endif
}  // namespace tensorflow
