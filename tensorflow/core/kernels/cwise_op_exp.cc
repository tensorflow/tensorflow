#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Exp", functor::exp, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Exp", functor::exp, float, double);
#endif
}  // namespace tensorflow
