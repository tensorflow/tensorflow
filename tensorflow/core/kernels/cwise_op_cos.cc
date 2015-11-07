#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Cos", functor::cos, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Cos", functor::cos, float, double);
#endif
}  // namespace tensorflow
