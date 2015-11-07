#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Sin", functor::sin, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Sin", functor::sin, float, double);
#endif
}  // namespace tensorflow
