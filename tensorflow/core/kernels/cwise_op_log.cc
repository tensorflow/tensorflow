#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Log", functor::log, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Log", functor::log, float, double);
#endif
}  // namespace tensorflow
