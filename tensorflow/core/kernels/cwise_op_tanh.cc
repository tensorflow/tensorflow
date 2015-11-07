#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Tanh", functor::tanh, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Tanh", functor::tanh, float, double);
#endif
}  // namespace tensorflow
