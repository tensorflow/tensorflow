#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Sqrt", functor::sqrt, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Sqrt", functor::sqrt, float, double);
#endif
}  // namespace tensorflow
