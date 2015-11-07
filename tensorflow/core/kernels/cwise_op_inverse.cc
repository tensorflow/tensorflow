#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Inv", functor::inverse, float, double, complex64);
#if GOOGLE_CUDA
REGISTER3(UnaryOp, GPU, "Inv", functor::inverse, float, double, int64);
#endif
}  // namespace tensorflow
