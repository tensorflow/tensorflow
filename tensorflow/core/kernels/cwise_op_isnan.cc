#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(UnaryOp, CPU, "IsNan", functor::isnan, float, double);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "IsNan", functor::isnan, float, double);
#endif
}  // namespace tensorflow
