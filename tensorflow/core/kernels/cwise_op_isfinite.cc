#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(UnaryOp, CPU, "IsFinite", functor::isfinite, float, double);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "IsFinite", functor::isfinite, float, double);
#endif
}  // namespace tensorflow
