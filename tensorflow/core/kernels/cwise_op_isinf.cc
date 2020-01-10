#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(UnaryOp, CPU, "IsInf", functor::isinf, float, double);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "IsInf", functor::isinf, float, double);
#endif
}  // namespace tensorflow
