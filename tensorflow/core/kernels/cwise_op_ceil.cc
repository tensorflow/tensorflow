#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(UnaryOp, CPU, "Ceil", functor::ceil, float, double);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Ceil", functor::ceil, float, double);
#endif
}  // namespace tensorflow
