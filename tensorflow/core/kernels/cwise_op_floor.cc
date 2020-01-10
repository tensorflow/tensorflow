#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(UnaryOp, CPU, "Floor", functor::floor, float, double);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Floor", functor::floor, float, double);
#endif
}  // namespace tensorflow
