#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER3(UnaryOp, CPU, "Sigmoid", functor::sigmoid, float, double, complex64);
#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Sigmoid", functor::sigmoid, float, double);
#endif
}  // namespace tensorflow
