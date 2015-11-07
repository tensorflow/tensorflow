#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER2(BinaryOp, CPU, "Mod", functor::mod, int32, int64);
REGISTER2(BinaryOp, CPU, "Mod", functor::fmod, float, double);
}  // namespace tensorflow
