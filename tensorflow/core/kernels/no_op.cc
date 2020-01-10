#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_GPU), NoOp);

}  // namespace tensorflow
