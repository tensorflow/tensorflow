#include "tensorflow/core/kernels/map_kernels.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("EmptyTensorMap").Device(DEVICE_CPU),
                        EmptyTensorMap);

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU),
                        ZeroOutOp);
}