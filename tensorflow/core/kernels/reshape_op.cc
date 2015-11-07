// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Reshape").Device(DEVICE_CPU).HostMemory("shape"),
                        ReshapeOp);

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T"), \
                          ReshapeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ReshapeOp);

}  // namespace tensorflow
