// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/identity_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Identity").Device(DEVICE_CPU), IdentityOp);
// StopGradient does the same thing as Identity, but has a different
// gradient registered.
REGISTER_KERNEL_BUILDER(Name("StopGradient").Device(DEVICE_CPU), IdentityOp);

REGISTER_KERNEL_BUILDER(Name("RefIdentity").Device(DEVICE_CPU), IdentityOp);

#define REGISTER_GPU_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Identity").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      IdentityOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("RefIdentity").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      IdentityOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("StopGradient").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      IdentityOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_KERNEL(bfloat16);

#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Identity")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        IdentityOp);

}  // namespace tensorflow
