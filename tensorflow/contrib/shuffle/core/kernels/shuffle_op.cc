#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/contrib/shuffle/core/kernels/shuffle_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Shuffle")
                            .Device(DEVICE_CPU)
                            .HostMemory("shape"),
                        ShuffleOp);

#define REGISTER_GPU_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Shuffle")                       \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("shape")              \
                              .TypeConstraint<type>("T"),       \
                          ShuffleOp);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Shuffle")
                            .Device(DEVICE_SYCL)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output"),
                        ShuffleOp);
#endif  // TENSORFLOW_USE_SYCL

}
