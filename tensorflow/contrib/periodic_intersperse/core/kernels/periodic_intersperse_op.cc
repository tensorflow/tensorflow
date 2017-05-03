#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/contrib/periodic_intersperse/core/kernels/periodic_intersperse_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("PeriodicIntersperse")
                            .HostMemory("desired_shape")
                            .Device(DEVICE_CPU),
                        PeriodicIntersperseOp);

#define REGISTER_GPU_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("PeriodicIntersperse")           \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("desired_shape")      \
                              .TypeConstraint<type>("T"),       \
                          PeriodicIntersperseOp);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("PeriodicIntersperse")
                            .Device(DEVICE_SYCL)
                            .HostMemory("output"),
                        PeriodicIntersperseOp);
#endif  // TENSORFLOW_USE_SYCL

}
