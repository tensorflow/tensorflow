#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

#define REGISTER_CPU_KERNEL(type, Tshape_type)            \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                 \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("tensor")       \
                              .HostMemory("shape")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T")  \
                              .TypeConstraint<Tshape_type>("Tshape"), \
                          ReshapeOp);

#define REGISTER_CPU_KERNELS(type) \
  REGISTER_CPU_KERNEL(type, int32); \
  REGISTER_CPU_KERNEL(type, int64_t);

TF_CALL_ALL_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type, Tshape_type)            \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T")  \
                              .TypeConstraint<Tshape_type>("Tshape"), \
                          ReshapeOp);

#define REGISTER_GPU_KERNELS(type) \
  REGISTER_GPU_KERNEL(type, int32); \
  REGISTER_GPU_KERNEL(type, int64_t);

TF_CALL_ALL_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_GPU_KERNEL
#endif

#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("shape")
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);
#endif

}  // namespace tensorflow
