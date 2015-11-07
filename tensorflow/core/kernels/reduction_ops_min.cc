#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

#define REGISTER_CPU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Min").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReductionOp<CPUDevice, type, Eigen::internal::MinReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)          \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Min")                           \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<type>("T")        \
          .HostMemory("reduction_indices"), \
      ReductionOp<GPUDevice, type, Eigen::internal::MinReducer<type>>);
REGISTER_GPU_KERNELS(float);
#undef REGISTER_GPU_KERNELS

#endif

}  // namespace tensorflow
