#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

#define REGISTER_CPU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Sum").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReductionOp<CPUDevice, type, Eigen::internal::SumReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// NOTE: We should have mean(complex64,int32), too. But that needs to
// change Eigen::internal::MeanReducer to cast int to complex<float>.
// We don't see immediate need of mean(complex64,int32) anyway.
REGISTER_KERNEL_BUILDER(
    Name("Sum").Device(DEVICE_CPU).TypeConstraint<complex64>("T"),
    ReductionOp<CPUDevice, complex64, Eigen::internal::SumReducer<complex64>>);

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)          \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Sum")                           \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<type>("T")        \
          .HostMemory("reduction_indices"), \
      ReductionOp<GPUDevice, type, Eigen::internal::SumReducer<type>>);
REGISTER_GPU_KERNELS(float);
#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("Sum").Device(DEVICE_GPU).TypeConstraint<complex64>("T"),
    ReductionOp<GPUDevice, complex64, Eigen::internal::SumReducer<complex64>>);

#endif

}  // namespace tensorflow
