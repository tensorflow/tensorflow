#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

#define REGISTER_CPU_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Mean").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReductionOp<CPUDevice, type, Eigen::internal::MeanReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

}  // namespace tensorflow
