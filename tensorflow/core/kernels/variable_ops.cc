#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("TemporaryVariable").Device(DEVICE_CPU),
                        TemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable").Device(DEVICE_CPU),
                        DestroyTemporaryVariableOp);

#if GOOGLE_CUDA
// Only register 'Variable' on GPU for the subset of types also supported by
// 'Assign' (see dense_update_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                       \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                      \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("dtype"),            \
                          TemporaryVariableOp);                          \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")               \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T"),                \
                          DestroyTemporaryVariableOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
