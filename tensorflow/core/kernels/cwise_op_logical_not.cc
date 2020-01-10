#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("LogicalNot").Device(DEVICE_CPU),
                        UnaryOp<CPUDevice, functor::logical_not>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LogicalNot").Device(DEVICE_GPU),
                        UnaryOp<GPUDevice, functor::logical_not>);
#endif
}  // namespace tensorflow
