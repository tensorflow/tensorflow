#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("LogicalOr").Device(DEVICE_CPU),
                        BinaryOp<CPUDevice, functor::logical_or>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LogicalOr").Device(DEVICE_GPU),
                        BinaryOp<GPUDevice, functor::logical_or>);
#endif
}  // namespace tensorflow
