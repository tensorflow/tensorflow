#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("LogicalAnd").Device(DEVICE_CPU),
                        BinaryOp<CPUDevice, functor::logical_and>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LogicalAnd").Device(DEVICE_GPU),
                        BinaryOp<GPUDevice, functor::logical_and>);
#endif
}  // namespace tensorflow
