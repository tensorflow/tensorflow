#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("Real").Device(DEVICE_CPU),
                        UnaryOp<CPUDevice, functor::get_real<complex64>>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Real").Device(DEVICE_GPU),
                        UnaryOp<GPUDevice, functor::get_real<complex64>>);
#endif
}  // namespace tensorflow
