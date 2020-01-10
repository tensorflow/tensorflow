#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("Complex").Device(DEVICE_CPU),
                        BinaryOp<CPUDevice, functor::make_complex<float>>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Complex").Device(DEVICE_GPU),
                        BinaryOp<GPUDevice, functor::make_complex<float>>);
#endif
}  // namespace tensorflow
