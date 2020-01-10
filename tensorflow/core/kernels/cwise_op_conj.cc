#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("Conj").Device(DEVICE_CPU),
                        UnaryOp<CPUDevice, functor::conj<complex64>>);
#if GOOGLE_CUDA
// REGISTER_KERNEL_BUILDER(Name("Conj").Device(DEVICE_GPU),
//                         UnaryOp<GPUDevice, functor::conj<complex64>>);
#endif
}  // namespace tensorflow
