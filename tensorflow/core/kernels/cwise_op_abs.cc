#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER4(UnaryOp, CPU, "Abs", functor::abs, float, double, int32, int64);
#ifndef __ANDROID__
REGISTER_KERNEL_BUILDER(Name("ComplexAbs").Device(DEVICE_CPU),
                        UnaryOp<CPUDevice, functor::abs<complex64>>);
#endif
#if GOOGLE_CUDA
REGISTER3(UnaryOp, GPU, "Abs", functor::abs, float, double, int64);
#endif

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Abs")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::abs<int32>>);

}  // namespace tensorflow
