#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER4(BinaryOp, CPU, "Less", functor::less, float, double, int32, int64);
#if GOOGLE_CUDA
REGISTER3(BinaryOp, GPU, "Less", functor::less, float, double, int64);
#endif

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Less")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less<int32>>);

}  // namespace tensorflow
