#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Any")
                            .Device(DEVICE_CPU)
                            .HostMemory("reduction_indices"),
                        ReductionOp<CPUDevice, bool, functor::AnyReducer>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Any")
                            .Device(DEVICE_GPU)
                            .HostMemory("reduction_indices"),
                        ReductionOp<GPUDevice, bool, functor::AnyReducer>);
#endif

}  // namespace tensorflow
