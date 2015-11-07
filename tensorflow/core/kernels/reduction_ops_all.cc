#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("All")
                            .Device(DEVICE_CPU)
                            .HostMemory("reduction_indices"),
                        ReductionOp<CPUDevice, bool, functor::AllReducer>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("All")
                            .Device(DEVICE_GPU)
                            .HostMemory("reduction_indices"),
                        ReductionOp<GPUDevice, bool, functor::AllReducer>);
#endif

}  // namespace tensorflow
