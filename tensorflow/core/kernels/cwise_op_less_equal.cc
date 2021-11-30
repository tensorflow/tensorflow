/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(BinaryOp, CPU, "LessEqual", functor::less_equal, float, Eigen::half,
          bfloat16, double, int32);
REGISTER7(BinaryOp, CPU, "LessEqual", functor::less_equal, int64_t, uint8,
          uint16, uint32, uint64, int8, int16);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER9(BinaryOp, GPU, "LessEqual", functor::less_equal, float, Eigen::half,
          double, int64, uint8, uint16, uint32, uint64, int8);
REGISTER(BinaryOp, GPU, "LessEqual", functor::less_equal, int16);
#endif

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("LessEqual")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less_equal<int32>>);
#endif
REGISTER_KERNEL_BUILDER(Name("LessEqual")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less_equal<int32>>);

}  // namespace tensorflow
