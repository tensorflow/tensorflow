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

REGISTER7(UnaryOp, CPU, "Square", functor::square, float, Eigen::half, double,
          int32, int64_t, complex64, complex128);
REGISTER7(UnaryOp, CPU, "Square", functor::square, bfloat16, int8, int16, uint8,
          uint16, uint32, uint64);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER4(UnaryOp, GPU, "Square", functor::square, float, Eigen::half, double,
          int64);
#endif

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Square")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::square<int32>>);
#endif

REGISTER_KERNEL_BUILDER(Name("Square")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::square<int32>>);

}  // namespace tensorflow
