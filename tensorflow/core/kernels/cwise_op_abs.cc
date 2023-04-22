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

#if !defined(MLIR_GENERATED_CPU_KERNELS_ENABLED) || \
    !defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)
REGISTER8(UnaryOp, CPU, "Abs", functor::abs, Eigen::half, bfloat16, float,
          double, int8, int16, int32, int64);
#else
REGISTER(UnaryOp, CPU, "Abs", functor::abs, bfloat16);
#endif

REGISTER2(UnaryOp, CPU, "ComplexAbs", functor::abs, complex64, complex128);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER4(UnaryOp, GPU, "Abs", functor::abs, Eigen::half, float, double, int64);
REGISTER2(UnaryOp, GPU, "ComplexAbs", functor::abs, complex64, complex128);
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
#endif
REGISTER_KERNEL_BUILDER(Name("Abs")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::abs<int32>>);

}  // namespace tensorflow
