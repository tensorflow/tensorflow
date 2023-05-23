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

REGISTER6(BinaryOp, CPU, "Mul", functor::mul, float, Eigen::half, double, uint8,
          int32, bfloat16);
REGISTER6(BinaryOp, CPU, "MulNoNan", functor::mul_no_nan, Eigen::half, float,
          double, complex64, complex128, bfloat16);

#if defined(__ANDROID_TYPES_SLIM__)
// We only register the first type when we have multi-argument calls in the
// case where we're trying to reduce executable size, but it turns out that the
// int32 version of this op is needed, so explicitly include it.
REGISTER(BinaryOp, CPU, "Mul", functor::mul, int32);
#endif  // __ANDROID_TYPES_SLIM__

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER4(BinaryOp, GPU, "Mul", functor::mul, Eigen::half, float, double,
          uint8);
#endif

REGISTER(BinaryOp, GPU, "Mul", functor::mul, bfloat16);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::mul<int32>>);
#endif
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::mul<int32>>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER5(BinaryOp, GPU, "MulNoNan", functor::mul_no_nan, Eigen::half, float,
          double, complex64, complex128);
#endif
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
