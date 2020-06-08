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
REGISTER8(BinaryOp, CPU, "Sub", functor::sub, float, Eigen::half, double, int32,
          int64, bfloat16, complex64, complex128);
#if !defined(__ANDROID_TYPES_SLIM__)
// Sub op for int8, uint8, int16, uint16
REGISTER5(BinaryOp, CPU, "Sub", functor::sub, int8, uint8, int16, uint16,
          uint32);
#else
// We only register the first type when we have multi-argument calls in the
// case where we're trying to reduce executable size, but it turns out that the
// int32 version of this op is needed, so explicitly include it.
REGISTER(BinaryOp, CPU, "Sub", functor::sub, int32);
#endif  // __ANDROID_TYPES_SLIM__

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER7(BinaryOp, GPU, "Sub", functor::sub, float, Eigen::half, double, int64,
          complex64, complex128, uint32);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::sub<int32>>);
#endif

#ifdef TENSORFLOW_USE_SYCL
REGISTER3(BinaryOp, SYCL, "Sub", functor::sub, float, double, int64);
REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::sub<int32>>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
