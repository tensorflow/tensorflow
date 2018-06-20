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
REGISTER9(BinaryOp, CPU, "Less", functor::less, float, Eigen::half, double,
          bfloat16, int32, int64, uint8, int8, int16);
#if GOOGLE_CUDA
REGISTER7(BinaryOp, GPU, "Less", functor::less, float, Eigen::half, double,
          int64, uint8, int8, int16);

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
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER3(BinaryOp, SYCL, "Less", functor::less, float, double, int64);
REGISTER_KERNEL_BUILDER(Name("Less")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::less<int32>>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
