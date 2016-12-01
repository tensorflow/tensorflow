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
REGISTER7(UnaryOp, CPU, "Neg", functor::neg, float, Eigen::half, double, int32,
          complex64, int64, complex128);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
                          Name("Neg")                                 \
                          .Device(DEVICE_SYCL)                        \
                          .TypeConstraint<TYPE>("T"),                 \
                          UnaryOp<SYCLDevice, functor::neg<TYPE>>);
REGISTER_SYCL_KERNEL(float);
#undef REGISTER_SYCL_KERNEL
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
REGISTER4(UnaryOp, GPU, "Neg", functor::neg, float, Eigen::half, double, int64);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::neg<int32>>);
#endif
}  // namespace tensorflow
