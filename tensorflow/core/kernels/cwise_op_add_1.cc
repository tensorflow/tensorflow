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
REGISTER6(BinaryOp, CPU, "Add", functor::add, float, Eigen::half, double, int32,
          int64, bfloat16);
REGISTER6(BinaryOp, CPU, "AddV2", functor::add, float, Eigen::half, double,
          int32, int64, bfloat16);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER3(BinaryOp, GPU, "Add", functor::add, float, Eigen::half, double);
REGISTER3(BinaryOp, GPU, "AddV2", functor::add, float, Eigen::half, double);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
REGISTER_KERNEL_BUILDER(Name("AddV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
#endif

#if TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                          \
  REGISTER(BinaryOp, SYCL, "Add", functor::add, type); \
  REGISTER(BinaryOp, SYCL, "AddV2", functor::add, type);

TF_CALL_SYCL_NUMBER_TYPES(REGISTER_KERNEL);

REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
REGISTER_KERNEL_BUILDER(Name("AddV2")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::add<int32>>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
