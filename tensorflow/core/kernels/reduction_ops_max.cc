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

#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

#define REGISTER_CPU_KERNELS(type)        \
  REGISTER_KERNEL_BUILDER(                \
      Name("Max")                         \
          .Device(DEVICE_CPU)             \
          .TypeConstraint<type>("T")      \
          .TypeConstraint<int32>("Tidx"), \
      ReductionOp<CPUDevice, type, Eigen::internal::MaxReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)          \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Max")                           \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<type>("T")        \
          .TypeConstraint<int32>("Tidx")    \
          .HostMemory("reduction_indices"), \
      ReductionOp<GPUDevice, type, Eigen::internal::MaxReducer<type>>);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("Max")
        .Device(DEVICE_GPU)
        .HostMemory("reduction_indices")
        .HostMemory("input")
        .HostMemory("output")
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx"),
    ReductionOp<CPUDevice, int32, Eigen::internal::MaxReducer<int32>>);

#undef REGISTER_GPU_KERNELS

#endif

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)         \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Max")                           \
          .Device(DEVICE_SYCL)              \
          .TypeConstraint<type>("T")        \
          .TypeConstraint<int32>("Tidx")    \
          .HostMemory("reduction_indices"), \
      ReductionOp<SYCLDevice, type, Eigen::internal::MaxReducer<type>>);
REGISTER_SYCL_KERNELS(float);
#undef REGISTER_SYCL_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("Max")
        .Device(DEVICE_SYCL)
        .HostMemory("reduction_indices")
        .HostMemory("input")
        .HostMemory("output")
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx"),
    ReductionOp<CPUDevice, int32, Eigen::internal::MaxReducer<int32>>);
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
