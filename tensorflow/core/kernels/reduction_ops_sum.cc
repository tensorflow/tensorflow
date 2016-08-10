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

#define REGISTER_CPU_KERNELS(type)                              \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Sum").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ReductionOp<CPUDevice, type, Eigen::internal::SumReducer<type>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
// NOTE: We should have mean(complex64,int32), too. But that needs to
// change Eigen::internal::MeanReducer to cast int to complex<float>.
// We don't see immediate need of mean(complex64,int32) anyway.
TF_CALL_complex64(REGISTER_CPU_KERNELS);
TF_CALL_complex128(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)          \
  REGISTER_KERNEL_BUILDER(                  \
      Name("Sum")                           \
          .Device(DEVICE_GPU)               \
          .TypeConstraint<type>("T")        \
          .HostMemory("reduction_indices"), \
      ReductionOp<GPUDevice, type, Eigen::internal::SumReducer<type>>);
REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
REGISTER_GPU_KERNELS(complex64);
REGISTER_GPU_KERNELS(complex128);
#undef REGISTER_GPU_KERNELS

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, Eigen::internal::SumReducer<int32>>);

#endif

}  // namespace tensorflow
