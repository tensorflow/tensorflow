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

#define REGISTER_CPU_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Max")                                                           \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int32>("Tidx"),                                   \
      ReductionOp<CPUDevice, type, int32,                                   \
                  Eigen::internal::MaxReducer<type, Eigen::PropagateNaN>>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Max")                                                           \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int64_t>("Tidx"),                                 \
      ReductionOp<CPUDevice, type, int64,                                   \
                  Eigen::internal::MaxReducer<type, Eigen::PropagateNaN>>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Max")                                                           \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int32>("Tidx")                                    \
          .HostMemory("reduction_indices"),                                 \
      ReductionOp<GPUDevice, type, int32,                                   \
                  Eigen::internal::MaxReducer<type, Eigen::PropagateNaN>>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Max")                                                           \
          .Device(DEVICE_GPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int64_t>("Tidx")                                  \
          .HostMemory("reduction_indices"),                                 \
      ReductionOp<GPUDevice, type, int64,                                   \
                  Eigen::internal::MaxReducer<type, Eigen::PropagateNaN>>);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(Eigen::bfloat16);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
REGISTER_GPU_KERNELS(int64_t);

#undef REGISTER_GPU_KERNELS
#endif

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("Max")
        .Device(DEVICE_DEFAULT)
        .HostMemory("reduction_indices")
        .HostMemory("input")
        .HostMemory("output")
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::MaxReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Max")
        .Device(DEVICE_DEFAULT)
        .HostMemory("reduction_indices")
        .HostMemory("input")
        .HostMemory("output")
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64_t>("Tidx"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::MaxReducer<int32>>);

}  // namespace tensorflow
