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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("VariableV2").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("TemporaryVariable").Device(DEVICE_CPU),
                        TemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable").Device(DEVICE_CPU),
                        DestroyTemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(DEVICE_CPU),
                        IsVariableInitializedOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
                          Name("Variable")                              \
                          .Device(DEVICE_SYCL)                          \
                          .TypeConstraint<TYPE>("dtype"),               \
                          VariableOp);                                  \
  REGISTER_KERNEL_BUILDER(Name("VariableV2")                            \
                          .Device(DEVICE_SYCL)                          \
                          .TypeConstraint<TYPE>("dtype"),               \
                          VariableOp);                                  \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                     \
                          .Device(DEVICE_SYCL)                          \
                          .TypeConstraint<TYPE>("dtype"),               \
                          TemporaryVariableOp);                         \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")              \
                          .Device(DEVICE_SYCL)                          \
                          .TypeConstraint<TYPE>("T"),                   \
                          DestroyTemporaryVariableOp);                  \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                 \
                          .Device(DEVICE_SYCL)                          \
                          .TypeConstraint<TYPE>("dtype")                \
                          .HostMemory("is_initialized"),                \
                          IsVariableInitializedOp);

REGISTER_SYCL_KERNEL(float);
#undef REGISTER_SYCL_KERNEL
#endif

#if GOOGLE_CUDA
// Only register 'Variable' on GPU for the subset of types also supported by
// 'Assign' (see dense_update_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("VariableV2").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                        \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype"),              \
                          TemporaryVariableOp);                            \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                 \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T"),                  \
                          DestroyTemporaryVariableOp);                     \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                    \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype")               \
                              .HostMemory("is_initialized"),               \
                          IsVariableInitializedOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
