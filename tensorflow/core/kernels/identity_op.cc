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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/identity_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Identity").Device(DEVICE_CPU), IdentityOp);
// StopGradient does the same thing as Identity, but has a different
// gradient registered.
REGISTER_KERNEL_BUILDER(Name("StopGradient").Device(DEVICE_CPU), IdentityOp);
// PreventGradient does the same thing as Identity, but has a NO
// gradient registered.
REGISTER_KERNEL_BUILDER(Name("PreventGradient").Device(DEVICE_CPU), IdentityOp);

// PlaceholderWithDefault does the same thing as Identity, but has a
// different shape function (and constant value function) registered.
REGISTER_KERNEL_BUILDER(Name("PlaceholderWithDefault").Device(DEVICE_CPU),
                        IdentityOp);

REGISTER_KERNEL_BUILDER(Name("RefIdentity").Device(DEVICE_CPU), IdentityOp);

// Identity op for gradients debugging in TensorFlow Debugger (hidden op in
// Python).
REGISTER_KERNEL_BUILDER(Name("DebugGradientIdentity").Device(DEVICE_CPU),
                        IdentityOp);
REGISTER_KERNEL_BUILDER(Name("DebugGradientRefIdentity").Device(DEVICE_CPU),
                        IdentityOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Identity").Device(DEVICE_SYCL).TypeConstraint<type>("T"),        \
      IdentityOp);                                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("PreventGradient").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      IdentityOp);                                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("RefIdentity").Device(DEVICE_SYCL).TypeConstraint<type>("T"),     \
      IdentityOp);                                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StopGradient").Device(DEVICE_SYCL).TypeConstraint<type>("T"),    \
      IdentityOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Identity")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          IdentityOp);                    \
  REGISTER_KERNEL_BUILDER(Name("RefIdentity")             \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          IdentityOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(bool);

#undef REGISTER_SYCL_HOST_KERNEL

#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_GPU_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Identity").Device(DEVICE_GPU).TypeConstraint<type>("T"),        \
      IdentityOp);                                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PreventGradient").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      IdentityOp);                                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("RefIdentity").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      IdentityOp);                                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StopGradient").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      IdentityOp);                                                          \
  REGISTER_KERNEL_BUILDER(Name("DebugGradientIdentity")                     \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T"),                   \
                          IdentityOp);                                      \
  REGISTER_KERNEL_BUILDER(Name("PlaceholderWithDefault")                    \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("dtype"),               \
                          IdentityOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(Variant);

#undef REGISTER_GPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Identity")                    \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("input")            \
                              .HostMemory("output")           \
                              .TypeConstraint<type>("T"),     \
                          IdentityOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefIdentity")                 \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("input")            \
                              .HostMemory("output")           \
                              .TypeConstraint<type>("T"),     \
                          IdentityOp);                        \
  REGISTER_KERNEL_BUILDER(Name("StopGradient")                \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("input")            \
                              .HostMemory("output")           \
                              .TypeConstraint<type>("T"),     \
                          IdentityOp);                        \
  REGISTER_KERNEL_BUILDER(Name("PlaceholderWithDefault")      \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("input")            \
                              .HostMemory("output")           \
                              .TypeConstraint<type>("dtype"), \
                          IdentityOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);
REGISTER_GPU_HOST_KERNEL(string);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL

#endif

}  // namespace tensorflow
