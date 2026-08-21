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

#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_CPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_CPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("Tshape"),
                        ReshapeOp);

#define REGISTER_DEFAULT_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                           \
                              .Device(DEVICE_DEFAULT)               \
                              .HostMemory("shape")                  \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("Tshape"),     \
                          ReshapeOp);                               \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                           \
                              .Device(DEVICE_DEFAULT)               \
                              .HostMemory("shape")                  \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("Tshape"),   \
                          ReshapeOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("Tshape"),
                        ReshapeOp);

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("Tshape"),
                        ReshapeOp);
#endif

}  // namespace tensorflow
