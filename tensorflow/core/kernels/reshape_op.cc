/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Reshape").Device(DEVICE_CPU).HostMemory("shape"),
                        ReshapeOp);

#define REGISTER_GPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Reshape")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T"), \
                          ReshapeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#if GOOGLE_CUDA
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("tensor")
                            .HostMemory("shape")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ReshapeOp);
#endif

}  // namespace tensorflow
