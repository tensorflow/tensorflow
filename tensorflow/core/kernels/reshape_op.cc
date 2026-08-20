/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations me the License.
==============================================================================*/

#include "tensorflow/core/kernels/reshape_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(
    Name("Reshape").Device(DEVICE_CPU).HostMemory("shape"),
    ReshapeOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(
    Name("Reshape").Device(DEVICE_GPU).HostMemory("shape"),
    ReshapeOp);
#endif

#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("Reshape").Device(DEVICE_SYCL).HostMemory("shape"),
    ReshapeOp);
#endif

}  // namespace tensorflow
