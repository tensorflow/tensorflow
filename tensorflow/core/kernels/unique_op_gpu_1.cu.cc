/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/unique_op_gpu.cu.h"

namespace tensorflow {

// Register with int64 out_idx.
#define REGISTER_UNIQUE_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOpGPU<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOpGPU<type, int64>)

TF_CALL_FLOAT_TYPES(REGISTER_UNIQUE_GPU);

#undef REGISTER_UNIQUE_GPU

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
