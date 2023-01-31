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

#include "gather_op.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<dev##Device, type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                             \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("axis"),                     \
                          GatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

} // namespace tensorflow
