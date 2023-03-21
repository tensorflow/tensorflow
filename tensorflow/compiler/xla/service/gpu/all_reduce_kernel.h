/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_KERNEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_KERNEL_H_

#include "third_party/nccl/nccl.h"

namespace xla::gpu {

inline constexpr int kMaxBuffers = 32;
inline constexpr int kMaxNumGpus = 16;
inline constexpr int kLaunchBounds = 256;

enum SyncFlag { SYNC_NONE = 0, SYNC_START = 1, SYNC_END = 2 };

const void* GetSyncKernel();
const void* GetAllReduceKernel(ncclDataType_t dtype, int64_t* num_elements,
                               int cc_major);

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_KERNEL_H_
