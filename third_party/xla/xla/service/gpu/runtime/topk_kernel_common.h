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

#ifndef XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_COMMON_H_
#define XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_COMMON_H_

#include <cstddef>

// Contains shared declarations between topk_kernel.cc and topk_kernel.cu.cc
// but avoids including ABSL, etc. which some CUDA compilers cannot
// handle.

namespace xla::gpu {

// We perform 2 32-way reductions, which means the largest number of threads per
// block we support is 1024.
static constexpr size_t kTopKMaxThreadsPerBlock = 1024;

template <typename T, size_t K>
void* GetTopKKernelForK(int n);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_COMMON_H_
