/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_RAGGED_ALL_TO_ALL_KERNEL_COMMON_H_
#define XLA_SERVICE_GPU_KERNELS_RAGGED_ALL_TO_ALL_KERNEL_COMMON_H_

#include <cstdint>

namespace xla::gpu {

// Maximum number of output pointers that can be passed to the kernel.
inline constexpr int64_t kMaxNumRaggedAllToAllOutputPtrs = 8;

template <typename T>
void* GetRaggedAllToAllKernel();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_RAGGED_ALL_TO_ALL_KERNEL_COMMON_H_
