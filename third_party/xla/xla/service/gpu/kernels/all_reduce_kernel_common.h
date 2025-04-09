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

#ifndef XLA_SERVICE_GPU_KERNELS_ALL_REDUCE_KERNEL_COMMON_H_
#define XLA_SERVICE_GPU_KERNELS_ALL_REDUCE_KERNEL_COMMON_H_

#include <cstdint>

namespace xla::gpu {

// The maximum number of input pointers that can be passed to the all-reduce
// kernel.
inline constexpr int64_t kMaxNumAllReduceInputPtrs = 8;

// Returns a pointer to the all-reduce kernel for the given element type.
// Returns nullptr if the element type is not supported.
template <typename T>
void* GetAllReduceKernel();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNELS_ALL_REDUCE_KERNEL_COMMON_H_
