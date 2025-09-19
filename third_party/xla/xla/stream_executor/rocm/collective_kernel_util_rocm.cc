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

#include "xla/stream_executor/gpu/collective_kernel_util.cu.h"

namespace stream_executor::gpu {

template <>
__device__ __forceinline__ void PutSignalFlag<PlatformType::ROCM>(
    uint32_t* addr, uint32_t val) {
  __atomic_store_n(addr, val, __ATOMIC_RELEASE);
  __threadfence_system();  // Ensure visibility across all GPUs
}

template <>
__device__ __forceinline__ void WaitSignalFlag<PlatformType::ROCM>(
    uint32_t* addr, uint32_t expected) {
  uint32_t val;
  do {
    __threadfence_system();  // Ensure we see the latest value
    val = __atomic_load_n(addr, __ATOMIC_ACQUIRE);
  } while (val < expected);
}

}  // namespace stream_executor::gpu
