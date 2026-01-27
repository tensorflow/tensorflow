/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_SIGNAL_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_SIGNAL_CU_H_

#include <cstdint>

namespace stream_executor::gpu {

enum class PlatformType : uint32_t {
  kRocm,
  kCuda,
};

// -----------------------------------------------------------------------------
// Synchronization Primitives
// -----------------------------------------------------------------------------

template <PlatformType T>
__device__ void PutSignalFlag(uint32_t* addr, uint32_t val);

template <PlatformType T>
__device__ void WaitSignalFlag(uint32_t* addr, uint32_t expected);

// Generic Barrier across blocks.
// MaxPeers is templated to allow reuse between different kernels.
template <PlatformType T, int64_t MaxPeers>
__device__ __forceinline__ void SyncRemoteBlocks(
    // Use raw pointer with __restrict__ directly here
    std::array<uint32_t* __restrict__, MaxPeers> signal_pad_ptrs, int64_t rank,
    int64_t num_ranks, uint32_t signal_value) {
  if (threadIdx.x < num_ranks) {
    auto target_rank = threadIdx.x;
    PutSignalFlag<T>(
        signal_pad_ptrs[target_rank] + blockIdx.x * num_ranks + rank,
        signal_value);
    WaitSignalFlag<T>(
        signal_pad_ptrs[rank] + blockIdx.x * num_ranks + target_rank,
        signal_value);
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_SIGNAL_CU_H_
