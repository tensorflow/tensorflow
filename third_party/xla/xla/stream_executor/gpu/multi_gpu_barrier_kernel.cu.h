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

#ifndef XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_CU_H_

#include <array>
#include <cassert>
#include <cstdint>

#include "xla/stream_executor/gpu/collective_signal.cu.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"

namespace stream_executor::gpu {

template <PlatformType PlatformT, int64_t MaxPeers>
__device__ __forceinline__ void SyncRemoteBlocksAndUpdateCounter(
    int64_t rank, int64_t num_ranks,
    const std::array<uint32_t* __restrict__, MaxPeers>& signal_buffers,
    uint32_t* sync_counter) {
  uint32_t signal_value = *sync_counter + 1;

  SyncRemoteBlocks<PlatformT, MaxPeers>(signal_buffers, rank, num_ranks,
                                        signal_value);

  // CTA-wide barrier: all threads must finish their remote puts/waits before
  // thread 0 updates sync_counter.
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *sync_counter = signal_value;
  }
}

template <PlatformType PlatformT>
__global__ void MultiGpuBarrierKernelImpl(
    int64_t rank, int64_t num_ranks,
    std::array<void*, MultiGpuBarrierKernel::kMaxPeers> signal_buffers_void,
    uint32_t* sync_counter) {
  __shared__
      std::array<uint32_t* __restrict__, MultiGpuBarrierKernel::kMaxPeers>
          signal_buffers;

  for (int64_t i = threadIdx.x; i < num_ranks; i += blockDim.x) {
    signal_buffers[i] = reinterpret_cast<uint32_t*>(signal_buffers_void[i]);
  }
  __syncthreads();

  SyncRemoteBlocksAndUpdateCounter<PlatformT, MultiGpuBarrierKernel::kMaxPeers>(
      rank, num_ranks, signal_buffers, sync_counter);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_CU_H_
