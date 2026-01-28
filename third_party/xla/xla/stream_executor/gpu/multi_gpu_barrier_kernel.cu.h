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

template <PlatformType PlatformT>
__global__ void MultiGpuBarrierKernelImpl(
    int64_t rank, int64_t num_ranks,
    std::array<void*, MultiGpuBarrierKernel::kMaxPeers> signal_buffers_void,
    uint32_t* sync_counter) {
  // 1. Cast void* pointers
  std::array<uint32_t* __restrict__, MultiGpuBarrierKernel::kMaxPeers>
      signal_buffers;

#pragma unroll
  for (int64_t i = 0; i < MultiGpuBarrierKernel::kMaxPeers; ++i) {
    signal_buffers[i] = reinterpret_cast<uint32_t*>(signal_buffers_void[i]);
  }

  // 2. Read State
  // Every rank maintains its own counter, so this read is local and fast.
  // Since we launch 1 block, a standard load is sufficient, as all threads see
  // the same value.
  uint32_t signal_value = *sync_counter;

  // 3. Barrier
  SyncRemoteBlocks<PlatformT, MultiGpuBarrierKernel::kMaxPeers>(
      signal_buffers, rank, num_ranks, signal_value);

  // 4. Update State
  // Only the first thread needs to update the counter for the NEXT execution.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *sync_counter = signal_value + 1;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_CU_H_
