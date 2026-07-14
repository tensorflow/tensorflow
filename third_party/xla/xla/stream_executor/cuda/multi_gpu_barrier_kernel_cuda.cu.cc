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

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/collective_signal_cuda.cu.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/collective_signal.cu.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.cu.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::gpu {

__global__ void MultiGpuBarrierWithNcclKernelImpl(
    int64_t rank, int64_t num_ranks, ncclWindow_t signal_buffers_handle,
    uint32_t* sync_counter) {
  // 1. Get individual signal buffers pointers.
  std::array<uint32_t* __restrict__, MultiGpuBarrierWithNcclKernel::kMaxPeers>
      signal_buffers;

#pragma unroll
  for (int64_t i = 0; i < MultiGpuBarrierWithNcclKernel::kMaxPeers; ++i) {
    if (i < num_ranks) {
      signal_buffers[i] = reinterpret_cast<uint32_t*>(
          ncclGetLsaPointer(signal_buffers_handle, 0, i));
    }
  }

  SyncRemoteBlocksAndUpdateCounter<PlatformType::kCuda>(
      rank, num_ranks, signal_buffers, sync_counter);
}

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MultiGpuBarrierKernelCuda,                    // 1. Identifier
    stream_executor::gpu::MultiGpuBarrierKernel,  // 2. KernelTrait
    stream_executor::cuda::kCudaPlatformId,       // 3. Platform ID
    ([](size_t arity) {                           // 4. Kernel Spec Creator
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(
              &MultiGpuBarrierKernelImpl<PlatformType::kCuda>),
          "multi_gpu_barrier_kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MultiGpuBarrierWithNcclKernelCuda,                    // 1. Identifier
    stream_executor::gpu::MultiGpuBarrierWithNcclKernel,  // 2. KernelTrait
    stream_executor::cuda::kCudaPlatformId,               // 3. Platform ID
    ([](size_t arity) {  // 4. Kernel Spec Creator
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&MultiGpuBarrierWithNcclKernelImpl),
          "multi_gpu_barrier_nccl_kernel", arity);
    }));

}  // namespace stream_executor::gpu
