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

#include "xla/backends/gpu/runtime/collective_kernel_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// Launches a cross-GPU barrier synchronization.
//
// This implements a decentralized peer-to-peer barrier synchronization:
// 1. Each device maintains a signal buffer array (one slot per peer) and a
//    local monotonic step counter.
// 2. During execution, a device writes to its designated slot in *every*
//    peer's signal buffer to indicate arrival.
// 3. The device then waits locally for all slots in its own signal buffer to
//    match the expected step value (confirming all peers have arrived).
//
// See MultiGpuBarrierKernel for more details.
absl::Status CollectiveKernelUtils::LaunchMultiGpuBarrier(
    stream_executor::Stream* stream, int64_t num_devices, RankId rank,
    std::vector<stream_executor::DeviceAddressBase> barrier_addresses,
    stream_executor::DeviceAddressBase local_barrier_signal_value) {
  using MultiGpuBarrierKernel = stream_executor::gpu::MultiGpuBarrierKernel;

  TF_RET_CHECK(num_devices <= MultiGpuBarrierKernel::kMaxPeers)
      << "Number of participants exceeds MultiGpuBarrierKernel::kMaxPeers";
  TF_RET_CHECK(barrier_addresses.size() == num_devices)
      << "Number of barrier addresses does not match number of peers";

  std::array<void*, MultiGpuBarrierKernel::kMaxPeers> signal_buffers;
  std::fill(signal_buffers.begin(), signal_buffers.end(), nullptr);

  for (int peer = 0; peer < num_devices; ++peer) {
    signal_buffers[peer] = barrier_addresses[peer].opaque();
  }

  stream_executor::StreamExecutor* executor = stream->parent();
  TF_ASSIGN_OR_RETURN(
      auto kernel, (stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
                        .LoadKernel<MultiGpuBarrierKernel>(executor)));

  stream_executor::DeviceAddress<uint32_t> typed_sync_counter(
      local_barrier_signal_value);

  return kernel.Launch(
      stream_executor::ThreadDim(MultiGpuBarrierKernel::kMaxPeers, 1, 1),
      stream_executor::BlockDim(1, 1, 1), stream,
      static_cast<int64_t>(rank.value()), static_cast<int64_t>(num_devices),
      signal_buffers, typed_sync_counter);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
