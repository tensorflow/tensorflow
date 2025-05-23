/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
// Number of blocks to launch the kernel in X dimension.
constexpr int64_t kLaunchBlockCountX = 8;
constexpr LaunchDimensions kLaunchDimensions(
    /*block_x_count=*/kLaunchBlockCountX,
    /*thread_x_count_per_block=*/512);

// Helper for allocating memory on the device.
absl::StatusOr<se::DeviceMemoryHandle> AllocateMemory(
    se::StreamExecutor* executor, int64_t size,
    absl::string_view debug_buffer_name) {
  se::DeviceMemoryHandle local_buffer_alloc(executor, executor->Allocate(size));
  if (local_buffer_alloc.memory().is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s for all-reduce.", debug_buffer_name));
  }
  return local_buffer_alloc;
};

}  // namespace

absl::Status CollectiveKernelThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  return resource_requests.AddClique(clique_key);
}

absl::Status CollectiveKernelThunk::RendezvousAfterInit(
    const GpuCliqueKey& clique_key, RankId rank, int64_t num_ranks) {
  std::string start_rendezvous_key =
      absl::StrFormat("start one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  // NB: This callback is called on a single thread after Rendezvous completion.
  auto completion_fn = [&]() -> bool {
    std::vector<const StreamState*> copy;
    {
      // We don't necessarily need a lock here since its only accessed by a
      // single thread. This keeps clang happy.
      absl::MutexLock lock(&mutex_);
      for (auto& [executor, state] : per_stream_state_) {
        copy.emplace_back(&state);
      }
    }
    // Sort by rank for stable order.
    absl::c_sort(copy,
                 [](const StreamState* const a, const StreamState* const b) {
                   return a->rank < b->rank;
                 });
    local_buffer_ptrs_.clear();
    signal_buffer_ptrs_.clear();
    for (auto const& state : copy) {
      local_buffer_ptrs_.emplace_back(state->local_buffer.memory());
      signal_buffer_ptrs_.emplace_back(state->signal_buffer.memory());
    }
    return true;
  };
  TF_ASSIGN_OR_RETURN(std::shared_ptr<bool> rendezvous_values,
                      Rendezvous<bool>(
                          /*name=*/start_rendezvous_key, /*key=*/clique_key,
                          /*num_threads=*/num_ranks, completion_fn));
  // Drop the rendezvous values. We don't need them once the callback is done.
  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::Initialize(const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  const int32_t kNumRanks = clique_key.num_devices();
  {
    absl::MutexLock lock(&mutex_);
    if (!per_stream_state_.contains(params.executor)) {
      // Step: Allocate local buffer
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle local_buffer_alloc,
          AllocateMemory(params.executor, buffer_.source_buffer.size(),
                         "LocalBuffer"));

      // Step: Allocate signal buffer
      // We needs 1 atomic flag per block per device on each device.
      const int64_t kNumSignalFlags =
          clique_key.num_local_participants() * kLaunchBlockCountX;
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle signal_flags_alloc,
          AllocateMemory(params.executor, kNumSignalFlags * sizeof(int32_t),
                         "SignalBuffer"));
      // One-shot kernel expects that the signal flags buffer is zeroed out.
      // Initial state of device memory is undefined, so we need to zero out the
      // buffer. The kernel will take care of leaving the buffer in correct
      // state after use, so we don't need to zero out after initialization.
      TF_RETURN_IF_ERROR(params.executor->SynchronousMemZero(
          signal_flags_alloc.memory_ptr(), signal_flags_alloc.memory().size()));

      // Step: Emplace into the stream state.
      per_stream_state_.emplace(
          params.executor,
          StreamState{rank.value(), std::move(local_buffer_alloc),
                      std::move(signal_flags_alloc)});
    }
  }
  return RendezvousAfterInit(clique_key, rank.value(), kNumRanks);
}

absl::Status xla::gpu::CollectiveKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  if (is_async_) {
    stream = params.collective_params->async_streams.at(
        static_cast<int64_t>(AsyncStreamKind::kCollective));
  }
  const int device_ordinal = stream->parent()->device_ordinal();
  VLOG(3) << "Performing one-shot all-reduce from device ordinal: "
          << device_ordinal;

  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  const int32_t kNumRanks = clique_key.num_devices();

  // TODO(b/407736956): Support variadic all-reduce.
  if (collective_config_.operand_element_type.size() != 1) {
    return absl::UnimplementedError(
        "Variadic arguments are not implemented for collective kernels.");
  }
  const PrimitiveType element_type = collective_config_.operand_element_type[0];
  se::DeviceMemoryBase source_buffer =
      params.buffer_allocations->GetDeviceAddress(buffer_.source_buffer);
  se::DeviceMemoryBase destination_buffer =
      params.buffer_allocations->GetDeviceAddress(buffer_.destination_buffer);

  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";

  // TODO(b/407736956): Change this to emitted kernel.
  return RunAllReduceKernel(/*stream=*/stream,
                            /*launch_dimensions=*/kLaunchDimensions,
                            /*element_type=*/element_type,
                            /*remote_input_buffers=*/local_buffer_ptrs_,
                            /*local_input_buffer=*/source_buffer,
                            /*output_buffer=*/destination_buffer,
                            /*rank=*/rank.value(),
                            /*num_ranks=*/kNumRanks,
                            /*num_elements=*/buffer_.element_count,
                            /*signal_flags_buffers=*/signal_buffer_ptrs_);
}

}  // namespace xla::gpu
