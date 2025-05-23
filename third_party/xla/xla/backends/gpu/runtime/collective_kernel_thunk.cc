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

// Contains the values that are passed between host threads with rendezvous.
struct RendezvousValue {
  RankId rank;
  se::DeviceMemoryBase input_buffer;
  se::DeviceMemoryBase signal_flags_buffer;

  bool operator<(const RendezvousValue& other) const {
    return rank < other.rank;
  }
};

// Once rendezvous is done all values are collected and sorted by rank.
std::vector<RendezvousValue> RendezvousCallback(
    absl::Span<const RendezvousValue* const> values) {
  std::vector<RendezvousValue> values_copy;
  for (const auto& value : values) {
    values_copy.push_back(*value);
  }
  // Sort to make sure that values are in the same order as the devices
  // are ordered in the communicator.
  absl::c_sort(values_copy);
  return values_copy;
};

// Executes the rendezvous before the kernel start.
absl::StatusOr<std::shared_ptr<std::vector<RendezvousValue>>>
RendezvousBeforeKernelStart(const GpuCliqueKey& clique_key, RankId rank,
                            int64_t num_ranks,
                            const se::DeviceMemoryBase& input_buffer,
                            se::Stream& stream,
                            const se::DeviceMemoryBase& signal_flags_buffer) {
  RendezvousValue rendezvous_value;
  rendezvous_value.rank = rank;
  rendezvous_value.input_buffer = input_buffer;
  rendezvous_value.signal_flags_buffer = signal_flags_buffer;

  std::string start_rendezvous_key =
      absl::StrFormat("start one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<RendezvousValue>> rendezvous_values,
      Rendezvous<std::vector<RendezvousValue>>(
          /*name=*/start_rendezvous_key, /*key=*/clique_key,
          /*value=*/rendezvous_value, /*num_threads=*/num_ranks,
          RendezvousCallback));

  return rendezvous_values;
}

}  // namespace

absl::Status CollectiveKernelThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  return resource_requests.AddClique(clique_key);
}

absl::Status CollectiveKernelThunk::Initialize(const InitializeParams& params) {
  {
    absl::MutexLock lock(&mutex_);
    if (!local_buffer_allocs_.contains(params.executor)) {
      se::DeviceMemoryHandle local_buffer_alloc(
          params.executor,
          params.executor->Allocate(buffer_.source_buffer.size()));
      local_buffer_allocs_.emplace(params.executor,
                                   std::move(local_buffer_alloc));
    }
    if (!signal_flags_allocs_.contains(params.executor)) {
      TF_ASSIGN_OR_RETURN(const GpuCliqueKey clique_key,
                          GetCollectiveGpuCliqueKey(*params.collective_params,
                                                    collective_config_));
      // We needs 1 atomic flag per block per device on each device.
      const int64_t kNumSignalFlags =
          clique_key.num_local_participants() * kLaunchBlockCountX;

      se::DeviceMemoryHandle signal_flags_alloc{
          params.executor,
          params.executor->Allocate(kNumSignalFlags * sizeof(int32_t))};
      if (signal_flags_alloc.memory().is_null()) {
        return absl::InternalError("Failed to allocate signal pads buffer.");
      }

      // One-shot kernel expects that the signal flags buffer is zeroed out.
      // Initial state of device memory is undefined, so we need to zero out the
      // buffer. The kernel will take care of leaving the buffer in correct
      // state after use, so we don't need to zero out only during
      // initialization.
      TF_RETURN_IF_ERROR(params.executor->SynchronousMemZero(
          signal_flags_alloc.memory_ptr(), signal_flags_alloc.memory().size()));

      signal_flags_allocs_.emplace(params.executor,
                                   std::move(signal_flags_alloc));
    }
  }
  return absl::OkStatus();
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
  se::DeviceMemoryBase local_buffer;
  se::DeviceMemoryBase signal_flags_buffer;
  {
    absl::MutexLock lock(&mutex_);
    local_buffer = local_buffer_allocs_[stream->parent()].memory();
    signal_flags_buffer = signal_flags_allocs_[stream->parent()].memory();
  }

  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<RendezvousValue>> rendezvous_values,
      RendezvousBeforeKernelStart(clique_key, rank.value(), kNumRanks,
                                  local_buffer, *stream, signal_flags_buffer));

  absl::InlinedVector<se::DeviceMemoryBase, 8> input_ptrs;
  absl::InlinedVector<se::DeviceMemoryBase, 8> signal_flags_ptrs;
  for (auto& value : *rendezvous_values) {
    input_ptrs.push_back(value.input_buffer);
    signal_flags_ptrs.push_back(value.signal_flags_buffer);
  }

  // TODO(b/407736956): Change this to emitted kernel.
  return RunAllReduceKernel(/*stream=*/stream,
                            /*launch_dimensions=*/kLaunchDimensions,
                            /*element_type=*/element_type,
                            /*remote_input_buffers=*/input_ptrs,
                            /*local_input_buffer=*/source_buffer,
                            /*output_buffer=*/destination_buffer,
                            /*rank=*/rank.value(),
                            /*num_ranks=*/kNumRanks,
                            /*num_elements=*/buffer_.element_count,
                            /*signal_flags_buffers=*/signal_flags_ptrs);
}

}  // namespace xla::gpu
