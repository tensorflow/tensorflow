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
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
using se::gpu::AllReduceStrategy;

static constexpr int64_t kMaxOneShotAllReduceSizeBytes = 256 * 1024;  // 256 KB
static constexpr int64_t kMaxTwoShotAllReduceSizeBytes =
    2 * 1024 * 1024;  // 2 MB

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

AllReduceStrategy GetAllReduceStrategy(int64_t input_size_bytes) {
  return input_size_bytes > kMaxOneShotAllReduceSizeBytes
             ? AllReduceStrategy::kTwoShot
             : AllReduceStrategy::kOneShot;
}

int64_t GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      return kMaxOneShotAllReduceSizeBytes;
    case AllReduceStrategy::kTwoShot:
      return kMaxTwoShotAllReduceSizeBytes;
  }
}

}  // namespace

absl::StatusOr<bool> CollectiveKernelThunk::IsSupported(
    const GpuCliqueKey& clique_key,
    const CollectiveCliques* collective_cliques) const {
  if (!collective_kernel_enabled_) {
    return false;
  }

  // TODO(b/407736956): Support variadic all-reduce.
  if (buffers_.size() != 1) {
    return false;
  }

  const int64_t num_elements = buffers_[0].element_count;
  const int64_t input_size_bytes = GetInputSizeBytes();
  const AllReduceStrategy strategy = GetAllReduceStrategy(input_size_bytes);
  // Comment the next two lines for testing out two-shot.
  if (strategy != AllReduceStrategy::kOneShot) {
    return false;
  }
  // Custom all-reduce strategy is only supported for small inputs.
  if (input_size_bytes > GetMaxSupportedAllReduceSizeBytes(strategy)) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool peer_access_enabled,
                      collective_cliques->peer_access_enabled(clique_key));

  // Check that peer access is enabled.
  if (!peer_access_enabled) {
    return false;
  }

  return IsAllReduceKernelSupported(
      clique_key.num_local_participants(), num_elements,
      collective_config_.operand_element_type[0], reduction_kind_, strategy);
}

absl::Status CollectiveKernelThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  return resource_requests.AddClique(clique_key);
}

int64_t CollectiveKernelThunk::GetInputSizeBytes() const {
  return buffers_[0].element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(
             collective_config_.operand_element_type[0]);
}

absl::Status CollectiveKernelThunk::RendezvousAfterInit(
    const GpuCliqueKey& clique_key, StreamState& state) {
  const int64_t num_ranks = clique_key.num_devices();
  std::string start_rendezvous_key = absl::StrFormat(
      "Initializing one-shot all-reduce for device %d, clique %s",
      state.device_ordinal, clique_key.ToString());
  // NB: This callback is called on one thread per participating group.
  // i.e.; If participating groups are {{0,1},{2,3}} then it would be called
  // twice. Once with StreamStates for 0,1 and once with StreamStates for 2,3.
  auto completion_fn = [](absl::Span<const StreamState*> states)
      -> std::vector<const StreamState*> {
    std::vector<const StreamState*> copy(states.begin(), states.end());
    // Sort by rank for stable order.
    absl::c_sort(copy,
                 [](const StreamState* const a, const StreamState* const b) {
                   return a->rank < b->rank;
                 });
    return copy;
  };
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<const StreamState*>> rendezvous_values,
      Rendezvous<std::vector<const StreamState*>>(
          /*name=*/start_rendezvous_key, /*key=*/clique_key,
          /*value=*/state,
          /*num_threads=*/num_ranks, completion_fn));

  // Sanity check to ensure that Rendezvous() was called only once.
  for (int i = 0; i < state.remote_buffer_ptrs.size(); ++i) {
    TF_RET_CHECK(state.remote_buffer_ptrs[i].empty())
        << "Remote buffer ptrs was expected to be empty. Was: "
        << state.remote_buffer_ptrs[i].size();
    TF_RET_CHECK(state.signal_buffer_ptrs[i].empty())
        << "Signal buffer ptrs was expected to be empty. Was: "
        << state.signal_buffer_ptrs[i].size();
  }
  for (auto* rendezvous_state : *rendezvous_values) {
    // NB: This is a double buffer allocation. So size of a single buffer is
    // half of the total allocation.
    const int64_t buffer_size =
        rendezvous_state->local_buffer.memory().size() / kNumBuffers;
    const int64_t signal_buffer_size =
        rendezvous_state->signal_buffer.memory().size() / kNumBuffers;
    for (int i = 0; i < state.remote_buffer_ptrs.size(); ++i) {
      state.remote_buffer_ptrs[i].emplace_back(
          rendezvous_state->local_buffer.memory().GetByteSlice(
              /*offset_bytes=*/i * buffer_size,
              /*size_bytes=*/buffer_size));
      state.signal_buffer_ptrs[i].emplace_back(
          rendezvous_state->signal_buffer.memory().GetByteSlice(
              /*offset_bytes=*/i * signal_buffer_size,
              /*size_bytes=*/signal_buffer_size));
    }
  }
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
  StreamState* state = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    if (!per_stream_state_.contains(params.executor)) {
      // Step1: Allocate local buffer
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle local_buffer_alloc,
          AllocateMemory(params.executor,
                         buffers_[0].source_buffer.size() * kNumBuffers,
                         "LocalBuffer"));

      // Step2: Allocate signal buffer
      // We needs 1 atomic flag per block per device on each device.
      const int64_t kNumSignalFlags =
          clique_key.num_local_participants() * kLaunchBlockCountX;
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle signal_flags_alloc,
          AllocateMemory(params.executor,
                         kNumSignalFlags * sizeof(int32_t) * kNumBuffers,
                         "SignalBuffer"));
      // One-shot kernel expects that the signal flags buffer is zeroed out.
      // Initial state of device memory is undefined, so we need to zero out
      // the buffer. The kernel will take care of leaving the buffer in
      // correct state after use, so we don't need to zero out after
      // initialization.
      TF_RETURN_IF_ERROR(params.executor->SynchronousMemZero(
          signal_flags_alloc.memory_ptr(), signal_flags_alloc.memory().size()));

      // Step3: Emplace into the stream state.
      per_stream_state_.emplace(
          params.executor,
          std::make_unique<StreamState>(
              params.executor->device_ordinal(), rank.value(),
              std::move(local_buffer_alloc), std::move(signal_flags_alloc)));
      state = per_stream_state_.at(params.executor).get();
    }
  }
  // Only invoke rendezvous if a new state was initialized.
  if (state != nullptr) {
    return RendezvousAfterInit(clique_key, *state);
  }
  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  if (is_async_) {
    stream = params.collective_params->async_streams.at(
        static_cast<int64_t>(AsyncStreamKind::kCollective));
  }
  const int device_ordinal = stream->parent()->device_ordinal();

  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  const int32_t kNumRanks = clique_key.num_devices();

  // TODO(b/407736956): Support variadic all-reduce.
  if (collective_config_.operand_element_type.size() != 1) {
    return absl::UnimplementedError(
        "Variadic arguments are not implemented for collective kernels.");
  }
  const CollectiveThunk::Buffer& buffer = buffers_[0];
  const PrimitiveType element_type = collective_config_.operand_element_type[0];
  se::DeviceMemoryBase source_buffer =
      params.buffer_allocations->GetDeviceAddress(buffer.source_buffer);
  se::DeviceMemoryBase destination_buffer =
      params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer);

  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  StreamState* state = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    auto it = per_stream_state_.find(stream->parent());
    TF_RET_CHECK(it != per_stream_state_.end())
        << "Stream not found in per_stream_state_";
    state = it->second.get();
  }
  const uint32_t buffer_index = state->invocation_count % kNumBuffers;
  auto const strategy = GetAllReduceStrategy(GetInputSizeBytes());
  // In case of two-shot we want to increment in multiples of 2.
  state->invocation_count += 1 + static_cast<uint32_t>(strategy);
  VLOG(3) << "Performing one-shot all-reduce from device ordinal: "
          << device_ordinal << " for clique " << clique_key.ToString();
  // TODO(b/407736956): Change this to emitted kernel.
  return RunAllReduceKernel(
      /*stream=*/stream,
      /*launch_dimensions=*/kLaunchDimensions,
      /*element_type=*/element_type,
      /*reduction_kind=*/reduction_kind_,
      /*all_reduce_strategy=*/strategy,
      /*remote_input_buffers=*/state->remote_buffer_ptrs[buffer_index],
      /*local_input_buffer=*/source_buffer,
      /*output_buffer=*/destination_buffer,
      /*rank=*/rank.value(),
      /*num_ranks=*/kNumRanks,
      /*num_elements=*/buffer.element_count,
      /*signal_flags_buffers=*/state->signal_buffer_ptrs[buffer_index],
      /*signal_value=*/state->invocation_count);
}

}  // namespace xla::gpu
