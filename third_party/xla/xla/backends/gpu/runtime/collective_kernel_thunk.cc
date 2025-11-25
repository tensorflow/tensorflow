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
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {
using se::gpu::AllReduceStrategy;

static constexpr int64_t kMaxOneShotAllReduceSizeBytes = 256 * 1024;  // 256 KB
static constexpr int64_t kMaxTwoShotAllReduceSizeBytes =
    2 * 1024 * 1024;  // 2 MB
// Number of arguments for the all-reduce kernel.
// - Metadata pointer.
// - Input buffer pointer.
// - Output buffer pointer.
// - Num elements.
// - Num elements per rank.
// - Rank offset.
// - Signal value.
static constexpr int kAllReduceArgsCount = 7;

// Helper for allocating memory on the device.
absl::StatusOr<se::DeviceMemoryHandle> AllocateMemory(
    se::StreamExecutor* executor, int64_t size,
    absl::string_view debug_buffer_name) {
  se::DeviceMemoryHandle local_buffer_alloc(
      executor,
      executor->Allocate(
          size, static_cast<int64_t>(stream_executor::MemoryType::kP2P)));
  if (local_buffer_alloc.memory().is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s for all-reduce.", debug_buffer_name));
  }
  return local_buffer_alloc;
};

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
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(input_size_bytes, is_multimem_enabled_);
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

absl::Status CollectiveKernelThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params != nullptr);
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*use_nccl=*/false));
  return params.clique_requests->RequestClique(clique_key);
}

int64_t CollectiveKernelThunk::GetInputSizeBytes() const {
  return buffers_[0].element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(
             collective_config_.operand_element_type[0]);
}

absl::Status CollectiveKernelThunk::ExchangeStateMetadata(
    const GpuCliqueKey& clique_key, const InitializeParams& params,
    StreamState& state) {
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";

  std::vector<se::DeviceMemoryBase> parameters;
  parameters.push_back(state.local_buffers_handle.memory());
  parameters.push_back(state.signal_buffers_handle.memory());

  const size_t param_to_peers_ptrs_size_bytes =
      parameters.size() * clique_key.num_devices() * sizeof(uint64_t);
  state.metadata = params.executor->Allocate(
      sizeof(CollectiveKernelMetadata) + param_to_peers_ptrs_size_bytes, 0);
  return CollectiveMetadataThunk::ConstructCollectiveMetadata(
      std::move(parameters), params.stream, clique_key,
      state.multicast_device_ptr, rank.value().value(), state.metadata);
}

absl::Status CollectiveKernelThunk::Initialize(const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*use_nccl=*/false));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(GetInputSizeBytes(), is_multimem_enabled_);
  const LaunchDimensions launch_dimensions = AllReduceLaunchDimensions(
      buffers_[0].element_count, clique_key.num_local_participants(), strategy);

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    if (!per_stream_state_.contains(params.executor)) {
      // Step1: Allocate signal and local buffers.
      const int64_t kNumSignalFlags =
          clique_key.num_local_participants() * launch_dimensions.num_blocks();

      int64_t kSignalBufferSize = xla::RoundUpTo<uint64_t>(
          kNumSignalFlags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
      const int64_t kLocalBufferSize = xla::RoundUpTo<uint64_t>(
          buffers_[0].source_buffer.size(), kXlaAllocatedBufferAlignBytes);

      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle local_buffers_handle,
          AllocateMemory(params.executor, kLocalBufferSize * kNumBuffers,
                         "Local buffers"));

      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryHandle signal_buffers_handle,
          AllocateMemory(params.executor, kLocalBufferSize * kNumBuffers,
                         "Signal buffers"));

      // Step2: We needs 1 atomic flag per block per device on each device.
      // One-shot kernel expects that the signal flags buffer is zeroed out.
      // Initial state of device memory is undefined, so we need to zero out
      // the buffer. The kernel will take care of leaving the buffer in
      // correct state after use, so we don't need to zero out after
      // initialization.
      TF_RETURN_IF_ERROR(params.executor->SynchronousMemZero(
          signal_buffers_handle.memory_ptr(),
          signal_buffers_handle.memory().size()));
      // Create a kernel for execution.
      std::unique_ptr<se::Kernel> kernel = nullptr;
      // If PTX is provided, we create a kernel from it.
      if (!kernel_name_.empty()) {
        VLOG(3) << "Creating kernel from PTX." << params.src.text;
        TF_ASSIGN_OR_RETURN(kernel,
                            CreateKernel(kernel_name_, kAllReduceArgsCount,
                                         params.src.text, params.executor, 0));
      }
      // Step3: Emplace into the stream state.
      per_stream_state_.emplace(
          params.executor,
          std::make_unique<StreamState>(
              params.executor->device_ordinal(), rank.value(),
              std::move(local_buffers_handle), std::move(signal_buffers_handle),
              std::move(kernel)));

      state = per_stream_state_.at(params.executor).get();

      // NB: This is a double buffer allocation. So size of a single buffer is
      // half of the total allocation.
      for (int i = 0; i < kNumBuffers; ++i) {
        state->remote_buffer_ptrs[i] =
            state->local_buffers_handle.memory_ptr()->GetByteSlice(
                /*offset_bytes=*/i * kLocalBufferSize,
                /*size_bytes=*/kLocalBufferSize);

        state->signal_buffer_ptrs[i] =
            state->signal_buffers_handle.memory_ptr()->GetByteSlice(
                /*offset_bytes=*/i * kSignalBufferSize,
                /*size_bytes=*/kSignalBufferSize);
      }
    }
  }

  if (state != nullptr) {
    if (strategy == AllReduceStrategy::kMultimem) {
      TF_ASSIGN_OR_RETURN(state->multicast_device_ptr,
                          address_space_provider_.SetupMultimemAddressSpace(
                              clique_key, params.executor,
                              state->local_buffers_handle.memory()));
    }
    TF_RETURN_IF_ERROR(ExchangeStateMetadata(clique_key, params, *state));
  }

  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  if (is_async_) {
    stream = params.collective_params->async_streams.at(
        static_cast<int64_t>(AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));
  }
  const int device_ordinal = stream->parent()->device_ordinal();

  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*use_nccl=*/false));
  const int32_t num_devices = clique_key.num_devices();

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
    absl::MutexLock lock(mutex_);
    auto it = per_stream_state_.find(stream->parent());
    TF_RET_CHECK(it != per_stream_state_.end())
        << "Stream not found in per_stream_state_";
    state = it->second.get();
  }

  const uint32_t buffer_index = state->invocation_count % kNumBuffers;
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(GetInputSizeBytes(), is_multimem_enabled_);
  const LaunchDimensions launch_dimensions =
      AllReduceLaunchDimensions(buffer.element_count, num_devices, strategy);
  // In case of two-shot we want to increment in multiples of 2.
  state->invocation_count += 1 + static_cast<uint32_t>(strategy);
  VLOG(3) << "[" << device_ordinal
          << "] Performing one-shot all-reduce for clique "
          << clique_key.ToString();

  se::DeviceMemoryBase input_buffer_ptr =
      state->remote_buffer_ptrs[buffer_index];
  se::DeviceMemoryBase signal_buffer_ptr =
      state->signal_buffer_ptrs[buffer_index];
  VLOG(3) << "[" << device_ordinal
          << "] input_buffer_ptr: " << input_buffer_ptr.opaque()
          << " signal_buffer_ptr: " << signal_buffer_ptr.opaque();
  VLOG(3) << "[" << device_ordinal
          << "] launch dimensions: " << launch_dimensions.num_blocks() << "x"
          << launch_dimensions.num_threads_per_block()
          << "(block x threadsPerBlock)";

  if (state->kernel != nullptr) {
    // NB: The assumption is one-shot all-reduce (for now).
    std::vector<se::KernelArgument> kernel_args = {
        state->metadata,
        source_buffer,
        destination_buffer,
        buffer.element_count,
        /*num_elements_per_rank=*/buffer.element_count / num_devices,
        /*rank_offset=*/0,
        state->invocation_count};
    TF_RET_CHECK(kernel_args.size() == kAllReduceArgsCount)
        << "Kernel argument size mismatch." << kernel_args.size()
        << " != " << kAllReduceArgsCount;
    return ExecuteKernelOnStream(*state->kernel, kernel_args, launch_dimensions,
                                 /*cluster_dim=*/std::nullopt, stream);
  }

  // TODO(b/407736956): Change this to emitted kernel.
  return RunAllReduceKernel(
      /*stream=*/stream,
      /*launch_dimensions=*/launch_dimensions,
      /*element_type=*/element_type,
      /*reduction_kind=*/reduction_kind_,
      /*all_reduce_strategy=*/strategy,
      /*symmetric_input_buffer=*/input_buffer_ptr,
      /*local_input_buffer=*/source_buffer,
      /*output_buffer=*/destination_buffer,
      /*rank=*/rank.value(),
      /*num_ranks=*/num_devices,
      /*num_elements=*/buffer.element_count,
      /*symmetric_signal_buffer=*/signal_buffer_ptr,
      /*signal_value=*/state->invocation_count,
      /*metadata=*/state->metadata);
}

}  // namespace xla::gpu
