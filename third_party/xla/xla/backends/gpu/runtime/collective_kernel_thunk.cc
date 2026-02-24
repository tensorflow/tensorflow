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

#include <array>
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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {
using se::gpu::AllReduceStrategy;

// Number of arguments for the all-reduce kernel.
// - Input buffer pointer.
// - Output buffer pointer.
// - Rank
// - Signal value.
// - Signal buffers
// - Remote buffers
static constexpr int32_t kAllReduceArgsCount = 6;
static constexpr int32_t kNumParameters = 2;

// Helper for allocating memory on the device.
absl::StatusOr<se::DeviceAddressHandle> AllocateMemory(
    se::StreamExecutor* executor, int64_t size,
    absl::string_view debug_buffer_name) {
  se::DeviceAddressHandle local_buffer_alloc(
      executor,
      executor->Allocate(
          size, static_cast<int64_t>(stream_executor::MemoryType::kP2P)));
  if (local_buffer_alloc.memory().is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s for all-reduce.", debug_buffer_name));
  }
  return local_buffer_alloc;
};

absl::StatusOr<int> GetLocalDeviceId(
    const GlobalDeviceId& global_device_id,
    const CollectiveParams& collective_params) {
  // If the global device id map is not provided, then we can assume that
  // execution is local.
  if (!collective_params.global_device_id_map) {
    return global_device_id.value();
  }

  for (const auto& local_device : *collective_params.global_device_id_map) {
    if (local_device.second == global_device_id) {
      return local_device.first.value();
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Global device id %d not found in global device id map.",
                      global_device_id.value()));
}

struct PtrFormatter {
  void operator()(std::string* out, const void* ptr) const {
    absl::StrAppend(out, absl::StrFormat("%p", ptr));
  }
};

absl::Status CopyCollectiveMetadataToDevice(
    se::Stream* stream, CollectiveKernelMetadata metadata,
    const std::vector<void*>& param_to_peers_ptrs,
    const std::vector<void*>& multimem_addresses,
    se::DeviceAddressBase destination) {
  const int64_t param_to_peers_ptrs_size =
      param_to_peers_ptrs.size() * sizeof(void*);
  se::DeviceAddressBase param_to_peers_ptrs_buffer = destination.GetByteSlice(
      sizeof(CollectiveKernelMetadata), param_to_peers_ptrs_size);

  const int64_t multimem_addresses_size =
      multimem_addresses.size() * sizeof(void*);
  se::DeviceAddressBase multimem_addresses_buffer = destination.GetByteSlice(
      sizeof(CollectiveKernelMetadata) + param_to_peers_ptrs_size,
      multimem_addresses_size);

  metadata.param_to_peers =
      reinterpret_cast<void**>(param_to_peers_ptrs_buffer.opaque());
  metadata.param_to_multimem_addresses =
      reinterpret_cast<void**>(multimem_addresses_buffer.opaque());
  TF_RETURN_IF_ERROR(stream->Memcpy(&destination, &metadata,
                                    sizeof(CollectiveKernelMetadata)));
  TF_RETURN_IF_ERROR(stream->Memcpy(&param_to_peers_ptrs_buffer,
                                    param_to_peers_ptrs.data(),
                                    param_to_peers_ptrs_size));
  TF_RETURN_IF_ERROR(stream->Memcpy(&multimem_addresses_buffer,
                                    multimem_addresses.data(),
                                    multimem_addresses_size));
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> CollectiveKernelThunk::IsSupported(
    const GpuCliqueKey& clique_key, se::StreamExecutor& executor,
    const CollectiveParams& collective_params) const {
  if (!collective_kernel_enabled_) {
    XLA_VLOG_DEVICE(3, executor.device_ordinal())
        << "Collective kernel is not enabled.";
    return false;
  }

  // TODO(b/407736956): Support variadic all-reduce.
  if (buffers_.size() != 1) {
    XLA_VLOG_DEVICE(3, executor.device_ordinal())
        << "Variadic arguments are not implemented for collective kernels.";
    return false;
  }
  const int64_t num_elements = buffers_[0].element_count;
  const int64_t input_size_bytes = GetInputSizeBytes();
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(input_size_bytes, is_multimem_enabled_);
  // Custom all-reduce strategy is only supported for small inputs.
  if (input_size_bytes > GetMaxSupportedAllReduceSizeBytes(strategy)) {
    XLA_VLOG_DEVICE(3, executor.device_ordinal())
        << "Custom all-reduce strategy is only supported for small inputs.";
    return false;
  }

  // Only single-host collectives are supported for now.
  if (!clique_key.is_local()) {
    XLA_VLOG_DEVICE(3, executor.device_ordinal())
        << "Cross-host symmetric memory collectives are not supported.";
    return false;
  }

  for (const GlobalDeviceId& device : clique_key.devices()) {
    TF_ASSIGN_OR_RETURN(const int peer_device_id,
                        GetLocalDeviceId(device, collective_params));
    if (!executor.CanEnablePeerAccessTo(peer_device_id)) {
      XLA_VLOG_DEVICE(3, executor.device_ordinal())
          << "Peer access is not supported with device " << peer_device_id;
      return false;
    }
  }

  return IsAllReduceKernelSupported(
      clique_key.num_local_participants(), num_elements,
      collective_config_.operand_element_type[0], reduction_kind_, strategy);
}

absl::Status CollectiveKernelThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*is_p2p=*/false));

  TF_ASSIGN_OR_RETURN(
      bool use_collective_kernel,
      IsSupported(clique_key, *params.executor, *params.collective_params));

  if (!use_collective_kernel) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetParticipatingDevicesGroups(*params.collective_params->device_assn,
                                    collective_config_.replica_groups,
                                    collective_config_.group_mode));

  TF_RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
      clique_key, std::move(device_groups)));

  absl::MutexLock lock(mutex_);
  if (!per_stream_memory_.contains(params.executor)) {
    // Allocate scratch buffers.
    const AllReduceStrategy strategy =
        GetAllReduceStrategy(GetInputSizeBytes(), is_multimem_enabled_);
    const LaunchDimensions launch_dimensions = AllReduceLaunchDimensions(
        buffers_[0].element_count, clique_key.num_local_participants(),
        strategy);
    const int64_t kNumSignalFlags =
        clique_key.num_local_participants() * launch_dimensions.num_blocks();
    const int64_t kSignalBufferSize = xla::RoundUpTo<uint64_t>(
        kNumSignalFlags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
    const int64_t kLocalBufferSize = xla::RoundUpTo<uint64_t>(
        buffers_[0].source_buffer.slice.size(), kXlaAllocatedBufferAlignBytes);
    TF_ASSIGN_OR_RETURN(
        se::DeviceAddressHandle local_buffers_handle,
        AllocateMemory(params.executor, kLocalBufferSize * kNumBuffers,
                       "Local buffers"));

    TF_ASSIGN_OR_RETURN(
        se::DeviceAddressHandle signal_buffers_handle,
        AllocateMemory(params.executor, kSignalBufferSize * kNumBuffers,
                       "Signal buffers"));

    per_stream_memory_.emplace(
        params.executor,
        std::make_unique<StreamMemory>(StreamMemory{
            std::move(local_buffers_handle), std::move(signal_buffers_handle),
            strategy, kLocalBufferSize, kSignalBufferSize}));

    // If we decided to run kernel using multimem strategy we request multimem
    // addresses for input and output buffers (both of them must be allocated
    // from the collective allocator at run time).
    auto& stream_memory = per_stream_memory_.at(params.executor);
    if (stream_memory->strategy == AllReduceStrategy::kMultimem) {
      XLA_VLOG_DEVICE(3, params.executor->device_ordinal())
          << "Request multicast address for source and destination buffers";

      TF_RETURN_IF_ERROR(
          params.collective_memory_requests->RequestMulticastAddress(
              clique_key, params.buffer_allocations->GetDeviceAddress(
                              buffers_[0].source_buffer.slice)));
      TF_RETURN_IF_ERROR(
          params.collective_memory_requests->RequestMulticastAddress(
              clique_key, params.buffer_allocations->GetDeviceAddress(
                              buffers_[0].destination_buffer.slice)));
    }
  }

  return absl::OkStatus();
}

int64_t CollectiveKernelThunk::GetInputSizeBytes() const {
  return buffers_[0].element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(
             collective_config_.operand_element_type[0]);
}

absl::Status CollectiveKernelThunk::Initialize(const InitializeParams& params) {
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*is_p2p=*/false));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    if (!per_stream_state_.contains(params.executor)) {
      StreamMemory* memory_state = per_stream_memory_.at(params.executor).get();
      // Step1: We needs 1 atomic flag per block per device on each device.
      // One-shot kernel expects that the signal flags buffer is zeroed out.
      // Initial state of device memory is undefined, so we need to zero out
      // the buffer. The kernel will take care of leaving the buffer in
      // correct state after use, so we don't need to zero out after
      // initialization.
      TF_RETURN_IF_ERROR(params.executor->SynchronousMemZero(
          memory_state->signal_buffers_handle.memory_ptr(),
          memory_state->signal_buffers_handle.memory().size()));
      // Create a kernel for execution.
      std::unique_ptr<se::Kernel> kernel = nullptr;
      if (!kernel_name_.empty()) {
        if (!params.src.binary.empty()) {
          TF_ASSIGN_OR_RETURN(
              kernel,
              CreateKernel(kernel_name_, kAllReduceArgsCount, params.src.binary,
                           params.executor, shmem_bytes_));

        } else {
          TF_ASSIGN_OR_RETURN(
              kernel,
              CreateKernel(kernel_name_, kAllReduceArgsCount, params.src.text,
                           params.executor, shmem_bytes_));
        }
      }
      // Step2: Emplace into the stream state.
      per_stream_state_.emplace(
          params.executor,
          std::make_unique<StreamState>(params.executor->device_ordinal(),
                                        rank.value(), std::move(kernel)));

      state = per_stream_state_.at(params.executor).get();

      // NB: This is a double buffer allocation. So size of a single buffer is
      // half of the total allocation.
      for (int i = 0; i < kNumBuffers; ++i) {
        state->remote_buffer_ptrs[i] =
            memory_state->local_buffers_handle.address().GetByteSlice(
                /*offset_bytes=*/i * memory_state->local_buffer_size_bytes,
                /*size_bytes=*/memory_state->local_buffer_size_bytes);

        state->signal_buffer_ptrs[i] =
            memory_state->signal_buffers_handle.address().GetByteSlice(
                /*offset_bytes=*/i * memory_state->signal_buffer_size_bytes,
                /*size_bytes=*/memory_state->signal_buffer_size_bytes);
      }
    }
  }

  StreamMemory* memory_state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    memory_state = per_stream_memory_.at(params.executor).get();
  }

  if (state != nullptr) {
    CollectiveKernelMetadata metadata;
    metadata.rank = state->rank.value();

    std::vector<void*> multimem_addresses;
    std::vector<void*> param_to_peers_ptrs;

    // Resolve multimem addresses that we requested earlier.
    if (memory_state->strategy == AllReduceStrategy::kMultimem) {
      auto src_addr = params.buffer_allocations->GetDeviceAddress(
          buffers_[0].source_buffer.slice);
      auto dst_addr = params.buffer_allocations->GetDeviceAddress(
          buffers_[0].destination_buffer.slice);

      // Multimem uses original buffers assigned by the buffer assigner for
      // source and destination buffers. Also the kernel needs to run
      // multimem operation on the output buffer so we need to add it to the
      // parameters list.
      std::vector<se::DeviceAddressBase> parameters{
          src_addr, memory_state->signal_buffers_handle.address(), dst_addr};

      const size_t param_to_peers_ptrs_size_bytes =
          parameters.size() * clique_key.num_devices() * sizeof(uint64_t);

      // Kernel has additionaly an output buffer as a parameter.
      multimem_addresses.resize(kNumParameters + 1, nullptr);
      const size_t multimem_addresses_size_bytes =
          multimem_addresses.size() * sizeof(void*);
      TF_ASSIGN_OR_RETURN(
          param_to_peers_ptrs,
          CollectParamToPeers(clique_key, state->rank, params.stream,
                              std::move(parameters)));
      state->metadata = params.executor->Allocate(
          sizeof(CollectiveKernelMetadata) + param_to_peers_ptrs_size_bytes +
              multimem_addresses_size_bytes,
          0);

      auto [src_mmem, src_mmem_offset] =
          params.collective_memory->FindMultimemAddress(clique_key, src_addr);
      auto [dst_mmem, dst_mmem_offset] =
          params.collective_memory->FindMultimemAddress(clique_key, dst_addr);
      TF_RET_CHECK(src_mmem)
          << "Multimem addresses for source buffer not found";
      TF_RET_CHECK(dst_mmem)
          << "Multimem addresses for destination buffer not found";

      multimem_addresses[0] =
          tsl::safe_reinterpret_cast<char*>(src_mmem) + src_mmem_offset;
      // Kernel doesn't use multimem operations for signal buffers.
      multimem_addresses[2] =
          tsl::safe_reinterpret_cast<char*>(dst_mmem) + dst_mmem_offset;

      XLA_VLOG_DEVICE(3, params.executor->device_ordinal())
          << "Constructed device state {"
          << " metadata rank: " << metadata.rank << ", param_to_peers: ("
          << absl::StrJoin(param_to_peers_ptrs, ", ", PtrFormatter{})
          << "), multimem_addresses: ("
          << absl::StrJoin(multimem_addresses, ", ", PtrFormatter{}) << ")}";
    } else {
      std::vector<se::DeviceAddressBase> parameters{
          memory_state->local_buffers_handle.address(),
          memory_state->signal_buffers_handle.address()};

      const size_t param_to_peers_ptrs_size_bytes =
          parameters.size() * clique_key.num_devices() * sizeof(uint64_t);
      state->metadata = params.executor->Allocate(
          sizeof(CollectiveKernelMetadata) + param_to_peers_ptrs_size_bytes, 0);

      TF_ASSIGN_OR_RETURN(
          param_to_peers_ptrs,
          CollectParamToPeers(clique_key, state->rank, params.stream,
                              std::move(parameters)));
    }

    TF_RETURN_IF_ERROR(CopyCollectiveMetadataToDevice(
        params.stream, metadata, param_to_peers_ptrs, multimem_addresses,
        state->metadata));
    return absl::OkStatus();
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
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_,
                                /*is_p2p=*/false));
  const int32_t num_devices = clique_key.num_devices();

  // TODO(b/407736956): Support variadic all-reduce.
  if (collective_config_.operand_element_type.size() != 1) {
    return absl::UnimplementedError(
        "Variadic arguments are not implemented for collective kernels.");
  }
  const CollectiveThunk::Buffer& buffer = buffers_[0];
  const PrimitiveType element_type = collective_config_.operand_element_type[0];
  se::DeviceAddressBase source_buffer =
      params.buffer_allocations->GetDeviceAddress(buffer.source_buffer.slice);
  se::DeviceAddressBase destination_buffer =
      params.buffer_allocations->GetDeviceAddress(
          buffer.destination_buffer.slice);

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
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing one-shot all-reduce for clique " << clique_key.ToString();

  se::DeviceAddressBase input_buffer_ptr =
      state->remote_buffer_ptrs[buffer_index];
  se::DeviceAddressBase signal_buffer_ptr =
      state->signal_buffer_ptrs[buffer_index];
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "input_buffer_ptr: " << input_buffer_ptr.opaque()
      << " signal_buffer_ptr: " << signal_buffer_ptr.opaque();
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "launch dimensions: " << launch_dimensions.num_blocks() << "x"
      << launch_dimensions.num_threads_per_block()
      << "(block x threadsPerBlock)";

  if (state->kernel != nullptr) {
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase remote_buffers,
                        GetParameterDeviceMemoryBase(
                            state->metadata, /*num_parameters=*/kNumParameters,
                            /*num_devices=*/num_devices,
                            /*parameter_index=*/0));
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase signal_buffers,
                        GetParameterDeviceMemoryBase(
                            state->metadata, /*num_parameters=*/kNumParameters,
                            /*num_devices=*/num_devices,
                            /*parameter_index=*/1));
    std::array<se::KernelArg, kAllReduceArgsCount> kernel_args = {
        source_buffer,
        destination_buffer,
        static_cast<int32_t>(state->rank.value()),
        /* signal_value= */ state->invocation_count,
        signal_buffers,
        remote_buffers};
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

/* static */ absl::StatusOr<se::DeviceAddressBase>
CollectiveKernelThunk::GetParameterDeviceMemoryBase(
    const se::DeviceAddressBase metadata, const int64_t num_parameters,
    const int64_t num_devices, const int64_t parameter_index) {
  TF_RET_CHECK(parameter_index >= 0 && parameter_index < num_parameters)
      << "Parameter index " << parameter_index << " is out of bounds [0, "
      << num_parameters << ")";
  // The pointer table is a flattened array laid out in parameter major order.
  // P0R0 P0R1 ... P0Rn P1R0
  // P1R1 ... P1Rn ... PnRn
  // Where Pn is the parameter index and Rn is the rank.
  se::DeviceAddressBase ptr_table_base = metadata.GetByteSlice(
      sizeof(CollectiveKernelMetadata),
      /*size_bytes=*/num_parameters * num_devices * sizeof(void*));
  return ptr_table_base.GetByteSlice(
      (parameter_index * num_devices) * sizeof(void*),
      /*size_bytes=*/num_devices * sizeof(void*));
}
}  // namespace xla::gpu
