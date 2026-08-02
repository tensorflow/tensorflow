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

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.pb.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Helper for allocating memory on the device.
absl::StatusOr<se::DeviceAddressHandle> AllocateMemory(
    se::StreamExecutor* executor, int64_t size,
    absl::string_view debug_buffer_name) {
  se::DeviceAddressHandle local_buffer_alloc(
      executor,
      executor->Allocate(size, static_cast<int64_t>(
                                   stream_executor::MemorySpace::kCollective)));
  if (local_buffer_alloc.address().is_null()) {
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
      multimem_addresses_size > 0
          ? reinterpret_cast<void**>(multimem_addresses_buffer.opaque())
          : nullptr;
  RETURN_IF_ERROR(stream->Memcpy(&destination, &metadata,
                                 sizeof(CollectiveKernelMetadata)));
  RETURN_IF_ERROR(stream->Memcpy(&param_to_peers_ptrs_buffer,
                                 param_to_peers_ptrs.data(),
                                 param_to_peers_ptrs_size));
  RETURN_IF_ERROR(stream->Memcpy(&multimem_addresses_buffer,
                                 multimem_addresses.data(),
                                 multimem_addresses_size));
  return absl::OkStatus();
}

absl::StatusOr<se::DeviceAddressBase> GetParameterDeviceMemoryBase(
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

absl::StatusOr<std::vector<se::KernelArg>> BuildKernelArguments(
    const CollectiveKernelSpec& kernel_spec,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    const Thunk::ExecuteParams& params, RankId rank, uint32_t invocation_count,
    const se::DeviceAddressBase metadata, const GpuCliqueKey& clique_key,
    int32_t num_parameters) {
  std::vector<se::KernelArg> kernel_args;
  kernel_args.reserve(kernel_spec.argument_descriptors.size());
  auto get_buffer_index = [](std::optional<int32_t> index,
                             int32_t num_buffers) -> absl::StatusOr<int32_t> {
    TF_RET_CHECK(index.has_value() && *index >= 0 && *index < num_buffers)
        << "Invalid buffer index: " << index.value_or(-999);
    return *index;
  };
  const int32_t param_index_offset = [](const CollectiveKernelSpec& spec) {
    // In the parameter metadata table (`state->metadata`), peer addresses
    // for multimem-enabled buffers are stored first (1 slot per multimem
    // operand, 1 slot per multimem result), followed by scratch buffer
    // allocations.
    static constexpr auto count_multimem_buffers =
        [](const IoBufferSpec& spec) -> bool { return spec.requires_multimem; };
    return absl::c_count_if(spec.input_buffer_specs, count_multimem_buffers) +
           absl::c_count_if(spec.output_buffer_specs, count_multimem_buffers);
  }(kernel_spec);
  for (const KernelArgDescriptor& desc : kernel_spec.argument_descriptors) {
    switch (desc.type) {
      case KernelArgType::kInputBuffer: {
        ASSIGN_OR_RETURN(const int32_t buffer_index,
                         get_buffer_index(desc.index, buffers.size()));
        kernel_args.push_back(params.buffer_allocations->GetDeviceAddress(
            buffers[buffer_index].source_buffer.slice));
        break;
      }
      case KernelArgType::kOutputBuffer: {
        ASSIGN_OR_RETURN(const int32_t buffer_index,
                         get_buffer_index(desc.index, buffers.size()));
        kernel_args.push_back(params.buffer_allocations->GetDeviceAddress(
            buffers[buffer_index].destination_buffer.slice));
        break;
      }
      case KernelArgType::kRuntimeRank:
        kernel_args.push_back(static_cast<int32_t>(rank.value()));
        break;
      case KernelArgType::kInvocationCount:
        kernel_args.push_back(invocation_count);
        break;
      case KernelArgType::kScratchBuffer: {
        ASSIGN_OR_RETURN(
            const int32_t buffer_index,
            get_buffer_index(desc.index, kernel_spec.scratch_buffers.size()));
        ASSIGN_OR_RETURN(se::DeviceAddressBase peer_buf,
                         GetParameterDeviceMemoryBase(
                             metadata, num_parameters, clique_key.num_devices(),
                             buffer_index + param_index_offset));
        kernel_args.push_back(peer_buf);
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported kernel argument type: ", desc.type));
    }
  }
  return kernel_args;
}

bool RequiresMultimem(CollectiveKernelSpec kernel_spec) {
  for (const auto& op : kernel_spec.input_buffer_specs) {
    if (op.requires_multimem) {
      return true;
    }
  }
  for (const auto& res : kernel_spec.output_buffer_specs) {
    if (res.requires_multimem) {
      return true;
    }
  }
  for (const auto& scratch : kernel_spec.scratch_buffers) {
    if (scratch.requires_multimem) {
      return true;
    }
  }
  return false;
}
}  // namespace

absl::Status CollectiveKernelThunk::IsSupported(
    const GpuCliqueKey& clique_key, se::StreamExecutor& executor,
    const CollectiveParams& collective_params) const {
  // For backward compatibility in proto we drop support for collective
  // kernels if the kernel name is empty.
  if (kernel_name_.empty()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Empty kernel name ('%s')", kernel_name_));
  }
  // Check if peer access is supported for all devices in the clique.
  for (const GlobalDeviceId& device : clique_key.devices()) {
    ASSIGN_OR_RETURN(const int peer_device_id,
                     GetLocalDeviceId(device, collective_params));
    if (!executor.CanEnablePeerAccessTo(peer_device_id)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Peer access is not supported from device %d to device %d",
          executor.device_ordinal(), peer_device_id));
    }
  }
  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn);

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));

  RETURN_IF_ERROR(
      IsSupported(clique_key, *params.executor, *params.collective_params));

  // Validate that the kernel spec is compatible with the thunk buffers.
  TF_RET_CHECK(kernel_spec_.input_buffer_specs.size() == buffers_.size())
      << "Kernel spec input_buffer_specs size ("
      << kernel_spec_.input_buffer_specs.size()
      << ") must equal thunk buffers size (" << buffers_.size() << ")";
  TF_RET_CHECK(kernel_spec_.output_buffer_specs.size() == buffers_.size())
      << "Kernel spec output_buffer_specs size ("
      << kernel_spec_.output_buffer_specs.size()
      << ") must equal thunk buffers size (" << buffers_.size() << ")";
  for (const KernelArgDescriptor& desc : kernel_spec_.argument_descriptors) {
    switch (desc.type) {
      case KernelArgType::kInputBuffer:
        TF_RET_CHECK(desc.index.has_value() &&
                     desc.index.value() <
                         kernel_spec_.input_buffer_specs.size() &&
                     desc.index.value() < buffers_.size())
            << "Invalid input buffer argument index: "
            << desc.index.value_or(-999);
        break;
      case KernelArgType::kOutputBuffer:
        TF_RET_CHECK(desc.index.has_value() &&
                     desc.index.value() <
                         kernel_spec_.output_buffer_specs.size() &&
                     desc.index.value() < buffers_.size())
            << "Invalid output buffer argument index: "
            << desc.index.value_or(-999);
        break;
      case KernelArgType::kScratchBuffer:
        TF_RET_CHECK(desc.index.has_value() &&
                     desc.index.value() < kernel_spec_.scratch_buffers.size())
            << "Invalid scratch buffer argument index: "
            << desc.index.value_or(-999);
        break;
      default:
        break;
    }
  }

  ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetParticipatingDevicesGroups(*params.collective_params->device_assn,
                                    collective_config_.replica_groups,
                                    collective_config_.group_mode));

  // Sort device groups: RequestClique expects pre-sorted groups.
  absl::c_for_each(device_groups, [](auto& group) { absl::c_sort(group); });
  absl::c_sort(device_groups);

  RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
      clique_key, device_groups));

  absl::MutexLock lock(mutex_);
  if (!per_stream_memory_.contains(params.executor)) {
    std::vector<se::DeviceAddressHandle> scratch_allocations;
    scratch_allocations.reserve(kernel_spec_.scratch_buffers.size());
    for (int32_t i = 0; i < kernel_spec_.scratch_buffers.size(); ++i) {
      const ScratchBufferSpec& buf_spec = kernel_spec_.scratch_buffers[i];
      const int64_t total_bytes = xla::RoundUpTo<uint64_t>(
          buf_spec.size_bytes *
              (buf_spec.should_double_buffer ? kNumBuffers : 1),
          kXlaAllocatedBufferAlignBytes);
      ASSIGN_OR_RETURN(se::DeviceAddressHandle alloc_handle,
                       AllocateMemory(params.executor, total_bytes,
                                      absl::StrCat("Scratch ", i)));
      scratch_allocations.push_back(std::move(alloc_handle));
    }
    per_stream_memory_.emplace(
        params.executor, std::make_unique<StreamMemory>(
                             StreamMemory{std::move(scratch_allocations)}));

    // If we decided to run kernel using multimem strategy we request symmetric
    // memory for buffers that explicitly requested it.
    for (size_t i = 0; i < kernel_spec_.input_buffer_specs.size(); ++i) {
      if (kernel_spec_.input_buffer_specs[i].symmetric_memory_type ==
          SymmetricMemoryType::kLoadStoreAccessible) {
        RETURN_IF_ERROR(
            params.collective_memory_requests->RequestSymmetricAllocation(
                clique_key, buffers_[i].source_buffer.slice.index()));
      }
    }
    for (size_t i = 0; i < kernel_spec_.output_buffer_specs.size(); ++i) {
      if (kernel_spec_.output_buffer_specs[i].symmetric_memory_type ==
          SymmetricMemoryType::kLoadStoreAccessible) {
        RETURN_IF_ERROR(
            params.collective_memory_requests->RequestSymmetricAllocation(
                clique_key, buffers_[i].destination_buffer.slice.index()));
      }
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
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
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
      // Step1: Zero out scratch buffers if needed.
      for (size_t i = 0; i < kernel_spec_.scratch_buffers.size(); ++i) {
        if (kernel_spec_.scratch_buffers[i].should_memzero) {
          RETURN_IF_ERROR(params.stream->MemZero(
              memory_state->scratch_allocations[i].address_ptr(),
              memory_state->scratch_allocations[i].address().size()));
        }
      }
      RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
      TF_RET_CHECK(!kernel_name_.empty())
          << "Kernel name must be set for collective kernel thunk.";
      // Create kernel for execution.
      std::unique_ptr<se::Kernel> kernel = nullptr;
      const int32_t num_args = kernel_spec_.argument_descriptors.size();
      if (cubin_.has_value()) {
        ASSIGN_OR_RETURN(kernel, CreateKernel(kernel_name_, num_args, *cubin_,
                                              params.executor, shmem_bytes_));
      } else if (!params.src.binary.empty()) {
        ASSIGN_OR_RETURN(kernel,
                         CreateKernel(kernel_name_, num_args, params.src.binary,
                                      params.executor, shmem_bytes_));
      } else {  // Use PTX.
        ASSIGN_OR_RETURN(kernel,
                         CreateKernel(kernel_name_, num_args, params.src.text,
                                      params.executor, shmem_bytes_));
      }
      kernel->set_use_pdl(use_pdl_);
      // Step2: Emplace into the stream state.
      per_stream_state_.emplace(
          params.executor,
          std::make_unique<StreamState>(params.executor->device_ordinal(),
                                        rank.value(), std::move(kernel)));
      state = per_stream_state_.at(params.executor).get();
    }
  }

  StreamMemory* memory_state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    memory_state = per_stream_memory_.at(params.executor).get();
  }

  if (state != nullptr) {
    std::vector<se::DeviceAddressBase> parameters;
    parameters.reserve(kernel_spec_.argument_descriptors.size());
    for (size_t i = 0; i < kernel_spec_.input_buffer_specs.size(); ++i) {
      if (kernel_spec_.input_buffer_specs[i].requires_multimem) {
        parameters.push_back(params.buffer_allocations->GetDeviceAddress(
            buffers_[i].source_buffer.slice));
      }
    }
    for (size_t i = 0; i < kernel_spec_.output_buffer_specs.size(); ++i) {
      if (kernel_spec_.output_buffer_specs[i].requires_multimem) {
        parameters.push_back(params.buffer_allocations->GetDeviceAddress(
            buffers_[i].destination_buffer.slice));
      }
    }
    for (const auto& alloc : memory_state->scratch_allocations) {
      parameters.push_back(alloc.address());
    }

    const size_t num_parameters = parameters.size();
    const size_t param_to_peers_ptrs_size_bytes =
        num_parameters * clique_key.num_devices() * sizeof(uint64_t);
    std::vector<void*> multimem_addresses;
    if (RequiresMultimem(kernel_spec_) && params.collective_memory != nullptr) {
      multimem_addresses.resize(num_parameters, nullptr);
      for (size_t i = 0; i < num_parameters; ++i) {
        auto [mmem, offset] = params.collective_memory->FindSymmetricMemory(
            clique_key, parameters[i]);
        if (mmem != nullptr) {
          ASSIGN_OR_RETURN(se::DeviceAddressBase mmem_addr,
                           mmem->multimem_addr());
          multimem_addresses[i] =
              tsl::safe_reinterpret_cast<char*>(mmem_addr.opaque()) + offset;
        }
      }
    }
    ASSIGN_OR_RETURN(std::vector<void*> param_to_peers_ptrs,
                     CollectParamToPeers(clique_key, state->rank, params.stream,
                                         std::move(parameters)));
    const size_t multimem_size_bytes =
        multimem_addresses.size() * sizeof(void*);
    state->metadata = params.executor->Allocate(
        sizeof(CollectiveKernelMetadata) + param_to_peers_ptrs_size_bytes +
            multimem_size_bytes,
        0);

    CollectiveKernelMetadata metadata;
    metadata.rank = state->rank.value();
    RETURN_IF_ERROR(CopyCollectiveMetadataToDevice(
        params.stream, metadata, param_to_peers_ptrs, multimem_addresses,
        state->metadata));
    if (VLOG_IS_ON(3)) {
      XLA_VLOG_DEVICE(3, params.executor->device_ordinal())
          << "Constructed device state {"
          << " metadata rank: " << metadata.rank << ", param_to_peers: ("
          << absl::StrJoin(param_to_peers_ptrs, ", ", PtrFormatter{})
          << "), multimem_addresses: ("
          << absl::StrJoin(multimem_addresses, ", ", PtrFormatter{}) << ")}";
    }
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  TF_RET_CHECK(stream != nullptr);
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));

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

  state->invocation_count += kernel_spec_.sync_count_increment;
  TF_RET_CHECK(state->kernel != nullptr)
      << "Kernel is not initialized for collective kernel thunk.";

  static constexpr auto has_multimem = [](const auto& buffer_spec) {
    return buffer_spec.requires_multimem;
  };
  int32_t num_parameters =
      absl::c_count_if(kernel_spec_.input_buffer_specs, has_multimem) +
      absl::c_count_if(kernel_spec_.output_buffer_specs, has_multimem) +
      kernel_spec_.scratch_buffers.size();

  ASSIGN_OR_RETURN(
      std::vector<se::KernelArg> kernel_args,
      BuildKernelArguments(kernel_spec_, buffers_, params, state->rank,
                           state->invocation_count, state->metadata, clique_key,
                           num_parameters));

  return ExecuteKernelOnStream(*state->kernel, kernel_args, launch_dimensions_,
                               /*cluster_dim=*/std::nullopt, stream);
}

Thunk::BufferUses CollectiveKernelThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(buffers_.size() * 2);
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                   buffer.source_buffer.shape));
    uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                    buffer.destination_buffer.shape));
  }
  return uses;
}

absl::StatusOr<std::unique_ptr<CollectiveKernelThunk>>
CollectiveKernelThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveKernelThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  CollectiveConfig collective_config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  LaunchDimensions launch_dimensions;
  if (!thunk_proto.has_launch_dimensions()) {
    return absl::InvalidArgumentError(
        "Launch dimensions are required for collective kernel thunk.");
  }
  ASSIGN_OR_RETURN(launch_dimensions, LaunchDimensions::FromProto(
                                          thunk_proto.launch_dimensions()));
  CollectiveKernelSpec kernel_spec;
  if (thunk_proto.has_kernel_spec()) {
    const CollectiveKernelSpecProto& proto_spec = thunk_proto.kernel_spec();
    kernel_spec.input_buffer_specs.reserve(
        proto_spec.input_buffer_specs_size());
    for (const auto& input : proto_spec.input_buffer_specs()) {
      TF_RET_CHECK(
          SymmetricMemoryTypeProto_IsValid(input.symmetric_memory_type()))
          << "Invalid symmetric_memory_type: " << input.symmetric_memory_type();
      kernel_spec.input_buffer_specs.push_back(
          {input.requires_multimem(),
           static_cast<SymmetricMemoryType>(input.symmetric_memory_type())});
    }
    kernel_spec.output_buffer_specs.reserve(
        proto_spec.output_buffer_specs_size());
    for (const auto& output : proto_spec.output_buffer_specs()) {
      TF_RET_CHECK(
          SymmetricMemoryTypeProto_IsValid(output.symmetric_memory_type()))
          << "Invalid symmetric_memory_type: "
          << output.symmetric_memory_type();
      kernel_spec.output_buffer_specs.push_back(
          {output.requires_multimem(),
           static_cast<SymmetricMemoryType>(output.symmetric_memory_type())});
    }
    kernel_spec.scratch_buffers.reserve(proto_spec.scratch_buffers_size());
    for (const auto& scratch : proto_spec.scratch_buffers()) {
      TF_RET_CHECK(
          SymmetricMemoryTypeProto_IsValid(scratch.symmetric_memory_type()))
          << "Invalid symmetric_memory_type: "
          << scratch.symmetric_memory_type();
      kernel_spec.scratch_buffers.push_back(
          {scratch.size_bytes(), scratch.requires_multimem(),
           static_cast<SymmetricMemoryType>(scratch.symmetric_memory_type()),
           scratch.should_memzero(), scratch.should_double_buffer()});
    }
    kernel_spec.argument_descriptors.reserve(
        proto_spec.argument_descriptors_size());
    for (const auto& arg : proto_spec.argument_descriptors()) {
      TF_RET_CHECK(KernelArgTypeProto_IsValid(arg.type()))
          << "Invalid argument type: " << arg.type();
      KernelArgDescriptor arg_desc;
      arg_desc.type = static_cast<KernelArgType>(arg.type());
      if (arg.has_index()) {
        TF_RET_CHECK(arg.index() >= 0)
            << "Invalid argument index: " << arg.index();
        arg_desc.index = arg.index();
      }
      kernel_spec.argument_descriptors.push_back(std::move(arg_desc));
    }
    kernel_spec.sync_count_increment = proto_spec.invocation_count_increment();
  } else {
    // Backward-compatibility fallback for legacy AOT-compiled kernels without
    // an explicit CollectiveKernelSpec.
    // Can be removed in February 2027 (6months backward compatibility window).
    kernel_spec.input_buffer_specs.push_back(
        {/*requires_multimem=*/false, SymmetricMemoryType::kNone});
    kernel_spec.output_buffer_specs.push_back(
        {/*requires_multimem=*/false, SymmetricMemoryType::kNone});
    TF_RET_CHECK(thunk_proto.buffers_size() > 0)
        << "At least one buffer is required for collective kernel thunk.";
    const int64_t input_size_bytes =
        thunk_proto.buffers(0).source_buffer().slice().size();
    TF_RET_CHECK(!collective_config.replica_groups.empty())
        << "At least one replica group is required for collective kernel "
           "thunk.";
    // NB: replica_ids_size() is the same for all replica groups (HLO invariant)
    const int32_t group_size =
        collective_config.replica_groups[0].replica_ids_size();
    const int64_t num_signal_flags =
        group_size * launch_dimensions.num_blocks();
    const int64_t signal_size = xla::RoundUpTo<uint64_t>(
        num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
    const int64_t remote_size = xla::RoundUpTo<uint64_t>(
        input_size_bytes, kXlaAllocatedBufferAlignBytes);
    kernel_spec.scratch_buffers = {
        // Signal buffer.
        {/*size_bytes=*/signal_size,
         /*requires_multimem=*/false,
         /*symmetric_memory_type=*/SymmetricMemoryType::kXlaRendezvous,
         /*should_memzero=*/true,
         /*should_double_buffer=*/true},
        // Data staging buffer.
        {/*size_bytes=*/remote_size,
         /*requires_multimem=*/false,
         /*symmetric_memory_type=*/SymmetricMemoryType::kXlaRendezvous,
         /*should_memzero=*/false,
         /*should_double_buffer=*/true}};
    kernel_spec.argument_descriptors = {
        {KernelArgType::kInputBuffer, /*index=*/0},
        {KernelArgType::kOutputBuffer, /*index=*/0},
        {KernelArgType::kRuntimeRank},
        {KernelArgType::kInvocationCount},
        {KernelArgType::kScratchBuffer, /*index=*/0},
        {KernelArgType::kScratchBuffer, /*index=*/1}};
    kernel_spec.sync_count_increment =
        1 + static_cast<uint32_t>(GetAllReduceStrategy(
                input_size_bytes, /*is_multimem_enabled=*/false));
  }
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(std::move(buffer));
  }

  std::optional<std::vector<uint8_t>> cubin = std::nullopt;
  if (thunk_proto.has_cubin()) {
    cubin = std::vector<uint8_t>{thunk_proto.cubin().begin(),
                                 thunk_proto.cubin().end()};
  }

  return std::make_unique<CollectiveKernelThunk>(
      thunk_info, collective_config, std::move(kernel_spec),
      thunk_proto.is_async(), std::move(buffers),
      thunk_proto.collective_kernel_enabled(), thunk_proto.kernel_name(),
      launch_dimensions, thunk_proto.shmem_bytes(), std::move(cubin),
      thunk_proto.use_pdl());
}

absl::StatusOr<ThunkProto> CollectiveKernelThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  CollectiveKernelThunkProto* thunk_proto =
      proto.mutable_collective_kernel_thunk();

  *thunk_proto->mutable_collective_config() = collective_config_.ToProto();

  auto* proto_spec = thunk_proto->mutable_kernel_spec();
  proto_spec->mutable_input_buffer_specs()->Reserve(
      kernel_spec_.input_buffer_specs.size());
  for (const auto& op : kernel_spec_.input_buffer_specs) {
    auto* io_proto = proto_spec->add_input_buffer_specs();
    io_proto->set_requires_multimem(op.requires_multimem);
    io_proto->set_symmetric_memory_type(
        static_cast<SymmetricMemoryTypeProto>(op.symmetric_memory_type));
  }
  proto_spec->mutable_output_buffer_specs()->Reserve(
      kernel_spec_.output_buffer_specs.size());
  for (const auto& res : kernel_spec_.output_buffer_specs) {
    auto* io_proto = proto_spec->add_output_buffer_specs();
    io_proto->set_requires_multimem(res.requires_multimem);
    io_proto->set_symmetric_memory_type(
        static_cast<SymmetricMemoryTypeProto>(res.symmetric_memory_type));
  }
  proto_spec->mutable_scratch_buffers()->Reserve(
      kernel_spec_.scratch_buffers.size());
  for (const auto& scratch : kernel_spec_.scratch_buffers) {
    auto* scratch_proto = proto_spec->add_scratch_buffers();
    scratch_proto->set_size_bytes(scratch.size_bytes);
    scratch_proto->set_requires_multimem(scratch.requires_multimem);
    scratch_proto->set_symmetric_memory_type(
        static_cast<SymmetricMemoryTypeProto>(scratch.symmetric_memory_type));
    scratch_proto->set_should_memzero(scratch.should_memzero);
    scratch_proto->set_should_double_buffer(scratch.should_double_buffer);
  }
  proto_spec->mutable_argument_descriptors()->Reserve(
      kernel_spec_.argument_descriptors.size());
  for (const auto& arg : kernel_spec_.argument_descriptors) {
    auto* arg_proto = proto_spec->add_argument_descriptors();
    arg_proto->set_type(static_cast<KernelArgTypeProto>(arg.type));
    if (arg.index.has_value()) {
      arg_proto->set_index(arg.index.value());
    }
  }
  proto_spec->set_invocation_count_increment(kernel_spec_.sync_count_increment);

  thunk_proto->set_is_async(is_async_);
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  thunk_proto->set_collective_kernel_enabled(collective_kernel_enabled_);
  thunk_proto->set_kernel_name(kernel_name_);
  *thunk_proto->mutable_launch_dimensions() = launch_dimensions_.ToProto();

  thunk_proto->set_shmem_bytes(shmem_bytes_);

  if (cubin_.has_value()) {
    thunk_proto->set_cubin(reinterpret_cast<const char*>(cubin_->data()),
                           cubin_->size());
  }

  thunk_proto->set_use_pdl(use_pdl_);

  return proto;
}
}  // namespace xla::gpu
