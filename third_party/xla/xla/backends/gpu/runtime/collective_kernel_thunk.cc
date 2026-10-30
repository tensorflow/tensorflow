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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.pb.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/collective_types.pb.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/kernel_thunk_utils.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"

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

// We use a double buffering strategy for the scratch buffers.
constexpr int64_t kNumBuffers = 2;

struct PtrFormatter {
  void operator()(std::string* out, const void* ptr) const {
    absl::StrAppend(out, absl::StrFormat("%p", ptr));
  }
};

// Device layout of the per-stream collective metadata block:
//   [CollectiveKernelMetadata]
//   [peer pointer table: num_parameters x num_devices void*]
//   [multimem addresses: num_parameters void*]   (only if requires_multimem)
struct CollectiveMetadataLayout {
  CollectiveMetadataLayout(int64_t num_devices, int64_t num_parameters,
                           bool requires_multimem)
      : peers_bytes(num_parameters * num_devices * sizeof(void*)),
        multimem_bytes(requires_multimem ? num_parameters * sizeof(void*) : 0) {
  }

  size_t total_bytes() const {
    return sizeof(CollectiveKernelMetadata) + peers_bytes + multimem_bytes;
  }
  se::DeviceAddressBase PeersTable(se::DeviceAddressBase block) const {
    return block.GetByteSlice(sizeof(CollectiveKernelMetadata), peers_bytes);
  }
  se::DeviceAddressBase MultimemTable(se::DeviceAddressBase block) const {
    return block.GetByteSlice(sizeof(CollectiveKernelMetadata) + peers_bytes,
                              multimem_bytes);
  }

  size_t peers_bytes;
  size_t multimem_bytes;
};

absl::Status CopyParamToPeersPtrsToDevice(
    se::Stream* stream, const CollectiveMetadataLayout& layout,
    se::DeviceAddressBase metadata,
    absl::Span<const void* const> param_to_peers_ptrs) {
  se::DeviceAddressBase dst = layout.PeersTable(metadata);
  return stream->Memcpy(&dst, param_to_peers_ptrs.data(),
                        param_to_peers_ptrs.size() * sizeof(void*));
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
  // The peer table offset does not depend on the multimem table.
  return CollectiveMetadataLayout(num_devices, num_parameters,
                                  /*requires_multimem=*/false)
      .PeersTable(metadata)
      .GetByteSlice((parameter_index * num_devices) * sizeof(void*),
                    /*size_bytes=*/num_devices * sizeof(void*));
}

// One kernel parameter that has a row in the device peer pointer table.
// Rows are ordered: multimem inputs, multimem outputs, then all scratch
// buffers.
struct SymmetricParameterSpec {
  SymmetricMemoryType symmetric_memory_type;
  bool requires_multimem;
  // Buffer slice for IO parameters, nullopt for scratch parameters.
  std::optional<BufferAllocation::Slice> io_slice;
  // Index into `kernel_spec.scratch_buffers` for scratch parameters.
  int64_t scratch_index = -1;
};

// Peer pointer table layout and scratch buffer layout derived from the kernel
// spec.
struct KernelSpecLayout {
  static constexpr size_t kInlineParams = 8;
  template <typename T>
  using ParamVector = absl::InlinedVector<T, kInlineParams>;

  ParamVector<SymmetricParameterSpec> symmetric_params;
  // Number of IO rows in `symmetric_params`; scratch rows follow.
  int64_t num_symmetric_io_params = 0;
  // Peer pointer table row for every scratch buffer.
  ParamVector<int32_t> scratch_param_indices;
  // Byte offset and size of every scratch buffer in the combined scratch
  // allocation.
  ParamVector<int64_t> scratch_offsets;
  ParamVector<int64_t> scratch_sizes;
  int64_t total_scratch_bytes = 0;

  bool AnyScratch(SymmetricMemoryType type) const {
    return absl::c_any_of(symmetric_params,
                          [type](const SymmetricParameterSpec& param) {
                            return param.scratch_index >= 0 &&
                                   param.symmetric_memory_type == type;
                          });
  }
  bool AnyMultimem() const {
    return absl::c_any_of(symmetric_params,
                          [](const SymmetricParameterSpec& param) {
                            return param.requires_multimem;
                          });
  }
};

KernelSpecLayout BuildKernelSpecLayout(
    const CollectiveKernelSpec& kernel_spec,
    absl::Span<const CollectiveThunk::Buffer> buffers) {
  KernelSpecLayout layout;

  auto add_param = [&](SymmetricMemoryType type, bool requires_multimem,
                       std::optional<BufferAllocation::Slice> io_slice,
                       int64_t scratch_index) -> int32_t {
    layout.symmetric_params.push_back(
        {type, requires_multimem, io_slice, scratch_index});
    return layout.symmetric_params.size() - 1;
  };

  auto add_io_params = [&](absl::Span<const IoBufferSpec> specs,
                           bool is_input) {
    for (auto [buffer, spec] : llvm::zip(buffers, specs)) {
      if (!spec.requires_multimem) {
        continue;
      }
      const BufferAllocation::Slice& slice =
          is_input ? buffer.source_buffer.slice
                   : buffer.destination_buffer.slice;
      add_param(spec.symmetric_memory_type, spec.requires_multimem, slice,
                /*scratch_index=*/-1);
    }
  };
  add_io_params(kernel_spec.codegen_config.input_buffer_specs,
                /*is_input=*/true);
  add_io_params(kernel_spec.codegen_config.output_buffer_specs,
                /*is_input=*/false);
  layout.num_symmetric_io_params = layout.symmetric_params.size();

  const size_t num_scratch = kernel_spec.scratch_buffers.size();
  for (size_t i = 0; i < num_scratch; ++i) {
    const ScratchBufferSpec& spec = kernel_spec.scratch_buffers[i];
    const int64_t slice_bytes = xla::RoundUpTo<uint64_t>(
        spec.size_bytes * (spec.should_double_buffer ? kNumBuffers : 1),
        kXlaAllocatedBufferAlignBytes);
    layout.scratch_offsets.push_back(layout.total_scratch_bytes);
    layout.scratch_sizes.push_back(slice_bytes);
    layout.total_scratch_bytes += slice_bytes;
    layout.scratch_param_indices.push_back(
        add_param(spec.symmetric_memory_type, spec.requires_multimem,
                  /*io_slice=*/std::nullopt, /*scratch_index=*/i));
  }
  return layout;
}

// Current device addresses of all table parameters, in row order.
KernelSpecLayout::ParamVector<se::DeviceAddressBase>
GetSymmetricParameterAddresses(
    const KernelSpecLayout& layout, const BufferAllocations& buffer_allocations,
    absl::Span<const se::DeviceAddressBase> scratch_slices) {
  KernelSpecLayout::ParamVector<se::DeviceAddressBase> addresses;
  for (const SymmetricParameterSpec& param : layout.symmetric_params) {
    if (param.io_slice.has_value()) {
      addresses.push_back(buffer_allocations.GetDeviceAddress(*param.io_slice));
    } else {
      addresses.push_back(scratch_slices[param.scratch_index]);
    }
  }
  return addresses;
}

// Exchanges the addresses of all table parameters with all ranks and returns
// the flattened peer pointer table.
absl::StatusOr<std::vector<void*>> CollectParamToPeersWithRendezvous(
    const GpuCliqueKey& clique_key, RankId rank,
    absl::Span<const se::DeviceAddressBase> addresses) {
  const int64_t num_devices = clique_key.num_devices();
  std::vector<void*> param_to_peers_ptrs(addresses.size() * num_devices);

  using DeviceParameters = absl::flat_hash_map<int, se::DeviceAddressBase>;

  DeviceParameters param_idx_to_device_address;
  for (int i = 0; i < addresses.size(); ++i) {
    param_idx_to_device_address[i] = addresses[i];
  }

  // Exchange device parameters with all ranks in the clique.
  ABSL_ASSIGN_OR_RETURN(
      auto device_parameters,
      GpuCliqueRendezvous::Join(clique_key, rank,
                                std::move(param_idx_to_device_address)));

  for (auto peer = RankId(0); peer < RankId(num_devices); ++peer) {
    ABSL_ASSIGN_OR_RETURN(const DeviceParameters& peer_parameters,
                     device_parameters->at<DeviceParameters>(peer));
    for (const auto& [param_idx, device_address] :
         tsl::SortedRange(peer_parameters)) {
      param_to_peers_ptrs[peer.value() + param_idx * num_devices] =
          device_address.opaque();
    }
  }

  return param_to_peers_ptrs;
}

// Multimem address of every table parameter that is backed by symmetric
// memory, nullptr otherwise. Empty if no parameter requires multimem.
absl::StatusOr<std::vector<void*>> GetMultimemAddresses(
    const KernelSpecLayout& layout, const GpuCliqueKey& clique_key,
    const CollectiveMemory& collective_memory,
    absl::Span<const se::DeviceAddressBase> addresses) {
  std::vector<void*> multimem_addresses;
  if (!layout.AnyMultimem()) {
    return multimem_addresses;
  }
  multimem_addresses.resize(addresses.size(), nullptr);
  for (size_t i = 0; i < addresses.size(); ++i) {
    auto [symmetric_memory, offset] =
        collective_memory.FindSymmetricMemory(clique_key, addresses[i]);
    if (symmetric_memory == nullptr) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase multimem_addr,
                     symmetric_memory->multimem_addr());
    multimem_addresses[i] =
        tsl::safe_reinterpret_cast<char*>(multimem_addr.opaque()) + offset;
  }
  return multimem_addresses;
}

absl::StatusOr<std::vector<se::KernelArg>> BuildKernelArguments(
    const CollectiveKernelSpec& kernel_spec, const KernelSpecLayout& layout,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    const BufferAllocations& buffer_allocations, se::DeviceAddressBase metadata,
    RankId rank, uint32_t invocation_count, int64_t num_devices) {
  const int32_t num_parameters = layout.symmetric_params.size();
  std::vector<se::KernelArg> kernel_args;
  kernel_args.reserve(kernel_spec.codegen_config.argument_descriptors.size());
  auto get_buffer_index = [](std::optional<int32_t> index,
                             int32_t num_buffers) -> absl::StatusOr<int32_t> {
    TF_RET_CHECK(index.has_value() && *index >= 0 && *index < num_buffers)
        << "Invalid buffer index: " << index.value_or(-999);
    return *index;
  };
  for (const KernelArgDescriptor& desc :
       kernel_spec.codegen_config.argument_descriptors) {
    switch (desc.type) {
      case KernelArgType::kInputBuffer: {
        ABSL_ASSIGN_OR_RETURN(const int32_t buffer_index,
                         get_buffer_index(desc.index, buffers.size()));
        kernel_args.push_back(buffer_allocations.GetDeviceAddress(
            buffers[buffer_index].source_buffer.slice));
        break;
      }
      case KernelArgType::kOutputBuffer: {
        ABSL_ASSIGN_OR_RETURN(const int32_t buffer_index,
                         get_buffer_index(desc.index, buffers.size()));
        kernel_args.push_back(buffer_allocations.GetDeviceAddress(
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
        ABSL_ASSIGN_OR_RETURN(
            const int32_t buffer_index,
            get_buffer_index(desc.index, kernel_spec.scratch_buffers.size()));
        ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase peer_buf,
                         GetParameterDeviceMemoryBase(
                             metadata, num_parameters, num_devices,
                             layout.scratch_param_indices[buffer_index]));
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

// Allocates the device metadata block and writes the header that points the
// kernel at the peer pointer table (and the multimem table, if any).
absl::StatusOr<se::DeviceAddressBase> AllocateCollectiveMetadata(
    se::StreamExecutor* executor, se::Stream* stream, RankId rank,
    const CollectiveMetadataLayout& layout) {
  se::DeviceAddressBase block = executor->Allocate(layout.total_bytes());

  CollectiveKernelMetadata metadata{};
  metadata.rank = rank.value();
  metadata.param_to_peers =
      reinterpret_cast<void**>(layout.PeersTable(block).opaque());
  metadata.param_to_multimem_addresses =
      layout.multimem_bytes > 0
          ? reinterpret_cast<void**>(layout.MultimemTable(block).opaque())
          : nullptr;
  ABSL_RETURN_IF_ERROR(
      stream->Memcpy(&block, &metadata, sizeof(CollectiveKernelMetadata)));
  return block;
}

// Recorded input -> symmetric scratch copy that precedes a kernel launch node.
struct RecordedCopy {
  const se::CommandBuffer::Command* command = nullptr;
  void* dst = nullptr;
  void* src = nullptr;
};

// Per command buffer state: maps the recorded kernel launch node (the command
// returned from Record) to the copy command that precedes it, so both can be
// updated in place.
struct CollectiveKernelRecordState : public CommandState {
  absl::flat_hash_map<const se::CommandBuffer::Command*, RecordedCopy>
      launch_to_copy;
};
}  // namespace

std::string CollectiveKernelThunk::ToString(int indent) const {
  return KernelThunkToString(kernel_name_, thunk_info().profile_annotation,
                             launch_dimensions_, /*cluster_dim=*/std::nullopt,
                             shmem_bytes_, use_pdl_);
}

absl::Status CollectiveKernelThunk::IsSupported(
    const GpuCliqueKey& clique_key, se::StreamExecutor& executor,
    const CollectiveParams& collective_params) const {
  // For backward compatibility in proto we drop support for collective
  // kernels if the kernel name is empty.
  if (kernel_name_.empty()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Empty kernel name ('%s')", kernel_name_));
  }
  // Check if peer access is supported for all devices in the clique managed by
  // the current process.
  for (const GlobalDeviceId& device : clique_key.devices()) {
    auto peer_device_id = GetLocalDeviceId(device, collective_params);
    if (!peer_device_id.ok()) {
      // Device is managed by a different process in multi-process setups.
      continue;
    }
    if (!executor.CanEnablePeerAccessTo(*peer_device_id)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Peer access is not supported from device %d to device %d",
          executor.device_ordinal(), *peer_device_id));
    }
  }
  return absl::OkStatus();
}

absl::Status CollectiveKernelThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn);

  ABSL_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));

  ABSL_RETURN_IF_ERROR(
      IsSupported(clique_key, *params.executor, *params.collective_params));

  // Validate that the kernel spec is compatible with the thunk buffers.
  TF_RET_CHECK(kernel_spec_.codegen_config.input_buffer_specs.size() ==
               buffers_.size())
      << "Kernel spec input_buffer_specs size ("
      << kernel_spec_.codegen_config.input_buffer_specs.size()
      << ") must equal thunk buffers size (" << buffers_.size() << ")";
  TF_RET_CHECK(kernel_spec_.codegen_config.output_buffer_specs.size() ==
               buffers_.size())
      << "Kernel spec output_buffer_specs size ("
      << kernel_spec_.codegen_config.output_buffer_specs.size()
      << ") must equal thunk buffers size (" << buffers_.size() << ")";
  for (const KernelArgDescriptor& desc :
       kernel_spec_.codegen_config.argument_descriptors) {
    switch (desc.type) {
      case KernelArgType::kInputBuffer:
        TF_RET_CHECK(
            desc.index.has_value() &&
            desc.index.value() <
                kernel_spec_.codegen_config.input_buffer_specs.size() &&
            desc.index.value() < buffers_.size())
            << "Invalid input buffer argument index: "
            << desc.index.value_or(-999);
        break;
      case KernelArgType::kOutputBuffer:
        TF_RET_CHECK(
            desc.index.has_value() &&
            desc.index.value() <
                kernel_spec_.codegen_config.output_buffer_specs.size() &&
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

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<GlobalDeviceId>> device_groups,
      GetParticipatingDevicesGroups(*params.collective_params->device_assn,
                                    collective_config_.replica_groups,
                                    collective_config_.group_mode));

  // Sort device groups: RequestClique expects pre-sorted groups.
  absl::c_for_each(device_groups, [](auto& group) { absl::c_sort(group); });
  absl::c_sort(device_groups);

  ABSL_RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
      clique_key, device_groups));

  // If we decided to run kernel using multimem or symmetric memory strategy we
  // request symmetric memory for buffers that explicitly requested it.
  auto request_io_symmetric_allocations =
      [&](absl::Span<const IoBufferSpec> specs, bool is_input) -> absl::Status {
    for (auto [buffer, spec] : llvm::zip(buffers_, specs)) {
      if (spec.symmetric_memory_type ==
          SymmetricMemoryType::kLoadStoreAccessible) {
        const BufferAllocation::Slice& slice =
            is_input ? buffer.source_buffer.slice
                     : buffer.destination_buffer.slice;
        ABSL_RETURN_IF_ERROR(
            params.collective_memory_requests->RequestSymmetricAllocation(
                clique_key, slice.index()));
      }
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(request_io_symmetric_allocations(
      kernel_spec_.codegen_config.input_buffer_specs, /*is_input=*/true));
  ABSL_RETURN_IF_ERROR(request_io_symmetric_allocations(
      kernel_spec_.codegen_config.output_buffer_specs, /*is_input=*/false));

  return absl::OkStatus();
}

int64_t CollectiveKernelThunk::GetInputSizeBytes() const {
  return buffers_[0].element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(
             collective_config_.operand_element_type[0]);
}

absl::Status CollectiveKernelThunk::Initialize(const InitializeParams& params) {
  ABSL_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  const int64_t num_devices = clique_key.num_devices();
  const KernelSpecLayout layout = BuildKernelSpecLayout(kernel_spec_, buffers_);
  const int64_t num_parameters = layout.symmetric_params.size();
  const CollectiveMetadataLayout metadata_layout(num_devices, num_parameters,
                                                 layout.AnyMultimem());

  // With load/store accessible scratch the peer pointers come from symmetric
  // memory created below (and re-created when the clique changes). Otherwise
  // they are exchanged once with a host rendezvous.
  const bool use_symmetric_memory =
      layout.AnyScratch(SymmetricMemoryType::kLoadStoreAccessible);

  auto copy_multimem_addresses =
      [&](se::DeviceAddressBase metadata,
          absl::Span<void* const> multimem_addresses) -> absl::Status {
    if (multimem_addresses.empty()) {
      return absl::OkStatus();
    }
    se::DeviceAddressBase dst = metadata_layout.MultimemTable(metadata);
    return params.stream->Memcpy(&dst, multimem_addresses.data(),
                                 multimem_addresses.size() * sizeof(void*));
  };

  ABSL_RETURN_IF_ERROR(per_device_state_.GetOrCreateAndInitialize(
      params.executor->device_ordinal(),
      [&](StreamState* state) -> absl::Status {
        se::DeviceAddressHandle scratch_allocation;
        std::vector<se::DeviceAddressBase> scratch_slices;
        scratch_slices.reserve(kernel_spec_.scratch_buffers.size());
        if (layout.total_scratch_bytes > 0) {
          ABSL_ASSIGN_OR_RETURN(
              scratch_allocation,
              AllocateMemory(params.executor, layout.total_scratch_bytes,
                             "Scratch"));
          for (size_t i = 0; i < kernel_spec_.scratch_buffers.size(); ++i) {
            scratch_slices.push_back(scratch_allocation.address().GetByteSlice(
                layout.scratch_offsets[i], layout.scratch_sizes[i]));
          }
        }
        // Step1: Zero out scratch buffers if needed.
        for (size_t i = 0; i < scratch_slices.size(); ++i) {
          if (kernel_spec_.scratch_buffers[i].should_memzero) {
            ABSL_RETURN_IF_ERROR(params.stream->MemZero(&scratch_slices[i],
                                                   scratch_slices[i].size()));
          }
        }
        ABSL_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
        TF_RET_CHECK(!kernel_name_.empty())
            << "Kernel name must be set for collective kernel thunk.";
        // Create kernel for execution.
        std::unique_ptr<se::Kernel> kernel = nullptr;
        const int32_t num_args =
            kernel_spec_.codegen_config.argument_descriptors.size();
        if (cubin_.has_value()) {
          ABSL_ASSIGN_OR_RETURN(kernel, CreateKernel(kernel_name_, num_args, *cubin_,
                                                params.executor, shmem_bytes_));
        } else if (!params.src.binary.empty()) {
          ABSL_ASSIGN_OR_RETURN(
              kernel, CreateKernel(kernel_name_, num_args, params.src.binary,
                                   params.executor, shmem_bytes_));
        } else {  // Use PTX.
          ABSL_ASSIGN_OR_RETURN(kernel,
                           CreateKernel(kernel_name_, num_args, params.src.text,
                                        params.executor, shmem_bytes_));
        }
        kernel->set_use_pdl(use_pdl_);
        // Step2: Populate the stream state.
        *state = StreamState(params.executor->device_ordinal(), rank.value(),
                             std::move(kernel), std::move(scratch_allocation),
                             std::move(scratch_slices));

        ABSL_ASSIGN_OR_RETURN(state->metadata, AllocateCollectiveMetadata(
                                              params.executor, params.stream,
                                              rank.value(), metadata_layout));

        if (use_symmetric_memory) {
          return absl::OkStatus();
        }
        TF_RET_CHECK(params.collective_memory != nullptr)
            << "Collective memory must not be null in "
               "CollectiveKernelThunk::Initialize";
        const KernelSpecLayout::ParamVector<se::DeviceAddressBase> addresses =
            GetSymmetricParameterAddresses(layout, *params.buffer_allocations,
                                           state->scratch_slices);
        ABSL_ASSIGN_OR_RETURN(
            std::vector<void*> multimem_addresses,
            GetMultimemAddresses(layout, clique_key, *params.collective_memory,
                                 addresses));
        ABSL_ASSIGN_OR_RETURN(std::vector<void*> param_to_peers_ptrs,
                         CollectParamToPeersWithRendezvous(
                             clique_key, state->rank, addresses));
        ABSL_RETURN_IF_ERROR(
            CopyParamToPeersPtrsToDevice(params.stream, metadata_layout,
                                         state->metadata, param_to_peers_ptrs));
        return copy_multimem_addresses(state->metadata, multimem_addresses);
      }));

  if (!use_symmetric_memory) {
    return absl::OkStatus();
  }

  StreamState* state =
      per_device_state_.Find(params.executor->device_ordinal());
  TF_RET_CHECK(state != nullptr);
  if (!state->scratch_symmetric_memory.Expired()) {
    return absl::OkStatus();
  }

  TF_RET_CHECK(params.collective_params != nullptr)
      << "Collective params must not be null in "
         "CollectiveKernelThunk::Initialize";
  TF_RET_CHECK(params.collective_cliques != nullptr)
      << "Collective cliques must not be null in "
         "CollectiveKernelThunk::Initialize";
  TF_RET_CHECK(params.collective_memory != nullptr)
      << "Collective memory must not be null in "
         "CollectiveKernelThunk::Initialize";

  ABSL_ASSIGN_OR_RETURN(GpuCommunicator * comm,
                   params.collective_cliques->GetComm(clique_key, *rank));
  if (!state->scratch_allocation.address().is_null()) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<SymmetricMemory> symmetric_memory,
                     comm->CreateSymmetricMemory(
                         state->scratch_allocation.address(),
                         params.collective_params->nccl_window_symmetric));
    ABSL_ASSIGN_OR_RETURN(state->scratch_symmetric_memory,
                     params.collective_cliques->Tie(
                         clique_key, std::move(symmetric_memory)));
  }

  const KernelSpecLayout::ParamVector<se::DeviceAddressBase> addresses =
      GetSymmetricParameterAddresses(layout, *params.buffer_allocations,
                                     state->scratch_slices);
  ABSL_ASSIGN_OR_RETURN(std::vector<void*> multimem_addresses,
                   GetMultimemAddresses(layout, clique_key,
                                        *params.collective_memory, addresses));

  std::vector<void*> param_to_peers_ptrs(num_parameters * num_devices, nullptr);
  if (layout.num_symmetric_io_params < num_parameters) {
    std::shared_ptr<SymmetricMemory> scratch_symmetric_memory =
        state->scratch_symmetric_memory.Lock();
    TF_RET_CHECK(scratch_symmetric_memory != nullptr)
        << "Symmetric memory for scratch allocation is no longer valid";
    for (int64_t i = layout.num_symmetric_io_params; i < num_parameters; ++i) {
      const int64_t scratch_index = layout.symmetric_params[i].scratch_index;
      for (int64_t device_rank = 0; device_rank < num_devices; ++device_rank) {
        ABSL_ASSIGN_OR_RETURN(
            se::DeviceAddressBase peer_base,
            scratch_symmetric_memory->peer_addr(RankId(device_rank)));
        param_to_peers_ptrs[i * num_devices + device_rank] =
            peer_base
                .GetByteSlice(layout.scratch_offsets[scratch_index],
                              layout.scratch_sizes[scratch_index])
                .opaque();
      }
    }
  }

  ABSL_RETURN_IF_ERROR(CopyParamToPeersPtrsToDevice(
      params.stream, metadata_layout, state->metadata, param_to_peers_ptrs));
  ABSL_RETURN_IF_ERROR(copy_multimem_addresses(state->metadata, multimem_addresses));

  if (VLOG_IS_ON(3)) {
    XLA_VLOG_DEVICE(3, params.executor->device_ordinal())
        << "Constructed device state {"
        << " metadata rank: " << *rank << ", param_to_peers: ("
        << absl::StrJoin(param_to_peers_ptrs, ", ", PtrFormatter{})
        << "), multimem_addresses: ("
        << absl::StrJoin(multimem_addresses, ", ", PtrFormatter{}) << ")}";
  }
  return absl::OkStatus();
}

absl::StatusOr<CollectiveKernelThunk::LaunchPlan>
CollectiveKernelThunk::PrepareLaunch(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  TF_RET_CHECK(stream != nullptr);
  ABSL_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, collective_config_));

  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  TF_RET_CHECK(rank.has_value())
      << "Device " << params.collective_params->global_device_id
      << "is not in the clique.";
  StreamState* state =
      per_device_state_.Find(stream->parent()->device_ordinal());
  TF_RET_CHECK(state != nullptr) << "Stream not found in per_device_state_";

  // Backward-compatibility fallback for legacy AOT-compiled kernels.
  if (!kernel_spec_.codegen_config.device_sync_count) {
    state->invocation_count += kernel_spec_.codegen_config.sync_count_increment;
  }
  TF_RET_CHECK(state->kernel != nullptr)
      << "Kernel is not initialized for collective kernel thunk.";

  LaunchPlan plan;
  plan.kernel = state->kernel.get();
  ABSL_ASSIGN_OR_RETURN(
      plan.kernel_args,
      BuildKernelArguments(
          kernel_spec_, BuildKernelSpecLayout(kernel_spec_, buffers_), buffers_,
          *params.buffer_allocations, state->metadata, state->rank,
          state->invocation_count, clique_key.num_devices()));

  // For AllGather: copy input buffer → local rank's symmetric scratch buffer
  // before kernel launch. All ranks perform this copy, then the entry barrier
  // in the kernel waits until every rank has finished its copy.
  if (kernel_spec_.codegen_config.copy_input_to_scratch) {
    // scratch_slices[1] is the symmetric remote buffer that holds this
    // rank's input slice. scratch_slices[0] is the signal buffer.
    TF_RET_CHECK(state->scratch_slices.size() >= 2)
        << "Expected at least 2 scratch slices for D2D copy.";
    plan.copy_dst = &state->scratch_slices[1];
    plan.copy_src = params.buffer_allocations->GetDeviceAddress(
        buffers_[0].source_buffer.slice);
    plan.copy_size = GetInputSizeBytes();
  }
  return plan;
}

absl::Status CollectiveKernelThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  ABSL_ASSIGN_OR_RETURN(LaunchPlan plan, PrepareLaunch(params));
  if (plan.copy_dst != nullptr) {
    ABSL_RETURN_IF_ERROR(
        params.stream->Memcpy(plan.copy_dst, plan.copy_src, plan.copy_size));
  }
  return ExecuteKernelOnStream(*plan.kernel, plan.kernel_args,
                               launch_dimensions_,
                               /*cluster_dim=*/std::nullopt, params.stream);
}

absl::StatusOr<const se::CommandBuffer::Command*> CollectiveKernelThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  // The kernel is recorded as an explicit graph kernel node. For kernels with a
  // host-side sync count PrepareLaunch bumps the per-stream invocation_count,
  // so every create/update carries the up-to-date barrier signal value in the
  // kernel node parameters. Kernels with a device-side sync count get constant
  // arguments and only need updates when buffer addresses change.
  ABSL_ASSIGN_OR_RETURN(LaunchPlan plan, PrepareLaunch(execute_params));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::KernelArgsPackedArrayBase> packed_args,
                   se::PackKernelArgs(absl::MakeConstSpan(plan.kernel_args),
                                      plan.kernel->metadata()));
  auto* state = record_params.state.GetOrCreate<CollectiveKernelRecordState>(
      this, command_buffer);

  // The input -> symmetric scratch copy is recorded as a (tiny) stream-captured
  // child graph. An explicit memcpy node targeting symmetric (collective)
  // memory crashes CUPTI during graph instantiation when profiling is enabled.
  // Its parameters only depend on buffer addresses, so it is re-captured only
  // when those change.
  auto trace_copy =
      [&]() -> absl::StatusOr<std::unique_ptr<se::CommandBuffer>> {
    return se::TraceCommandBufferFactory::Create(
        execute_params.stream->parent(),
        execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
          std::optional<tsl::profiler::ScopedAnnotation> annotation =
              std::holds_alternative<RecordUpdate>(record_action)
                  ? GetInstructionAnnotation(profile_annotation())
                  : std::nullopt;
          return stream->Memcpy(plan.copy_dst, plan.copy_src, plan.copy_size);
        });
  };

  auto create = [&](const RecordCreate& action)
      -> absl::StatusOr<const se::CommandBuffer::Command*> {
    absl::Span<const se::CommandBuffer::Command* const> deps =
        action.dependencies;
    const se::CommandBuffer::Command* copy_cmd = nullptr;
    if (plan.copy_dst != nullptr) {
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested, trace_copy());
      ABSL_ASSIGN_OR_RETURN(copy_cmd,
                       command_buffer->CreateChildCommand(*nested, deps));
      deps = absl::MakeConstSpan(&copy_cmd, 1);
    }
    ABSL_ASSIGN_OR_RETURN(
        const se::CommandBuffer::Command* launch_cmd,
        command_buffer->CreateLaunch(
            launch_dimensions_.thread_counts_per_block(),
            launch_dimensions_.block_counts(), /*cluster_dims=*/std::nullopt,
            *plan.kernel, *packed_args, deps, priority()));
    if (copy_cmd != nullptr) {
      state->launch_to_copy[launch_cmd] = RecordedCopy{
          copy_cmd, plan.copy_dst->opaque(), plan.copy_src.opaque()};
    }
    return launch_cmd;
  };
  auto update = [&](const RecordUpdate& action)
      -> absl::StatusOr<const se::CommandBuffer::Command*> {
    if (plan.copy_dst != nullptr) {
      auto it = state->launch_to_copy.find(action.command);
      TF_RET_CHECK(it != state->launch_to_copy.end())
          << "Missing recorded D2D copy command for collective kernel update";
      RecordedCopy& copy = it->second;
      if (copy.dst != plan.copy_dst->opaque() ||
          copy.src != plan.copy_src.opaque()) {
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested,
                         trace_copy());
        ABSL_RETURN_IF_ERROR(
            command_buffer->UpdateChildCommand(copy.command, *nested));
        copy.dst = plan.copy_dst->opaque();
        copy.src = plan.copy_src.opaque();
      }
    }
    ABSL_RETURN_IF_ERROR(command_buffer->UpdateLaunch(
        action.command, launch_dimensions_.thread_counts_per_block(),
        launch_dimensions_.block_counts(), /*cluster_dims=*/std::nullopt,
        *plan.kernel, *packed_args));
    return action.command;
  };
  return std::visit(absl::Overload(create, update), record_action);
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
    absl::Span<const BufferAllocation> buffer_allocations,
    int devices_in_process) {
  CollectiveConfig collective_config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  LaunchDimensions launch_dimensions;
  if (!thunk_proto.has_launch_dimensions()) {
    return absl::InvalidArgumentError(
        "Launch dimensions are required for collective kernel thunk.");
  }
  ABSL_ASSIGN_OR_RETURN(launch_dimensions, LaunchDimensions::FromProto(
                                          thunk_proto.launch_dimensions()));
  CollectiveKernelSpec kernel_spec;
  if (thunk_proto.has_kernel_spec()) {
    const CollectiveKernelSpecProto& proto_spec = thunk_proto.kernel_spec();
    kernel_spec.codegen_config.input_buffer_specs.reserve(
        proto_spec.input_buffer_specs_size());
    for (const auto& input : proto_spec.input_buffer_specs()) {
      TF_RET_CHECK(
          SymmetricMemoryTypeProto_IsValid(input.symmetric_memory_type()))
          << "Invalid symmetric_memory_type: " << input.symmetric_memory_type();
      kernel_spec.codegen_config.input_buffer_specs.push_back(
          {input.requires_multimem(),
           static_cast<SymmetricMemoryType>(input.symmetric_memory_type())});
    }
    kernel_spec.codegen_config.output_buffer_specs.reserve(
        proto_spec.output_buffer_specs_size());
    for (const auto& output : proto_spec.output_buffer_specs()) {
      TF_RET_CHECK(
          SymmetricMemoryTypeProto_IsValid(output.symmetric_memory_type()))
          << "Invalid symmetric_memory_type: "
          << output.symmetric_memory_type();
      kernel_spec.codegen_config.output_buffer_specs.push_back(
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
    kernel_spec.codegen_config.argument_descriptors.reserve(
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
      kernel_spec.codegen_config.argument_descriptors.push_back(
          std::move(arg_desc));
    }
    kernel_spec.codegen_config.sync_count_increment =
        proto_spec.invocation_count_increment();
    kernel_spec.codegen_config.copy_input_to_scratch =
        proto_spec.copy_input_to_scratch();
    kernel_spec.codegen_config.device_sync_count =
        proto_spec.device_sync_count();
  } else {
    // Backward-compatibility fallback for legacy AOT-compiled kernels without
    // an explicit CollectiveKernelSpec.
    // Can be removed in February 2027 (6months backward compatibility window).
    kernel_spec.codegen_config.input_buffer_specs.push_back(
        {/*requires_multimem=*/false, SymmetricMemoryType::kNone});
    kernel_spec.codegen_config.output_buffer_specs.push_back(
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
    kernel_spec.codegen_config.argument_descriptors = {
        {KernelArgType::kInputBuffer, /*index=*/0},
        {KernelArgType::kOutputBuffer, /*index=*/0},
        {KernelArgType::kRuntimeRank},
        {KernelArgType::kInvocationCount},
        {KernelArgType::kScratchBuffer, /*index=*/0},
        {KernelArgType::kScratchBuffer, /*index=*/1}};
    kernel_spec.codegen_config.sync_count_increment =
        1 + static_cast<uint32_t>(GetAllReduceStrategy(
                input_size_bytes, group_size, /*is_multimem_enabled=*/false));
    // Legacy kernels read the host-provided invocation count.
    kernel_spec.codegen_config.device_sync_count = false;
  }
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ABSL_ASSIGN_OR_RETURN(
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
      thunk_info, collective_config, std::move(kernel_spec), std::move(buffers),
      thunk_proto.collective_kernel_enabled(), thunk_proto.kernel_name(),
      launch_dimensions, devices_in_process, thunk_proto.shmem_bytes(),
      std::move(cubin), thunk_proto.use_pdl());
}

absl::StatusOr<ThunkProto> CollectiveKernelThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  CollectiveKernelThunkProto* thunk_proto =
      proto.mutable_collective_kernel_thunk();

  *thunk_proto->mutable_collective_config() = collective_config_.ToProto();

  auto* proto_spec = thunk_proto->mutable_kernel_spec();
  proto_spec->mutable_input_buffer_specs()->Reserve(
      kernel_spec_.codegen_config.input_buffer_specs.size());
  for (const auto& op : kernel_spec_.codegen_config.input_buffer_specs) {
    auto* io_proto = proto_spec->add_input_buffer_specs();
    io_proto->set_requires_multimem(op.requires_multimem);
    io_proto->set_symmetric_memory_type(
        static_cast<SymmetricMemoryTypeProto>(op.symmetric_memory_type));
  }
  proto_spec->mutable_output_buffer_specs()->Reserve(
      kernel_spec_.codegen_config.output_buffer_specs.size());
  for (const auto& res : kernel_spec_.codegen_config.output_buffer_specs) {
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
      kernel_spec_.codegen_config.argument_descriptors.size());
  for (const auto& arg : kernel_spec_.codegen_config.argument_descriptors) {
    auto* arg_proto = proto_spec->add_argument_descriptors();
    arg_proto->set_type(static_cast<KernelArgTypeProto>(arg.type));
    if (arg.index.has_value()) {
      arg_proto->set_index(arg.index.value());
    }
  }
  proto_spec->set_invocation_count_increment(
      kernel_spec_.codegen_config.sync_count_increment);
  proto_spec->set_copy_input_to_scratch(
      kernel_spec_.codegen_config.copy_input_to_scratch);
  proto_spec->set_device_sync_count(
      kernel_spec_.codegen_config.device_sync_count);

  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    ABSL_ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
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
