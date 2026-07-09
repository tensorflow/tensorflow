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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

uint64_t RoundUpToGranularity(uint64_t size, uint64_t granularity) {
  if (granularity == 0) {
    return size;
  }
  return ((size + granularity - 1) / granularity) * granularity;
}

absl::Status CheckAlignment(const BufferAllocation& allocation,
                            se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    }
    if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    }
    return kXlaAllocatedBufferAlignBytes;
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

// Resolve GpuCollectives instance that we should use for the run.
// TODO(ezhulenev): We have almost identical method in `collective_params.cc`,
// this one has to be removed.
GpuCollectives* ResolveGpuCollectives(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options) {
  auto* gpu_options = run_options->run_options().gpu_executable_run_options();
  if (gpu_options && gpu_options->collectives()) {
    return gpu_options->collectives();
  }

  absl::string_view platform_name =
      run_options->run_options().stream()->parent()->GetPlatform()->Name();

  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    absl::StatusOr<Collectives*> collectives = CollectivesRegistry::Get(
        platform_name, debug_options->xla_gpu_collectives_implementation());
    CHECK_OK(collectives)  // Crash OK
        << "Failed to get GPU collectives implementation: "
        << debug_options->xla_gpu_collectives_implementation();
    return absl::down_cast<GpuCollectives*>(*collectives);
  }

  return GpuCollectives::Default(platform_name);
}

}  // namespace

absl::StatusOr<uint64_t>
GpuExecutableBufferAllocator::Remapping::GetReservationOffset(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_offset.find(idx);
  if (it == allocation_to_reservation_offset.end()) {
    return Internal("No VA reservation offset for allocation %d", idx);
  }
  return it->second;
}

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    GpuExecutableBufferAllocator* owner, Remapping* remapping,
    se::DeviceAddressVmmAllocator* vmm_allocator,
    std::unique_ptr<absl::MutexLock> remap_lock, bool address_policy_active)
    : owner_(owner),
      remapping_(remapping),
      vmm_allocator_(vmm_allocator),
      remap_lock_(std::move(remap_lock)),
      address_policy_active_(address_policy_active) {}

absl::StatusOr<GpuExecutableBufferAllocator::MemoryReservationAlias>
GpuExecutableBufferAllocator::ExecutionScope::GetReservationAlias(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_aliases_.find(idx);
  if (it == allocation_to_reservation_aliases_.end()) {
    return Internal("No VA reservation alias for allocation %d", idx);
  }
  return it->second;
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::PrepareReservation(
    const ServiceExecutableRunOptions* run_options, int device_ordinal,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (!command_buffer_active()) {
    return absl::OkStatus();
  }

  uint64_t granularity =
      vmm_allocator_->GetAllocationGranularity(run_options->stream()->parent());
  if (remapping_->va_reservation != nullptr &&
      remapping_->granularity != granularity) {
    return Internal(
        "Command buffer VA remapping granularity changed for module %s: "
        "previous=%u current=%u",
        owner_->module_name_, remapping_->granularity, granularity);
  }
  if (remapping_->va_reservation != nullptr) {
    return absl::OkStatus();
  }

  // First execution on this executor creates the persistent reservation. Later
  // executions reuse the same reservation and deterministic layout.
  remapping_->granularity = granularity;
  remapping_->total_size = 0;
  remapping_->allocation_to_reservation_offset.clear();
  for (BufferAllocation::Index idx :
       owner_->command_buffer_va_remapped_allocation_indexes_) {
    const BufferAllocation& allocation = *owner_->allocations_[idx];
    uint64_t buffer_size = allocation.size();
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size =
          RoundUpToGranularity(buffer_size, static_cast<uint64_t>(it->second));
    }
    remapping_->allocation_to_reservation_offset[idx] = remapping_->total_size;
    remapping_->total_size =
        remapping_->total_size +
        RoundUpToGranularity(buffer_size, remapping_->granularity);
  }
  ASSIGN_OR_RETURN(
      remapping_->va_reservation,
      vmm_allocator_->CreateReservation(run_options->stream()->parent(),
                                        remapping_->total_size));
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: reserved range for module %s VA=%p total_size=%u "
      "granularity=%u",
      owner_->module_name_, remapping_->va_reservation->address().opaque(),
      remapping_->total_size, remapping_->granularity);
  return absl::OkStatus();
}

bool GpuExecutableBufferAllocator::ExecutionScope::ShouldRemapAllocation(
    BufferAllocation::Index index) const {
  if (!command_buffer_active() ||
      !owner_->command_buffer_va_remapped_allocation_indexes_.contains(index)) {
    return false;
  }
  if (!remapping_->policy_va_remapped_index_set.has_value()) {
    return true;
  }
  return remapping_->policy_va_remapped_index_set->contains(index);
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
GpuExecutableBufferAllocator::ExecutionScope::AllocateBuffer(
    int device_ordinal, const BufferAllocation& allocation, int64_t buffer_size,
    bool return_reservation_address) {
  if (!ShouldRemapAllocation(allocation.index())) {
    return Internal("Allocation %d is not command-buffer VA remapped",
                    allocation.index());
  }
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  uint64_t mapping_size = RoundUpToGranularity(
      static_cast<uint64_t>(buffer_size), remapping_->granularity);
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
      vmm_allocator_->Allocate(
          device_ordinal, mapping_size, /*retry_on_failure=*/true,
          /*memory_space=*/allocation.color(), remapping_->va_reservation.get(),
          va_offset, mapping_size, return_reservation_address);
  if (buffer.ok() && !return_reservation_address) {
    se::DeviceAddressBase reservation_address =
        remapping_->va_reservation->address().GetByteSlice(va_offset,
                                                           mapping_size);
    allocation_to_reservation_aliases_[allocation.index()] =
        MemoryReservationAlias{va_offset, mapping_size, reservation_address};
  }
  return buffer;
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::BufferForAllocation(
    ParameterBufferResolver get_parameter_buffer,
    const BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (allocation.is_thread_local()) {
    return se::DeviceAddressBase{};
  }
  if (allocation.is_entry_computation_parameter()) {
    ASSIGN_OR_RETURN(ParameterBuffer registered_buffer,
                     get_parameter_buffer(allocation));
    if (registered_buffer.buffer.is_null() &&
        registered_buffer.buffer.size() > 0 &&
        !registered_buffer.allow_null_buffer) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(),
          registered_buffer.parameter_number);
    }
    return registered_buffer.buffer;
  }
  if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceAddressBase();
    }
    return it->second;
  }

  // Allocate each allocation that might escape, or is the temp buffer.
  CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
  int64_t buffer_size = allocation.size();
  se::DeviceAddressBase buffer_address;
  if (buffer_size > 0) {
    // Maybe round up buffer allocation size to the requested granularity.
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size = RoundUpTo(buffer_size, it->second);
    }
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer;
    if (ShouldRemapAllocation(allocation.index())) {
      bool return_reservation_address =
          !(allocation.maybe_live_out() &&
            owner_->returned_output_allocation_indexes_.contains(
                allocation.index()));
      buffer = AllocateBuffer(device_ordinal, allocation, buffer_size,
                              return_reservation_address);
    } else {
      buffer = memory_allocator->Allocate(device_ordinal, buffer_size,
                                          /*retry_on_failure=*/true,
                                          /*memory_space=*/allocation.color());
    }
    ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> scoped_buffer,
                     std::move(buffer));
    buffer_address = scoped_buffer.Release();
  }
  return buffer_address;
}

absl::StatusOr<BufferAllocations>
GpuExecutableBufferAllocator::ExecutionScope::GenerateBufferAllocations(
    const ServiceExecutableRunOptions* run_options,
    ParameterBufferResolver get_parameter_buffer,
    const BufferAllocToDeviceMemoryMap* globals,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::flat_hash_map<LogicalBuffer::Color, int64_t> allocate_granularity;
  if (auto* collectives =
          ResolveGpuCollectives(run_options, owner_->debug_options_)) {
    // BFC allocator ignores memory alignment and always allocates 256 byte
    // aligned buffers, however for collective memory underlying libraries
    // require larger alignment. We conservatively round up all allocation
    // sizes to the alignment requirement. Proper fix must be done in BFC
    // allocator and all the other allocator adaptors that we have in XLA.
    static constexpr int64_t kCollectiveMemoryColor = 1;
    allocate_granularity[kCollectiveMemoryColor] =
        collectives->SymmetricMemoryAlignment();
  }

  // Tag allocations made in this invocation as multi-device for VMM reuse.
  se::DeviceAddressVmmAllocator::DeviceAssignmentScope
      vmm_device_assignment_scope(
          run_options->run_options().device_assignment());

  const int64_t num_buffers = owner_->allocations_.size();
  RETURN_IF_ERROR(
      PrepareReservation(run_options, device_ordinal, allocate_granularity));

  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *owner_->allocations_[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i,
                            allocate_granularity));
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return BufferAllocations(buffers, device_ordinal, memory_allocator);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::AllocateCopyProtectedOutputBuffer(
    const ServiceExecutableRunOptions* run_options,
    BufferAllocations& buffer_allocations, const ShapeIndex& index,
    const BufferAllocation& allocation, int device_ordinal,
    se::DeviceAddressAllocator* const memory_allocator,
    absl::FunctionRef<absl::Status(absl::Status)> allocation_error) {
  // The caller guards this against aliasing pass-through params, as we do not
  // need to write into the output buffer in that case.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Using copy-protection: aliasing is specified, but the "
         "buffer is not donated; allocating a fresh buffer";
  int64_t allocation_size = ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(owner_->result_shape_, index));
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer;
  if (ShouldRemapAllocation(allocation.index())) {
    allocated_buffer =
        AllocateBuffer(device_ordinal, allocation, allocation_size,
                       /*return_reservation_address=*/false);
  } else {
    allocated_buffer = memory_allocator->Allocate(
        device_ordinal, allocation_size, /*retry_on_failure=*/true,
        /*memory_space=*/allocation.color());
  }
  if (!allocated_buffer.ok()) {
    return allocation_error(allocated_buffer.status());
  }
  se::DeviceAddressBase result_buffer = allocated_buffer->Release();
  se::DeviceAddressBase& aliased_buffer =
      buffer_allocations.GetMutableDeviceAddress(allocation.index());
  CHECK_EQ(aliased_buffer.size(), result_buffer.size());
  RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
      &result_buffer, aliased_buffer, aliased_buffer.size()));
  aliased_buffer = result_buffer;
  return result_buffer;
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::UpdateAllocationAddressPolicy() {
  if (!command_buffer_active() ||
      owner_->update_mode_ == DebugOptions::ALWAYS_UPDATE ||
      remapping_->policy_persistent_alloc_indices.has_value()) {
    return absl::OkStatus();
  }

  if (owner_->update_mode_ != DebugOptions::NEVER_UPDATE &&
      owner_->update_mode_ != DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
    return Internal("Unsupported command buffer update mode: %d",
                    owner_->update_mode_);
  }

  remapping_->policy_persistent_alloc_indices.emplace(
      owner_->command_buffer_persistent_allocation_indexes_.begin(),
      owner_->command_buffer_persistent_allocation_indexes_.end());
  remapping_->policy_va_remapped_index_set.emplace(
      owner_->command_buffer_va_remapped_allocation_indexes_);
  return absl::OkStatus();
}

std::optional<absl::Span<const BufferAllocation::Index>>
GpuExecutableBufferAllocator::ExecutionScope::GetPersistentAllocIndices()
    const {
  if (command_buffer_active()) {
    if (!remapping_->policy_persistent_alloc_indices.has_value()) {
      return std::nullopt;
    }
    return absl::MakeConstSpan(*remapping_->policy_persistent_alloc_indices);
  }
  return absl::MakeConstSpan(owner_->command_buffer_persistent_alloc_indices_);
}

absl::StatusOr<BufferAllocations>
GpuExecutableBufferAllocator::ExecutionScope::BuildExecutionBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal) {
  std::vector<se::DeviceAddressBase> execution_buffers;
  execution_buffers.reserve(owning_buffer_allocations.size());

  struct SourceMapping {
    se::DeviceAddressBase source_address;
    MemoryReservationAlias alias;
  };
  std::vector<SourceMapping> source_mappings;

  for (BufferAllocation::Index i = 0; i < owning_buffer_allocations.size();
       ++i) {
    se::DeviceAddressBase owning_address =
        owning_buffer_allocations.GetDeviceAddress(i);
    if (!ShouldRemapAllocation(i)) {
      execution_buffers.push_back(owning_address);
      continue;
    }
    if (owning_address.is_null()) {
      return Internal("Command buffer allocation %d has null address", i);
    }

    ASSIGN_OR_RETURN(uint64_t va_offset, remapping_->GetReservationOffset(i));

    if (allocation_to_reservation_aliases_.contains(i)) {
      ASSIGN_OR_RETURN(MemoryReservationAlias alias, GetReservationAlias(i));
      execution_buffers.push_back(alias.reservation_address);
      aliases_to_unmap_.push_back(alias);
      continue;
    }

    const BufferAllocation& allocation = *owner_->allocations_[i];
    if (!allocation.is_entry_computation_parameter()) {
      se::DeviceAddressBase expected_reservation_address =
          remapping_->va_reservation->address().GetByteSlice(
              va_offset, owning_address.size());
      if (!owning_address.IsSameAs(expected_reservation_address)) {
        return Internal(
            "Command buffer allocation %d expected reservation-backed "
            "allocator address %p but got %p",
            i, expected_reservation_address.opaque(), owning_address.opaque());
      }
      execution_buffers.push_back(owning_address);
      continue;
    }

    bool reused_source_mapping = false;
    for (const SourceMapping& source_mapping : source_mappings) {
      if (source_mapping.source_address.IsSameAs(owning_address)) {
        execution_buffers.push_back(source_mapping.alias.reservation_address);
        reused_source_mapping = true;
        break;
      }
    }
    if (reused_source_mapping) {
      continue;
    }

    uint64_t mapping_size =
        RoundUpToGranularity(owning_address.size(), remapping_->granularity);
    MemoryReservationAlias alias{
        va_offset, mapping_size,
        remapping_->va_reservation->address().GetByteSlice(va_offset,
                                                           mapping_size)};
    RETURN_IF_ERROR(vmm_allocator_->Map(device_ordinal, owning_address,
                                        remapping_->va_reservation.get(),
                                        alias.reservation_offset, alias.size));
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: Mapped allocation %d for module %s from %p to %p "
        "size=%u",
        i, owner_->module_name_, owning_address.opaque(),
        alias.reservation_address.opaque(), alias.size);
    aliases_to_unmap_.push_back(alias);
    source_mappings.push_back(SourceMapping{owning_address, alias});
    execution_buffers.push_back(alias.reservation_address);
  }

  return BufferAllocations(execution_buffers,
                           owning_buffer_allocations.device_ordinal(),
                           owning_buffer_allocations.memory_allocator());
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::UnmapAliases(
    int device_ordinal) {
  absl::Status status;
  absl::flat_hash_set<void*> unmapped_aliases;
  auto unmap_alias = [&](const MemoryReservationAlias& alias) {
    if (alias.reservation_address.is_null()) {
      return;
    }
    if (!unmapped_aliases.insert(alias.reservation_address.opaque()).second) {
      return;
    }
    absl::Status unmap_status =
        vmm_allocator_->UnMap(device_ordinal, remapping_->va_reservation.get(),
                              alias.reservation_offset, alias.size);
    if (!unmap_status.ok() && status.ok()) {
      status = unmap_status;
    }
  };

  for (const MemoryReservationAlias& alias : aliases_to_unmap_) {
    unmap_alias(alias);
  }
  for (const auto& [_, alias] : allocation_to_reservation_aliases_) {
    unmap_alias(alias);
  }
  return status;
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute) {
  if (!command_buffer_active()) {
    if (!address_policy_active()) {
      return execute(owning_buffer_allocations, std::nullopt);
    }
    return execute(owning_buffer_allocations, GetPersistentAllocIndices());
  }

  RETURN_IF_ERROR(UpdateAllocationAddressPolicy());
  std::optional<absl::Span<const BufferAllocation::Index>>
      persistent_alloc_indices = GetPersistentAllocIndices();
  absl::StatusOr<BufferAllocations> execution_buffer_allocations_or =
      BuildExecutionBufferAllocations(owning_buffer_allocations,
                                      device_ordinal);
  if (!execution_buffer_allocations_or.ok()) {
    absl::Status build_status = execution_buffer_allocations_or.status();
    absl::Status cleanup_status = UnmapAliases(device_ordinal);
    RETURN_IF_ERROR(build_status);
    RETURN_IF_ERROR(cleanup_status);
    return absl::OkStatus();
  }

  BufferAllocations execution_buffer_allocations =
      std::move(execution_buffer_allocations_or).value();
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: module %s executing with %d command buffer allocation(s)",
      owner_->module_name_,
      owner_->command_buffer_va_remapped_allocation_indexes_.size());

  absl::Status execute_status =
      execute(execution_buffer_allocations, persistent_alloc_indices);
  absl::Status unmap_status = UnmapAliases(device_ordinal);

  RETURN_IF_ERROR(execute_status);
  RETURN_IF_ERROR(unmap_status);
  return absl::OkStatus();
}

absl::StatusOr<GpuExecutableBufferAllocator::CommandBufferAllocationIndexes>
GpuExecutableBufferAllocator::CollectCommandBufferAllocationIndexes(
    ThunkExecutor* thunk_executor,
    absl::Span<const BufferAllocation* const> allocations,
    DebugOptions::CommandBufferUpdateMode update_mode) {
  CommandBufferAllocationIndexes allocation_indexes;
  if (thunk_executor == nullptr) {
    return allocation_indexes;
  }
  if (update_mode != DebugOptions::NEVER_UPDATE &&
      update_mode != DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
    return allocation_indexes;
  }

  RETURN_IF_ERROR(thunk_executor->thunks().WalkNested([&](const Thunk* thunk)
                                                          -> absl::Status {
    auto* command_buffer_thunk = dynamic_cast<const CommandBufferThunk*>(thunk);
    if (command_buffer_thunk == nullptr) return absl::OkStatus();
    return command_buffer_thunk->WalkCommands(
        [&](const Command* command) -> absl::Status {
          bool remap_command_allocs =
              update_mode != DebugOptions::CAPTURE_CMD_NEVER_UPDATE ||
              command->IsTracedCommand();
          for (const BufferUse& use : command->buffer_uses()) {
            BufferAllocation::Index index = use.slice().index();
            if (index >= 0 && static_cast<size_t>(index) < allocations.size()) {
              const BufferAllocation& allocation = *allocations[index];
              if (allocation.size() == 0) {
                continue;
              }
              if (allocation.is_constant()) {
                allocation_indexes.persistent.insert(index);
                continue;
              }
            }
            if (remap_command_allocs) {
              allocation_indexes.persistent.insert(index);
              allocation_indexes.va_remapped.insert(index);
            }
          }
          return absl::OkStatus();
        });
  }));
  return allocation_indexes;
}

GpuExecutableBufferAllocator::GpuExecutableBufferAllocator(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor,
    AllocationIndexSet returned_output_allocation_indexes)
    : module_name_(module_name),
      allocations_(allocations.begin(), allocations.end()),
      result_shape_(result_shape),
      debug_options_(debug_options),
      update_mode_(debug_options_ != nullptr
                       ? debug_options_->xla_gpu_command_buffer_update_mode()
                       : DebugOptions::ALWAYS_UPDATE),
      returned_output_allocation_indexes_(
          std::move(returned_output_allocation_indexes)) {
  auto command_buffer_allocation_indexes =
      CollectCommandBufferAllocationIndexes(thunk_executor, allocations_,
                                            update_mode_);
  CHECK_OK(command_buffer_allocation_indexes.status());
  CommandBufferAllocationIndexes indexes =
      std::move(command_buffer_allocation_indexes).value();
  command_buffer_persistent_allocation_indexes_ = std::move(indexes.persistent);
  command_buffer_persistent_alloc_indices_.assign(
      command_buffer_persistent_allocation_indexes_.begin(),
      command_buffer_persistent_allocation_indexes_.end());
  command_buffer_va_remapped_allocation_indexes_ =
      std::move(indexes.va_remapped);

  VLOG(3) << "VA remapping: collected "
          << command_buffer_persistent_allocation_indexes_.size()
          << " persistent allocation indexes and "
          << command_buffer_va_remapped_allocation_indexes_.size()
          << " VA-remapped allocation indexes for module " << module_name_;
}

GpuExecutableBufferAllocator::~GpuExecutableBufferAllocator() {
  absl::MutexLock lock(remappings_mutex_);
  for (auto& [executor, remapping] : remappings_) {
    absl::MutexLock remap_lock(remapping.mutex);
    if (remapping.vmm_allocator == nullptr) {
      continue;
    }
    absl::Status status = remapping.vmm_allocator->SynchronizePendingOperations(
        executor->device_ordinal());
    if (!status.ok()) {
      LOG(ERROR) << "Failed to synchronize command buffer VA remapping "
                    "deferred operations for module "
                 << module_name_ << ": " << status;
    }
  }
}

absl::StatusOr<GpuExecutableBufferAllocator::ExecutionScope>
GpuExecutableBufferAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  auto inactive_scope = [&] {
    return ExecutionScope(this, nullptr, nullptr, nullptr,
                          /*address_policy_active=*/false);
  };

  if (update_mode_ == DebugOptions::ALWAYS_UPDATE) {
    return inactive_scope();
  }
  if (update_mode_ != DebugOptions::NEVER_UPDATE &&
      update_mode_ != DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
    return Internal("Unsupported command buffer update mode: %d", update_mode_);
  }
  if (command_buffer_persistent_allocation_indexes_.empty()) {
    return inactive_scope();
  }
  if (command_buffer_va_remapped_allocation_indexes_.empty()) {
    return ExecutionScope(this, nullptr, nullptr, nullptr,
                          /*address_policy_active=*/true);
  }

  auto* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator);
  if (vmm_allocator == nullptr) {
    return inactive_scope();
  }

  ASSIGN_OR_RETURN(se::Stream * allocator_stream,
                   vmm_allocator->GetStream(device_ordinal));
  if (allocator_stream != run_options->stream()) {
    return Internal(
        "Command buffer VA remapping requires the VMM allocator stream "
        "and execution stream to match");
  }

  Remapping* remapping = nullptr;
  se::StreamExecutor* executor = run_options->stream()->parent();
  {
    absl::MutexLock lock(remappings_mutex_);
    // This is the lifetime remapping object for this executable/executor. It
    // owns the VA reservation reused by later ExecuteAsyncOnStream calls.
    remapping = &remappings_[executor];
  }

  auto remap_lock = std::make_unique<absl::MutexLock>(&remapping->mutex);
  if (remapping->vmm_allocator != nullptr &&
      remapping->vmm_allocator != vmm_allocator) {
    return Internal(
        "Command buffer VA remapping for module %s changed VMM allocator for "
        "executor %p",
        module_name_, executor);
  }
  remapping->vmm_allocator = vmm_allocator;

  // Deferred deallocations and unmaps from the previous execution are left
  // pending on purpose: Allocate() and Map() reactivate a compatible stale
  // mapping at the same reservation VA, which keeps the same physical
  // allocation across executions and avoids waiting for the previous execution
  // to retire. Incompatible stale mappings are completed lazily and per-record
  // by the fresh-mapping paths.
  return ExecutionScope(this, remapping, vmm_allocator, std::move(remap_lock),
                        /*address_policy_active=*/true);
}

}  // namespace gpu
}  // namespace xla
