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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::ScopedAnnotation;

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

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    GpuExecutableBufferAllocator* owner)
    : owner_(owner) {}

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    GpuExecutableBufferAllocator* owner,
    GpuExecutableBufferAllocator::VaRanges* va_ranges,
    const ServiceExecutableRunOptions* run_options)
    : owner_(owner), va_ranges_(va_ranges), run_options_(run_options) {}

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
    ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> buffer,
        memory_allocator->Allocate(device_ordinal, buffer_size,
                                   /*retry_on_failure=*/true,
                                   /*memory_space=*/allocation.color()));
    buffer_address = buffer.Release();
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
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
      memory_allocator->Allocate(device_ordinal, allocation_size,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/allocation.color());
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
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<absl::Status(const BufferAllocations&)> execute) {
  if (command_buffer_active()) {
    return ExecuteWithVaRemapping(owning_buffer_allocations, device_ordinal,
                                  execute);
  }
  return execute(owning_buffer_allocations);
}

// VA remapping execution flow for 2 consecutive calls on the same executor:
//
// clang-format off
// NOLINTBEGIN
//                   +---------------------+---------------------++---------------------+---------------------+
// GPU               |  VA1 Execute        |  VA2 Execute        ||  VA1 Execute        |  VA2 Execute        |
//                   +---------------------+---------------------++---------------------+---------------------+
//         +---------+             +---------++---------+             +---------++---------+
// CPU     | VA Map  |             |VA UnMap || VA Map  |             |VA UnMap || VA Map  |
//         +---------+             +---------++---------+             +---------++---------+
// NOLINTEND
// clang-format on
absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithVaRemapping(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<absl::Status(const BufferAllocations&)> execute) {
  se::StreamExecutor* executor = run_options_->stream()->parent();

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "VA remapping: module " << owner_->module_name_ << " num_allocations="
      << owner_->command_buffer_allocation_indexes_.size();

  // Get the DeviceAddressVmmAllocator to look up physical allocations.
  // vmm_allocator is guaranteed non-null here because CreateExecutionScope
  // already checked for it.
  se::DeviceAddressVmmAllocator* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(run_options_->allocator());
  if (vmm_allocator == nullptr) {
    return Internal("DeviceAddressVmmAllocator cast failed unexpectedly");
  }

  uint64_t granularity = vmm_allocator->GetAllocationGranularity(executor);

  // Acquire per-executor mutex to protect VA range operations. This ensures
  // only one thread uses the VA ranges at a time for this executor.
  absl::MutexLock va_lock(va_ranges_->mutex);

  // Initialize VA ranges if this is first use (va_reservation is null).
  if (va_ranges_->va_reservation == nullptr) {
    ScopedAnnotation annotation_va_reserve([&] {
      return absl::StrFormat("command_buffer_va_range_reserve:#module=%s#",
                             owner_->module_name_);
    });

    // Calculate total size for all command buffer allocations, rounding each
    // allocation up to the allocation granularity.
    uint64_t total_va_size = 0;
    for (BufferAllocation::Index i :
         owner_->command_buffer_allocation_indexes_) {
      const uint64_t size =
          owning_buffer_allocations.GetDeviceAddress(i).size();
      total_va_size += RoundUpToGranularity(size, granularity);
    }

    // Reserve a single large VA range for all command buffer allocations.
    ASSIGN_OR_RETURN(va_ranges_->va_reservation,
                     vmm_allocator->CreateReservation(executor, total_va_size));
    ASSIGN_OR_RETURN(va_ranges_->unmap_event, executor->CreateEvent());

    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: Reserved single VA range for module %s "
        "VA: %p total_size: %d granularity: %d",
        owner_->module_name_, va_ranges_->va_reservation->address().opaque(),
        total_va_size, granularity);
  } else {
    ScopedAnnotation annotation_va_unmap([&] {
      return absl::StrFormat("command_buffer_va_range_unmap:#module=%s#",
                             owner_->module_name_);
    });

    // VA range is already initialized; wait for the unmap event to be marked
    // and then do the VA unmapping.
    RETURN_IF_ERROR(va_ranges_->unmap_event->Synchronize());

    // Unmap physical addresses from the single reserved VA range. Clearing
    // ScopedMappings calls UnMap via their destructors.
    va_ranges_->scoped_mapping.reset();
  }

  // Build a map from allocation index to its offset within va_reservation.
  // Iterate through command_buffer_allocation_indexes_ in order (btree_set
  // provides deterministic iteration order) and accumulate offsets.
  absl::flat_hash_map<BufferAllocation::Index, uint64_t> allocation_va_offsets;
  uint64_t current_offset = 0;
  for (BufferAllocation::Index idx :
       owner_->command_buffer_allocation_indexes_) {
    const uint64_t size =
        owning_buffer_allocations.GetDeviceAddress(idx).size();
    allocation_va_offsets[idx] = current_offset;
    current_offset += RoundUpToGranularity(size, granularity);
  }

  if (!allocation_va_offsets.empty() && va_ranges_->va_reservation == nullptr) {
    return Internal("Reserved VA address range is null");
  }

  // Map physical memory to reserved VA addresses.
  std::vector<se::DeviceAddressBase> mapped_buffers;
  mapped_buffers.reserve(owning_buffer_allocations.size());

  {
    ScopedAnnotation annotation_va_remap([&] {
      return absl::StrFormat("command_buffer_va_range_remap:#module=%s#",
                             owner_->module_name_);
    });

    // Collect mapping descriptors for the batch MapTo call. Descriptors are
    // accumulated in reservation_offset order (guaranteed because
    // allocation_va_offsets was built from a sorted btree_set and the loop
    // below iterates allocation indices in ascending order).
    std::vector<se::MemoryReservation::MappingDescriptor> mapping_descriptors;

    const BufferAllocation::Index num_allocations =
        static_cast<BufferAllocation::Index>(owning_buffer_allocations.size());
    for (BufferAllocation::Index i = 0; i < num_allocations; ++i) {
      se::DeviceAddressBase original_buffer =
          owning_buffer_allocations.GetDeviceAddress(i);

      // Only do VA mapping for allocations accessed by CommandBufferThunk.
      auto offset_it = allocation_va_offsets.find(i);
      if (offset_it == allocation_va_offsets.end()) {
        // Not a command buffer allocation (or zero-size), use the original
        // buffer.
        mapped_buffers.push_back(original_buffer);
        continue;
      }

      // This allocation is used by command buffer - validate it's not null.
      if (original_buffer.is_null()) {
        return Internal("Command buffer allocation %d has null address", i);
      }

      // Get the physical memory allocation from the VMM allocator.
      se::MemoryAllocation* raw_alloc = vmm_allocator->GetRawAllocation(
          executor->device_ordinal(), original_buffer);
      if (raw_alloc == nullptr) {
        return Internal(
            "No raw allocation found for command buffer allocation %d", i);
      }
      const uint64_t mapping_size = raw_alloc->address().size();

      // Calculate the sub-range VA address for this allocation.
      uint64_t va_offset = offset_it->second;
      void* sub_range_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(
              va_ranges_->va_reservation->address().opaque()) +
          va_offset);
      se::DeviceAddressBase sub_range_va(sub_range_ptr, original_buffer.size());

      XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
          "Mapping allocation %d physical: %p -> VA: %p "
          "(offset: %d) size: %d",
          i, original_buffer.opaque(), sub_range_va.opaque(), va_offset,
          original_buffer.size());

      mapping_descriptors.push_back(
          {va_offset, /*allocation_offset=*/0, mapping_size, raw_alloc});

      // Use VA address for execution.
      mapped_buffers.push_back(
          se::DeviceAddressBase(sub_range_va.opaque(), original_buffer.size()));
    }

    // Batch-map all command buffer allocations into the reserved VA range in
    // a single call. This maps the contiguous range formed by the descriptors
    // and enables device access before returning.
    if (!mapping_descriptors.empty()) {
      ASSIGN_OR_RETURN(se::MemoryReservation::ScopedMapping scoped_mapping,
                       va_ranges_->va_reservation->MapTo(
                           absl::MakeSpan(mapping_descriptors)));
      va_ranges_->scoped_mapping = std::move(scoped_mapping);
    }
  }

  if (VLOG_IS_ON(3)) {
    void* va_base = (va_ranges_->va_reservation != nullptr)
                        ? va_ranges_->va_reservation->address().opaque()
                        : nullptr;
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: Mapped %d allocations to single VA range at %p",
        allocation_va_offsets.size(), va_base);
    for (const auto& [alloc_idx, va_offset] : allocation_va_offsets) {
      se::DeviceAddressBase physical_addr =
          owning_buffer_allocations.GetDeviceAddress(alloc_idx);
      void* va_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(va_base) + va_offset);
      XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
          "  allocation[%d] physical: %p -> VA: %p (offset: %d) size: %d",
          alloc_idx, physical_addr.opaque(), va_ptr, va_offset,
          physical_addr.size());
    }
  }

  BufferAllocations remapped_buffer_allocations(
      mapped_buffers, owning_buffer_allocations.device_ordinal(),
      owning_buffer_allocations.memory_allocator());

  RETURN_IF_ERROR(execute(remapped_buffer_allocations));

  // Record event so VA range can be reclaimed after GPU finishes.
  RETURN_IF_ERROR(
      run_options_->stream()->RecordEvent(va_ranges_->unmap_event.get()));

  return absl::OkStatus();
}

absl::StatusOr<GpuExecutableBufferAllocator::AllocationIndexSet>
GpuExecutableBufferAllocator::CollectCommandBufferAllocationIndexes(
    ThunkExecutor* thunk_executor,
    absl::Span<const BufferAllocation* const> allocations,
    DebugOptions::CommandBufferUpdateMode update_mode) {
  AllocationIndexSet allocation_indexes;
  if (thunk_executor == nullptr ||
      (update_mode != DebugOptions::NEVER_UPDATE &&
       update_mode != DebugOptions::CAPTURE_CMD_NEVER_UPDATE)) {
    return allocation_indexes;
  }

  RETURN_IF_ERROR(
      thunk_executor->thunks().WalkNested([&](const Thunk* t) -> absl::Status {
        auto* cbt = dynamic_cast<const CommandBufferThunk*>(t);
        if (cbt == nullptr) {
          return absl::OkStatus();
        }
        return cbt->WalkCommands([&](const Command* cmd) -> absl::Status {
          if (update_mode == DebugOptions::CAPTURE_CMD_NEVER_UPDATE &&
              !cmd->IsTracedCommand()) {
            return absl::OkStatus();
          }
          for (const BufferUse& use : cmd->buffer_uses()) {
            BufferAllocation::Index index = use.slice().index();
            if (index >= 0 && index < allocations.size()) {
              const BufferAllocation* alloc = allocations[index];
              if (alloc->is_constant() || alloc->size() == 0) {
                continue;
              }
            }
            allocation_indexes.insert(index);
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
    DebugOptions::CommandBufferUpdateMode update_mode,
    AllocationIndexSet allocation_indexes)
    : module_name_(module_name),
      allocations_(allocations.begin(), allocations.end()),
      result_shape_(result_shape),
      debug_options_(debug_options),
      update_mode_(update_mode),
      command_buffer_allocation_indexes_(std::move(allocation_indexes)) {
  VLOG(3) << "VA remapping: collected "
          << command_buffer_allocation_indexes_.size()
          << " allocation indexes for module " << module_name_;
}

absl::StatusOr<GpuExecutableBufferAllocator::ExecutionScope>
GpuExecutableBufferAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  if (command_buffer_allocation_indexes_.empty() ||
      update_mode_ == DebugOptions::ALWAYS_UPDATE ||
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator) ==
          nullptr) {
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "CreateExecutionScope: command_buffer_allocation_indexes_.size()=%d "
        "use_command_buffer_va_remapping=0",
        command_buffer_allocation_indexes_.size());
    return ExecutionScope(this);
  }

  se::StreamExecutor* executor = run_options->stream()->parent();
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "CreateExecutionScope: command_buffer_allocation_indexes_.size()=%d "
      "use_command_buffer_va_remapping=1",
      command_buffer_allocation_indexes_.size());

  VaRanges* va_ranges = nullptr;
  {
    absl::MutexLock lock(va_ranges_mutex_);
    va_ranges = &module_va_ranges_[executor];
  }
  return ExecutionScope(this, va_ranges, run_options);
}

}  // namespace gpu
}  // namespace xla
