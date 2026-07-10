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

#include "absl/container/flat_hash_map.h"
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
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
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

namespace xla::gpu {
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

struct CollectedAllocationIndices {
  GpuExecutableBufferAllocator::AllocationIndexSet constant;
  GpuExecutableBufferAllocator::AllocationIndexSet persistent;
  GpuExecutableBufferAllocator::AllocationIndexSet va_remapped;
};

CollectedAllocationIndices CollectAllocationIndices(
    absl::Span<const BufferAllocation* const> allocations,
    const ThunkExecutor* thunk_executor, bool persist_temp_allocations) {
  CollectedAllocationIndices indices;
  if (thunk_executor == nullptr) {
    return indices;
  }

  CHECK_OK(thunk_executor->thunks().WalkNested(
      [&](const Thunk* thunk) -> absl::Status {
        auto* command_buffer_thunk =
            dynamic_cast<const CommandBufferThunk*>(thunk);
        if (command_buffer_thunk == nullptr) {
          return absl::OkStatus();
        }
        for (BufferAllocation::Index index :
             command_buffer_thunk->allocs_indices()) {
          if (index < 0 || static_cast<size_t>(index) >= allocations.size()) {
            continue;
          }
          const BufferAllocation& allocation = *allocations[index];
          if (allocation.size() == 0) {
            continue;
          }
          if (allocation.is_constant()) {
            indices.constant.insert(index);
          } else if (persist_temp_allocations &&
                     allocation.IsPreallocatedTempBuffer()) {
            indices.va_remapped.insert(index);
          }
        }
        return absl::OkStatus();
      }));

  indices.persistent = indices.constant;
  indices.persistent.insert(indices.va_remapped.begin(),
                            indices.va_remapped.end());
  return indices;
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
    std::unique_ptr<absl::MutexLock> remap_lock)
    : owner_(owner),
      remapping_(remapping),
      vmm_allocator_(vmm_allocator),
      remap_lock_(std::move(remap_lock)) {}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::PrepareReservation(
    const ServiceExecutableRunOptions* run_options, int device_ordinal) {
  if (!va_remap_enabled()) {
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
  for (BufferAllocation::Index idx : owner_->va_remapped_alloc_indices_) {
    const BufferAllocation& allocation = *owner_->allocations_[idx];
    uint64_t buffer_size = allocation.size();
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
  return va_remap_enabled() &&
         owner_->va_remapped_alloc_indices_.contains(index);
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
GpuExecutableBufferAllocator::ExecutionScope::AllocateBuffer(
    int device_ordinal, const BufferAllocation& allocation,
    int64_t buffer_size) {
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  uint64_t mapping_size = RoundUpToGranularity(
      static_cast<uint64_t>(buffer_size), remapping_->granularity);
  return vmm_allocator_->Allocate(
      device_ordinal, mapping_size, /*retry_on_failure=*/true,
      /*memory_space=*/allocation.color(), remapping_->va_reservation.get(),
      va_offset, mapping_size, /*return_reservation_address=*/true);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::BufferForAllocation(
    ParameterBufferResolver get_parameter_buffer,
    const BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx) {
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
  int64_t buffer_size = allocation.size();
  se::DeviceAddressBase buffer_address;
  if (buffer_size > 0) {
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer;
    if (ShouldRemapAllocation(allocation.index())) {
      buffer = AllocateBuffer(device_ordinal, allocation, buffer_size);
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

  // Tag allocations made in this invocation as multi-device for VMM reuse.
  se::DeviceAddressVmmAllocator::DeviceAssignmentScope
      vmm_device_assignment_scope(
          run_options->run_options().device_assignment());

  const int64_t num_buffers = owner_->allocations_.size();
  RETURN_IF_ERROR(PrepareReservation(run_options, device_ordinal));

  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *owner_->allocations_[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i));
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
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute) {
  if (va_remap_enabled()) {
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: module %s executing with %d command buffer "
        "allocation(s)",
        owner_->module_name_, owner_->va_remapped_alloc_indices_.size());
    return execute(owning_buffer_allocations,
                   absl::MakeConstSpan(owner_->persistent_alloc_indices_));
  }
  return execute(owning_buffer_allocations,
                 absl::MakeConstSpan(owner_->constant_alloc_indices_));
}

GpuExecutableBufferAllocator::GpuExecutableBufferAllocator(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor)
    : module_name_(module_name),
      allocations_(allocations.begin(), allocations.end()),
      result_shape_(result_shape),
      debug_options_(debug_options) {
  const DebugOptions::CommandBufferUpdateMode update_mode =
      debug_options_ != nullptr
          ? debug_options_->xla_gpu_command_buffer_update_mode()
          : DebugOptions::ALWAYS_UPDATE;
  CHECK(update_mode == DebugOptions::ALWAYS_UPDATE ||
        update_mode == DebugOptions::SKIP_TEMP)
      << "Unsupported command buffer update mode: " << update_mode;

  CollectedAllocationIndices indices = CollectAllocationIndices(
      allocations_, thunk_executor, update_mode == DebugOptions::SKIP_TEMP);
  constant_alloc_indices_.assign(indices.constant.begin(),
                                 indices.constant.end());
  persistent_alloc_indices_.assign(indices.persistent.begin(),
                                   indices.persistent.end());
  va_remapped_alloc_indices_ = std::move(indices.va_remapped);

  VLOG(3) << "Command buffer allocation policy: collected "
          << persistent_alloc_indices_.size()
          << " persistent allocation indices and "
          << va_remapped_alloc_indices_.size()
          << " VA-remapped allocation indices for module " << module_name_;
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
  auto scope_without_remapping = [&] {
    return ExecutionScope(this, nullptr, nullptr, nullptr);
  };

  if (va_remapped_alloc_indices_.empty()) {
    return scope_without_remapping();
  }

  auto* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator);
  if (vmm_allocator == nullptr) {
    return scope_without_remapping();
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

  auto remap_lock = std::make_unique<absl::MutexLock>(remapping->mutex);
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
  return ExecutionScope(this, remapping, vmm_allocator, std::move(remap_lock));
}

}  // namespace xla::gpu
