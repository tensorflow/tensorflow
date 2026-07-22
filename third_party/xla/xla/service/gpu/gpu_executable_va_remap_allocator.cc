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

#include "xla/service/gpu/gpu_executable_va_remap_allocator.h"

#include <cstdint>
#include <memory>
#include <optional>
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
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

uint64_t RoundUpToGranularity(uint64_t size, uint64_t granularity) {
  if (granularity == 0) {
    return size;
  }
  return ((size + granularity - 1) / granularity) * granularity;
}

}  // namespace

absl::StatusOr<uint64_t>
GpuExecutableVaRemapAllocator::Remapping::GetReservationOffset(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_offset.find(idx);
  if (it == allocation_to_reservation_offset.end()) {
    return Internal("No VA reservation offset for allocation %d", idx);
  }
  return it->second;
}

absl::StatusOr<uint64_t>
GpuExecutableVaRemapAllocator::Remapping::GetMappingSize(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_mapping_size.find(idx);
  if (it == allocation_to_mapping_size.end()) {
    return Internal("No VA mapping size for allocation %d", idx);
  }
  return it->second;
}

// Per-run execution scope with command buffer VA remapping active. The
// command-buffer-referenced temp buffer allocations are backed by physical
// VMM allocations while execution sees stable reserved VA addresses. Holds
// the lock for the executable/executor remapping state for the whole
// execution.
class GpuExecutableVaRemapAllocator::VaRemapExecutionScope
    : public GpuExecutableBufferAllocator::ExecutionScope {
 public:
  VaRemapExecutionScope(const GpuExecutableVaRemapAllocator* owner,
                        Remapping* remapping,
                        se::DeviceAddressVmmAllocator* vmm_allocator,
                        std::unique_ptr<absl::MutexLock> remap_lock)
      : ExecutionScope(owner),
        owner_(owner),
        remapping_(remapping),
        vmm_allocator_(vmm_allocator),
        remap_lock_(std::move(remap_lock)) {}

  // Releases any reservation-address aliases still active for this
  // execution. This is a safety net for error paths; the normal release
  // happens inside ExecuteWithBufferAllocations.
  ~VaRemapExecutionScope() override;

  bool va_remap_enabled() const override { return true; }

  // Builds an execution-only address table, maps caller- or allocator-owned
  // input/output buffers into their stable reservation addresses, and passes
  // the persistent allocation indices (constants and VA-remapped allocations)
  // to `execute`. The owning address table remains unchanged for result
  // handling and TearDown.
  absl::Status ExecuteWithBufferAllocations(
      const BufferAllocations& owning_buffer_allocations, int device_ordinal,
      absl::FunctionRef<
          absl::Status(const BufferAllocations&,
                       std::optional<absl::Span<const BufferAllocation::Index>>
                           persistent_alloc_indices)>
          execute) override;

 private:
  // One reservation-address alias installed for this execution that must be
  // released with UnMap() when the execution finishes.
  struct StepAlias {
    uint64_t reservation_offset = 0;
    uint64_t mapping_size = 0;
  };

  absl::Status Prepare(const ServiceExecutableRunOptions* run_options,
                       int device_ordinal) override;
  absl::StatusOr<se::DeviceAddressBase> AllocateTransientBuffer(
      int device_ordinal, const BufferAllocation& allocation,
      int64_t buffer_size,
      se::DeviceAddressAllocator* memory_allocator) override;

  bool ShouldRemapAllocation(BufferAllocation::Index index) const;
  // Returns the reservation slice [offset, offset + size) as an address.
  se::DeviceAddressBase ReservationSlice(uint64_t offset, uint64_t size) const;
  void RecordStepAlias(int device_ordinal, uint64_t reservation_offset,
                       uint64_t mapping_size);
  // UnMaps all aliases recorded by RecordStepAlias. Aliases that fail to
  // unmap remain recorded so the execution-scope destructor can retry them.
  absl::Status ReleaseStepAliases();
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> AllocateBuffer(
      int device_ordinal, const BufferAllocation& allocation,
      int64_t buffer_size);
  // Installs reservation aliases for selected input/output buffers and returns
  // an execution-only address table that refers to them. The owning address
  // table remains unchanged.
  absl::StatusOr<BufferAllocations> GetRemappedBufferAllocations(
      const BufferAllocations& owning_buffer_allocations, int device_ordinal);

  const GpuExecutableVaRemapAllocator* owner_ = nullptr;
  Remapping* remapping_ = nullptr;
  se::DeviceAddressVmmAllocator* vmm_allocator_ = nullptr;
  std::unique_ptr<absl::MutexLock> remap_lock_;

  // Per-execution alias bookkeeping filled by RecordStepAlias.
  int step_device_ordinal_ = -1;
  std::vector<StepAlias> step_aliases_;
};

GpuExecutableVaRemapAllocator::VaRemapExecutionScope::~VaRemapExecutionScope() {
  if (!step_aliases_.empty()) {
    // Normally ReleaseStepAliases runs inside ExecuteWithBufferAllocations;
    // reaching this point means mapping, execution, or an earlier release
    // failed after some aliases were installed.
    absl::Status status = ReleaseStepAliases();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to release command buffer VA remapping aliases "
                    "for module "
                 << owner_->module_name() << ": " << status;
    }
  }
}

absl::Status GpuExecutableVaRemapAllocator::VaRemapExecutionScope::Prepare(
    const ServiceExecutableRunOptions* run_options, int device_ordinal) {
  uint64_t granularity =
      vmm_allocator_->GetAllocationGranularity(run_options->stream()->parent());
  if (remapping_->va_reservation != nullptr &&
      remapping_->granularity != granularity) {
    return Internal(
        "Command buffer VA remapping granularity changed for module %s: "
        "previous=%u current=%u",
        owner_->module_name(), remapping_->granularity, granularity);
  }
  if (remapping_->va_reservation != nullptr) {
    return absl::OkStatus();
  }

  // First execution on this executor creates the persistent reservation. Later
  // executions reuse the same reservation and deterministic layout.
  remapping_->granularity = granularity;
  remapping_->total_size = 0;
  remapping_->allocation_to_reservation_offset.clear();
  remapping_->allocation_to_mapping_size.clear();
  for (BufferAllocation::Index idx : owner_->va_remapped_alloc_indices_) {
    const BufferAllocation& allocation = *owner_->allocations()[idx];
    uint64_t buffer_size = allocation.size();
    uint64_t mapping_size =
        RoundUpToGranularity(buffer_size, remapping_->granularity);
    remapping_->allocation_to_reservation_offset[idx] = remapping_->total_size;
    remapping_->allocation_to_mapping_size[idx] = mapping_size;
    remapping_->total_size = remapping_->total_size + mapping_size;
  }
  ASSIGN_OR_RETURN(
      remapping_->va_reservation,
      vmm_allocator_->CreateReservation(run_options->stream()->parent(),
                                        remapping_->total_size));
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: reserved range for module %s VA=%p total_size=%u "
      "granularity=%u",
      owner_->module_name(), remapping_->va_reservation->address().opaque(),
      remapping_->total_size, remapping_->granularity);
  return absl::OkStatus();
}

bool GpuExecutableVaRemapAllocator::VaRemapExecutionScope::
    ShouldRemapAllocation(BufferAllocation::Index index) const {
  return owner_->va_remapped_alloc_indices_.contains(index);
}

se::DeviceAddressBase
GpuExecutableVaRemapAllocator::VaRemapExecutionScope::ReservationSlice(
    uint64_t offset, uint64_t size) const {
  return se::DeviceAddressBase(
      static_cast<char*>(remapping_->va_reservation->address().opaque()) +
          offset,
      size);
}

void GpuExecutableVaRemapAllocator::VaRemapExecutionScope::RecordStepAlias(
    int device_ordinal, uint64_t reservation_offset, uint64_t mapping_size) {
  DCHECK(step_aliases_.empty() || step_device_ordinal_ == device_ordinal);
  step_device_ordinal_ = device_ordinal;
  step_aliases_.push_back({reservation_offset, mapping_size});
}

absl::Status
GpuExecutableVaRemapAllocator::VaRemapExecutionScope::ReleaseStepAliases() {
  absl::Status status;
  std::vector<StepAlias> failed_aliases;
  failed_aliases.reserve(step_aliases_.size());
  for (const StepAlias& alias : step_aliases_) {
    absl::Status unmap_status = vmm_allocator_->UnMap(
        step_device_ordinal_, remapping_->va_reservation.get(),
        alias.reservation_offset, alias.mapping_size);
    if (!unmap_status.ok()) {
      failed_aliases.push_back(alias);
      status.Update(unmap_status);
    }
  }
  step_aliases_ = std::move(failed_aliases);
  if (step_aliases_.empty()) {
    step_device_ordinal_ = -1;
  }
  return status;
}

absl::StatusOr<BufferAllocations> GpuExecutableVaRemapAllocator::
    VaRemapExecutionScope::GetRemappedBufferAllocations(
        const BufferAllocations& owning_buffer_allocations,
        int device_ordinal) {
  // This is a non-owning copy of the finalized address table. Output donation
  // and copy protection run before this method and may replace input/output
  // entries, so aliases must be installed from these current addresses instead
  // of during GenerateBufferAllocations.
  BufferAllocations execution_allocations(
      owning_buffer_allocations.buffers(),
      owning_buffer_allocations.device_ordinal(),
      owning_buffer_allocations.memory_allocator());

  for (BufferAllocation::Index index : owner_->va_remapped_alloc_indices_) {
    const BufferAllocation& allocation = *owner_->allocations()[index];
    if (!allocation.IsInputOrOutput()) {
      continue;
    }
    const se::DeviceAddressBase buffer =
        owning_buffer_allocations.GetDeviceAddress(index);
    if (buffer.is_null()) {
      return Internal(
          "Command buffer VA remapping selected input/output allocation %d, "
          "but its buffer is null for this execution",
          allocation.index());
    }
    ASSIGN_OR_RETURN(uint64_t va_offset,
                     remapping_->GetReservationOffset(allocation.index()));
    ASSIGN_OR_RETURN(uint64_t mapping_size,
                     remapping_->GetMappingSize(allocation.index()));
    // Map() reactivates a matching stale mapping from the previous execution,
    // so an input/output that keeps its address across executions performs no
    // VMM driver calls here.
    RETURN_IF_ERROR(vmm_allocator_->Map(device_ordinal, buffer,
                                        remapping_->va_reservation.get(),
                                        va_offset, mapping_size));
    RecordStepAlias(device_ordinal, va_offset, mapping_size);
    execution_allocations.GetMutableDeviceAddress(index) =
        ReservationSlice(va_offset, mapping_size);
  }
  return execution_allocations;
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
GpuExecutableVaRemapAllocator::VaRemapExecutionScope::AllocateBuffer(
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
GpuExecutableVaRemapAllocator::VaRemapExecutionScope::AllocateTransientBuffer(
    int device_ordinal, const BufferAllocation& allocation, int64_t buffer_size,
    se::DeviceAddressAllocator* memory_allocator) {
  if (!ShouldRemapAllocation(allocation.index()) ||
      allocation.maybe_live_out()) {
    return ExecutionScope::AllocateTransientBuffer(
        device_ordinal, allocation, buffer_size, memory_allocator);
  }
  ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> buffer,
                   AllocateBuffer(device_ordinal, allocation, buffer_size));
  return buffer.Release();
}

absl::Status GpuExecutableVaRemapAllocator::VaRemapExecutionScope::
    ExecuteWithBufferAllocations(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(
            const BufferAllocations&,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices)>
            execute) {
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: module %s executing with %d command buffer "
      "allocation(s)",
      owner_->module_name(), owner_->va_remapped_alloc_indices_.size());

  absl::StatusOr<BufferAllocations> execution_allocations =
      GetRemappedBufferAllocations(owning_buffer_allocations, device_ordinal);
  if (!execution_allocations.ok()) {
    absl::Status release_status = ReleaseStepAliases();
    if (!release_status.ok()) {
      LOG(ERROR) << "Failed to release command buffer VA remapping aliases "
                    "after mapping failed for module "
                 << owner_->module_name() << ": " << release_status;
    }
    return execution_allocations.status();
  }

  absl::Status execute_status =
      execute(*execution_allocations,
              absl::MakeConstSpan(owner_->persistent_alloc_indices()));
  // Release per-execution aliases even when execution failed. The owning
  // address table was never modified, so result handling and TearDown continue
  // to see caller- or allocator-owned addresses.
  absl::Status release_status = ReleaseStepAliases();
  if (!execute_status.ok() && !release_status.ok()) {
    LOG(ERROR) << "Failed to release command buffer VA remapping aliases "
                  "after execution failed for module "
               << owner_->module_name() << ": " << release_status;
  }
  RETURN_IF_ERROR(execute_status);
  return release_status;
}

GpuExecutableVaRemapAllocator::GpuExecutableVaRemapAllocator(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor)
    : GpuExecutableBufferAllocator(module_name, allocations, result_shape,
                                   debug_options, thunk_executor) {
  const DebugOptions::CommandBufferUpdateMode update_mode =
      debug_options != nullptr
          ? debug_options->xla_gpu_command_buffer_update_mode()
          : DebugOptions::ALWAYS_UPDATE;
  CHECK(update_mode == DebugOptions::SKIP_TEMP)
      << "Unsupported command buffer update mode for VA remapping: "
      << update_mode;

  ForEachCommandBufferAllocation(
      allocations, thunk_executor,
      [&](BufferAllocation::Index index, const BufferAllocation& allocation) {
        if (allocation.is_constant()) {
          // Collected by the base class.
          return;
        }
        if (allocation.IsPreallocatedTempBuffer()) {
          va_remapped_alloc_indices_.insert(index);
        }
      });

  AllocationIndexSet persistent(constant_alloc_indices().begin(),
                                constant_alloc_indices().end());
  persistent.insert(va_remapped_alloc_indices_.begin(),
                    va_remapped_alloc_indices_.end());
  set_persistent_alloc_indices(std::vector<BufferAllocation::Index>(
      persistent.begin(), persistent.end()));

  VLOG(3) << "Command buffer allocation policy: collected "
          << command_buffer_allocation_count()
          << " persistent allocation indices and "
          << va_remapped_alloc_indices_.size()
          << " VA-remapped allocation indices for module "
          << this->module_name();
}

void GpuExecutableVaRemapAllocator::AddVaRemappedAllocationForTesting(
    BufferAllocation::Index index) {
  va_remapped_alloc_indices_.insert(index);
  AllocationIndexSet persistent(constant_alloc_indices().begin(),
                                constant_alloc_indices().end());
  persistent.insert(va_remapped_alloc_indices_.begin(),
                    va_remapped_alloc_indices_.end());
  set_persistent_alloc_indices(std::vector<BufferAllocation::Index>(
      persistent.begin(), persistent.end()));
}

GpuExecutableVaRemapAllocator::~GpuExecutableVaRemapAllocator() {
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
                 << module_name() << ": " << status;
    }
  }
}

absl::StatusOr<std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope>>
GpuExecutableVaRemapAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  auto scope_without_remapping = [&] {
    return GpuExecutableBufferAllocator::CreateExecutionScope(
        run_options, memory_allocator, device_ordinal);
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
        module_name(), executor);
  }
  remapping->vmm_allocator = vmm_allocator;

  // Deferred deallocations and unmaps from the previous execution are left
  // pending on purpose: Allocate() reactivates a compatible stale mapping at
  // the same reservation VA, which keeps the same physical allocation across
  // executions and avoids waiting for the previous execution to retire.
  // Incompatible stale mappings are completed lazily and per-record by the
  // fresh-mapping paths.
  return std::make_unique<VaRemapExecutionScope>(this, remapping, vmm_allocator,
                                                 std::move(remap_lock));
}

}  // namespace gpu
}  // namespace xla
