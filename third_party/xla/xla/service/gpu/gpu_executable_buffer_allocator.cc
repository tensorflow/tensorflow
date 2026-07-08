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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
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

namespace vmm_internal {

// The steady-state command-buffer execution remaps the SAME physical buffers
// into the SAME reservation slots every step, so the per-step VA unmap/remap is
// redundant work. Driving the already-tested
// se::MemoryReservation::ScopedMapping::Remap() with a per-slot change cache
// lets unchanged slots keep their mapping (no unmap/map/SetAccess), and skips
// the unmap-event sync entirely when nothing changed.
//
// ROCm-only by design: other platforms keep their existing full
// reset()+MapTo() behavior unchanged. The flag
// (xla_gpu_experimental_command_buffer_vmm_skip_remap) lets ROCm fall back to
// the legacy path for A/B measurement; it has no effect on other platforms.
bool VmmRemapSkipEnabled(absl::string_view platform_name, bool flag_enabled) {
  if (platform_name != "ROCM") {
    return false;
  }
  return flag_enabled;
}

// Size threshold (bytes) for the ROCm copy-into-shadow path. Command-buffer
// slices with size <= this value bypass VA remapping: they get a stable shadow
// buffer (mapped once) and are refreshed with a stream-ordered D2D copy each
// step instead of an hipMemUnmap/Map/SetAccess round-trip. This targets the
// small slices (scale/scalar/metric buffers) that change address every step.
// 0 (default) disables the path entirely, so other platforms and the default
// ROCm behavior are unchanged. A negative flag value is clamped to 0.
uint64_t VmmCopyThresholdBytes(absl::string_view platform_name,
                               int64_t flag_threshold_bytes) {
  if (platform_name != "ROCM" || flag_threshold_bytes <= 0) {
    return 0;
  }
  return static_cast<uint64_t>(flag_threshold_bytes);
}

}  // namespace vmm_internal

namespace {

uint64_t RoundUpToGranularity(uint64_t size, uint64_t granularity) {
  if (granularity == 0) {
    return size;
  }
  return ((size + granularity - 1) / granularity) * granularity;
}

// A command-buffer slice qualifies for the copy-into-shadow path when the
// threshold is enabled (>0) and the slice is non-empty and at or below it.
bool IsSmallCopySlice(uint64_t size, uint64_t copy_threshold) {
  return copy_threshold > 0 && size > 0 && size <= copy_threshold;
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

std::optional<absl::Span<const BufferAllocation::Index>>
GpuExecutableBufferAllocator::ExecutionScope::GetPersistentAllocIndices()
    const {
  // Only when this scope drives VMM VA-remapping are the module's
  // command-buffer allocations at stable (reservation-backed or shadow)
  // addresses; report them as persistent so the command buffer is traced once
  // and never updated. When inactive, return nullopt so the command buffer
  // treats every allocation as dynamic (the pre-upstream default behavior for
  // this path).
  if (!command_buffer_active()) {
    return std::nullopt;
  }
  return absl::MakeConstSpan(owner_->command_buffer_persistent_alloc_indices_);
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute) {
  std::optional<absl::Span<const BufferAllocation::Index>>
      persistent_alloc_indices = GetPersistentAllocIndices();
  if (command_buffer_active()) {
    return ExecuteWithVaRemapping(owning_buffer_allocations, device_ordinal,
                                  execute, persistent_alloc_indices);
  }
  return execute(owning_buffer_allocations, persistent_alloc_indices);
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
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute,
    std::optional<absl::Span<const BufferAllocation::Index>>
        persistent_alloc_indices) {
  se::StreamExecutor* executor = run_options_->stream()->parent();

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "VA remapping: module " << owner_->module_name_ << " num_allocations="
      << owner_->command_buffer_va_remapped_allocation_indexes_.size();

  // Get the DeviceAddressVmmAllocator to look up physical allocations.
  // vmm_allocator is guaranteed non-null here because CreateExecutionScope
  // already checked for it.
  se::DeviceAddressVmmAllocator* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(run_options_->allocator());
  if (vmm_allocator == nullptr) {
    return Internal("DeviceAddressVmmAllocator cast failed unexpectedly");
  }

  uint64_t granularity = vmm_allocator->GetAllocationGranularity(executor);

  // ROCm-only per-slot skip-remap: reuse mappings for slots whose source
  // buffer did not move since the previous step. Gated by the
  // xla_gpu_experimental_command_buffer_vmm_* DebugOptions flags (defaults:
  // skip-remap on, copy-into-shadow disabled); falls back to those defaults if
  // no DebugOptions were provided.
  const absl::string_view platform_name = executor->GetPlatform()->Name();
  const DebugOptions* debug_options = owner_->debug_options_;
  const bool skip_remap_flag =
      debug_options == nullptr
          ? true
          : debug_options->xla_gpu_experimental_command_buffer_vmm_skip_remap();
  const bool skip_remap_enabled =
      vmm_internal::VmmRemapSkipEnabled(platform_name, skip_remap_flag);

  // ROCm copy-into-shadow: small command-buffer slices bypass VA remapping and
  // are refreshed by a stream-ordered D2D copy. 0 = disabled (default).
  int64_t copy_threshold_flag = 0;
  if (debug_options != nullptr) {
    copy_threshold_flag =
        debug_options
            ->xla_gpu_experimental_command_buffer_vmm_copy_threshold_bytes();
  }
  const uint64_t copy_threshold =
      vmm_internal::VmmCopyThresholdBytes(platform_name, copy_threshold_flag);

  // Acquire per-executor mutex to protect VA range operations. This ensures
  // only one thread uses the VA ranges at a time for this executor.
  absl::MutexLock va_lock(va_ranges_->mutex);

  RETURN_IF_ERROR(EnsureVaReservation(owning_buffer_allocations, executor,
                                      vmm_allocator, granularity,
                                      copy_threshold, device_ordinal));
  // NOTE: when the VA range already holds a mapping from a previous step, the
  // unmap-event sync and the unmap itself are intentionally deferred to
  // EstablishMapping. This lets the ROCm skip-remap path compare this step's
  // source buffers against the previous step and avoid unmapping (and even
  // avoid the unmap-event sync) for slots that did not change.

  const absl::flat_hash_map<BufferAllocation::Index, uint64_t>
      allocation_va_offsets = ComputeReservationOffsets(
          owning_buffer_allocations, granularity, copy_threshold);
  if (!allocation_va_offsets.empty() && va_ranges_->va_reservation == nullptr) {
    return Internal("Reserved VA address range is null");
  }

  // Build this step's mapping plan (VA descriptors, shadow copies, and the
  // buffers handed to execution) and (re)establish the VA->physical mapping.
  StepMappingPlan plan;
  {
    ScopedAnnotation annotation_va_remap([&] {
      return absl::StrFormat("command_buffer_va_range_remap:#module=%s#",
                             owner_->module_name_);
    });
    RETURN_IF_ERROR(BuildStepMappingPlan(
        owning_buffer_allocations, executor, vmm_allocator, granularity,
        copy_threshold, allocation_va_offsets, device_ordinal, plan));
    RETURN_IF_ERROR(EstablishMapping(plan, skip_remap_enabled, device_ordinal));
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
      plan.mapped_buffers, owning_buffer_allocations.device_ordinal(),
      owning_buffer_allocations.memory_allocator());

  // Copy-IN small-slice inputs (real->shadow), stream-ordered before the
  // command buffer so each shadow holds this step's data without any VA remap
  // or unmap-event sync.
  if (!plan.small_copies.empty()) {
    se::Stream* exec_stream = run_options_->stream();
    for (SmallSliceCopy& c : plan.small_copies) {
      RETURN_IF_ERROR(exec_stream->Memcpy(&c.shadow, c.real, c.size));
    }
    XLA_VLOG_DEVICE(2, device_ordinal) << absl::StreamFormat(
        "VA remapping: copied %d small slices to shadow (threshold=%d B)",
        static_cast<int>(plan.small_copies.size()), copy_threshold);
  }

  RETURN_IF_ERROR(
      execute(remapped_buffer_allocations, persistent_alloc_indices));

  // Copy-BACK small-slice outputs (shadow->real), stream-ordered after the
  // command buffer wrote results (e.g. loss / token-count scalars) into the
  // stable shadow, propagating them to the real buffers the host/later thunks
  // read. Runs before the unmap event is recorded.
  if (!plan.small_copies.empty()) {
    se::Stream* exec_stream = run_options_->stream();
    for (SmallSliceCopy& c : plan.small_copies) {
      RETURN_IF_ERROR(exec_stream->Memcpy(&c.real, c.shadow, c.size));
    }
  }

  // Record event so VA range can be reclaimed after GPU finishes.
  RETURN_IF_ERROR(
      run_options_->stream()->RecordEvent(va_ranges_->unmap_event.get()));

  return absl::OkStatus();
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::EnsureVaReservation(
    const BufferAllocations& owning_buffer_allocations,
    se::StreamExecutor* executor, se::DeviceAddressVmmAllocator* vmm_allocator,
    uint64_t granularity, uint64_t copy_threshold, int device_ordinal) {
  // Initialize VA ranges on first use. The unmap_event is the init sentinel
  // (always created); the VA reservation may be absent when every
  // command-buffer slice is handled by the copy-into-shadow path
  // (total_va_size == 0).
  if (va_ranges_->unmap_event != nullptr) {
    return absl::OkStatus();
  }
  ScopedAnnotation annotation_va_reserve([&] {
    return absl::StrFormat("command_buffer_va_range_reserve:#module=%s#",
                           owner_->module_name_);
  });

  // Calculate total size for the command buffer allocations that are
  // VA-remapped (small copy-shadow slices are excluded from the reservation),
  // rounding each allocation up to the allocation granularity.
  uint64_t total_va_size = 0;
  for (BufferAllocation::Index i :
       owner_->command_buffer_va_remapped_allocation_indexes_) {
    const uint64_t size = owning_buffer_allocations.GetDeviceAddress(i).size();
    if (IsSmallCopySlice(size, copy_threshold)) {
      continue;
    }
    total_va_size += RoundUpToGranularity(size, granularity);
  }

  ASSIGN_OR_RETURN(va_ranges_->unmap_event, executor->CreateEvent());

  // Reserve a single large VA range for the remapped command buffer
  // allocations. Skip the reservation entirely if everything is copy-shadow.
  if (total_va_size > 0) {
    ASSIGN_OR_RETURN(va_ranges_->va_reservation,
                     vmm_allocator->CreateReservation(executor, total_va_size));
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: Reserved single VA range for module %s "
        "VA: %p total_size: %d granularity: %d",
        owner_->module_name_, va_ranges_->va_reservation->address().opaque(),
        total_va_size, granularity);
  }
  return absl::OkStatus();
}

absl::flat_hash_map<BufferAllocation::Index, uint64_t>
GpuExecutableBufferAllocator::ExecutionScope::ComputeReservationOffsets(
    const BufferAllocations& owning_buffer_allocations, uint64_t granularity,
    uint64_t copy_threshold) const {
  // Build a map from allocation index to its offset within va_reservation.
  // Iterate through the VA-remapped set in order (btree_set provides
  // deterministic iteration order) and accumulate offsets.
  absl::flat_hash_map<BufferAllocation::Index, uint64_t> allocation_va_offsets;
  uint64_t current_offset = 0;
  for (BufferAllocation::Index idx :
       owner_->command_buffer_va_remapped_allocation_indexes_) {
    const uint64_t size =
        owning_buffer_allocations.GetDeviceAddress(idx).size();
    // Small copy-shadow slices are not part of the contiguous VA reservation;
    // keep them out so offsets stay aligned with the mapping descriptors.
    if (IsSmallCopySlice(size, copy_threshold)) {
      continue;
    }
    allocation_va_offsets[idx] = current_offset;
    current_offset += RoundUpToGranularity(size, granularity);
  }
  return allocation_va_offsets;
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::GetOrCreateSmallShadow(
    BufferAllocation::Index i, uint64_t size, se::StreamExecutor* executor) {
  auto it = va_ranges_->small_shadow.find(i);
  if (it == va_ranges_->small_shadow.end()) {
    // First use of this slice: allocate a stable shadow buffer in the SAME
    // memory space (color) as the original. Most small slices are default-space
    // (color 0) scale/scalar/metric buffers, but a non-zero color (e.g.
    // collective memory) must not be silently placed in space 0. An
    // out-of-range index would indicate an allocation-tracking bug, so fail
    // loudly rather than guess.
    if (i >=
        static_cast<BufferAllocation::Index>(owner_->allocations_.size())) {
      return Internal(
          "Command-buffer slice index %d out of range (%d allocations)", i,
          static_cast<int64_t>(owner_->allocations_.size()));
    }
    const int64_t memory_space =
        static_cast<int64_t>(owner_->allocations_[i]->color());
    se::DeviceAddressBase shadow = executor->Allocate(size, memory_space);
    if (shadow.is_null()) {
      return Internal(
          "Failed to allocate %d-byte shadow for small command-buffer slice %d",
          size, i);
    }
    it = va_ranges_->small_shadow.insert_or_assign(i, shadow).first;
  } else if (it->second.size() < size) {
    // The shadow address was baked into the command buffer when it was captured
    // on the first step; under trace-once replay (NEVER_UPDATE) the command
    // buffer is never re-captured, so the shadow cannot be moved. A grown slice
    // means the shape changed, which this path does not support -- fail clearly
    // instead of reallocating to a new address that the replayed command buffer
    // would not see (silent corruption). Static shapes never hit this.
    return Internal(
        "Small command-buffer slice %d grew from %d to %d bytes; "
        "copy-into-shadow requires static shapes under trace-once replay",
        i, static_cast<int64_t>(it->second.size()), size);
  }
  return se::DeviceAddressBase(it->second.opaque(), size);
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::BuildStepMappingPlan(
    const BufferAllocations& owning_buffer_allocations,
    se::StreamExecutor* executor, se::DeviceAddressVmmAllocator* vmm_allocator,
    uint64_t granularity, uint64_t copy_threshold,
    const absl::flat_hash_map<BufferAllocation::Index, uint64_t>&
        allocation_va_offsets,
    int device_ordinal, StepMappingPlan& plan) {
  plan.mapped_buffers.reserve(owning_buffer_allocations.size());

  // Descriptors are accumulated in reservation_offset order (guaranteed because
  // allocation_va_offsets was built from a sorted btree_set and the loop below
  // iterates allocation indices in ascending order).
  const BufferAllocation::Index num_allocations =
      static_cast<BufferAllocation::Index>(owning_buffer_allocations.size());
  for (BufferAllocation::Index i = 0; i < num_allocations; ++i) {
    se::DeviceAddressBase original_buffer =
        owning_buffer_allocations.GetDeviceAddress(i);

    // Only do VA mapping for allocations accessed by CommandBufferThunk.
    auto offset_it = allocation_va_offsets.find(i);
    if (offset_it == allocation_va_offsets.end()) {
      // Small command-buffer slices bypass VA remapping: bake a stable shadow
      // address into the command buffer and refresh it with a D2D copy each
      // step instead of an unmap/map/SetAccess round-trip.
      if (IsSmallCopySlice(original_buffer.size(), copy_threshold) &&
          owner_->command_buffer_va_remapped_allocation_indexes_.contains(i)) {
        if (original_buffer.is_null()) {
          return Internal("Command buffer allocation %d has null address", i);
        }
        const uint64_t sz = original_buffer.size();
        ASSIGN_OR_RETURN(se::DeviceAddressBase shadow_va,
                         GetOrCreateSmallShadow(i, sz, executor));
        plan.small_copies.push_back({shadow_va, original_buffer, sz});
        plan.mapped_buffers.push_back(shadow_va);
        continue;
      }
      // Not a command buffer allocation (or zero-size), use the original
      // buffer.
      plan.mapped_buffers.push_back(original_buffer);
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

    plan.mapping_descriptors.push_back(
        {va_offset, /*allocation_offset=*/0, mapping_size, raw_alloc});
    plan.new_src_addrs.push_back(original_buffer.opaque());

    // Use VA address for execution.
    plan.mapped_buffers.push_back(
        se::DeviceAddressBase(sub_range_va.opaque(), original_buffer.size()));
  }
  return absl::OkStatus();
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::EstablishMapping(
    StepMappingPlan& plan, bool skip_remap_enabled, int device_ordinal) {
  // Decide how to (re)establish the VA->physical mapping for this step.
  const bool can_skip_remap =
      skip_remap_enabled && va_ranges_->scoped_mapping.has_value() &&
      !plan.mapping_descriptors.empty() &&
      va_ranges_->last_mapped_src_addrs.size() == plan.new_src_addrs.size();

  if (can_skip_remap) {
    // ROCm skip-remap: only slices whose backing source buffer moved since the
    // previous step are unmapped+remapped; unchanged slices keep their mapping
    // (no driver calls). SetAccess is issued once over the full range only if
    // something changed (handled inside Remap()).
    std::vector<se::MemoryReservation::RemappingDescriptor> remaps;
    remaps.reserve(plan.mapping_descriptors.size());
    int changed = 0;
    for (size_t k = 0; k < plan.mapping_descriptors.size(); ++k) {
      const se::MemoryReservation::MappingDescriptor& md =
          plan.mapping_descriptors[k];
      // The per-slot address comparison below is only valid if slot k maps the
      // SAME allocation as last step. Under NEVER_UPDATE / static shapes the
      // index->reservation-offset assignment is fixed, so verify that invariant
      // (debug-only): a count-preserving reshuffle would otherwise compare
      // addresses of two different allocations.
      DCHECK(va_ranges_->last_mapped_offsets.empty() ||
             va_ranges_->last_mapped_offsets[k] == md.reservation_offset)
          << "skip-remap slot " << k
          << " allocation identity changed across steps (reservation offset "
          << va_ranges_->last_mapped_offsets[k] << " -> "
          << md.reservation_offset << ")";
      const bool remap_required =
          plan.new_src_addrs[k] != va_ranges_->last_mapped_src_addrs[k];
      changed += remap_required ? 1 : 0;
      remaps.push_back({md.reservation_offset, md.allocation_offset, md.size,
                        md.allocation, remap_required});
    }

    XLA_VLOG_DEVICE(2, device_ordinal) << absl::StreamFormat(
        "VA remapping(skip): module %s slots=%d changed=%d",
        owner_->module_name_, remaps.size(), changed);

    if (changed > 0) {
      // At least one slot moved: wait for the previous step's GPU work to
      // finish using the VA range before unmapping the changed slices.
      RETURN_IF_ERROR(va_ranges_->unmap_event->Synchronize());
      // Remap() consumes the current mapping via std::move. On failure we must
      // clear scoped_mapping: otherwise it still has_value() but holds a
      // moved-from (empty) ScopedMapping, and the next step's skip-remap path
      // would call Remap() on it again (UB). Resetting forces the legacy
      // reset()+MapTo() recovery path next step instead.
      absl::StatusOr<se::MemoryReservation::ScopedMapping> remapped =
          std::move(*va_ranges_->scoped_mapping).Remap(absl::MakeSpan(remaps));
      if (!remapped.ok()) {
        va_ranges_->scoped_mapping.reset();
        return remapped.status();
      }
      va_ranges_->scoped_mapping = std::move(*remapped);
    }
    // changed == 0: steady state. Keep the existing mapping untouched; no
    // unmap, no map, no SetAccess, and no unmap-event sync.
  } else {
    // Legacy / first-step path (also the unchanged non-ROCm behavior): wait for
    // any prior GPU use of the VA range, drop the old mapping, then map the
    // full contiguous range in a single MapTo call.
    if (va_ranges_->scoped_mapping.has_value()) {
      RETURN_IF_ERROR(va_ranges_->unmap_event->Synchronize());
      va_ranges_->scoped_mapping.reset();
    }
    if (!plan.mapping_descriptors.empty()) {
      ASSIGN_OR_RETURN(se::MemoryReservation::ScopedMapping scoped_mapping,
                       va_ranges_->va_reservation->MapTo(
                           absl::MakeSpan(plan.mapping_descriptors)));
      va_ranges_->scoped_mapping = std::move(scoped_mapping);
    }
  }

  va_ranges_->last_mapped_src_addrs = std::move(plan.new_src_addrs);
  // Track per-slot reservation offsets so the next step can DCHECK the
  // slot->allocation mapping stayed stable (see the skip-remap loop above).
  // Always maintained (not #ifndef NDEBUG-guarded): measured at ~10-150 ns per
  // step for typical slot counts and N*8 bytes of RAM -- negligible next to the
  // microsecond-to-millisecond hipMemUnmap/hipMemMap it sits beside.
  va_ranges_->last_mapped_offsets.clear();
  va_ranges_->last_mapped_offsets.reserve(plan.mapping_descriptors.size());
  for (const auto& md : plan.mapping_descriptors) {
    va_ranges_->last_mapped_offsets.push_back(md.reservation_offset);
  }
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
          // With CAPTURE_CMD_NEVER_UPDATE only traced commands bind buffers at
          // trace time; other update modes remap every command's allocations.
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
              // Constants live at globally stable addresses: persistent, but
              // never VA-remapped.
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
  auto indexes_or = CollectCommandBufferAllocationIndexes(
      thunk_executor, allocations_, update_mode_);
  CHECK_OK(indexes_or.status());
  CommandBufferAllocationIndexes indexes = std::move(indexes_or).value();
  command_buffer_persistent_allocation_indexes_ = std::move(indexes.persistent);
  command_buffer_persistent_alloc_indices_.assign(
      command_buffer_persistent_allocation_indexes_.begin(),
      command_buffer_persistent_allocation_indexes_.end());
  command_buffer_va_remapped_allocation_indexes_ =
      std::move(indexes.va_remapped);
  VLOG(3) << "VA remapping: collected "
          << command_buffer_persistent_allocation_indexes_.size()
          << " persistent and "
          << command_buffer_va_remapped_allocation_indexes_.size()
          << " VA-remapped allocation indexes for module " << module_name_;
}

absl::StatusOr<GpuExecutableBufferAllocator::ExecutionScope>
GpuExecutableBufferAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  if (command_buffer_va_remapped_allocation_indexes_.empty() ||
      update_mode_ == DebugOptions::ALWAYS_UPDATE ||
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator) ==
          nullptr) {
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "CreateExecutionScope: va_remapped_allocation_indexes.size()=%d "
        "use_command_buffer_va_remapping=0",
        command_buffer_va_remapped_allocation_indexes_.size());
    return ExecutionScope(this);
  }

  se::StreamExecutor* executor = run_options->stream()->parent();
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "CreateExecutionScope: va_remapped_allocation_indexes.size()=%d "
      "use_command_buffer_va_remapping=1",
      command_buffer_va_remapped_allocation_indexes_.size());

  VaRanges* va_ranges = nullptr;
  {
    absl::MutexLock lock(va_ranges_mutex_);
    va_ranges = &module_va_ranges_[executor];
    // Record the owning executor so the VaRanges destructor can free any
    // copy-into-shadow buffers it allocates (idempotent; same key every time).
    va_ranges->executor = executor;
  }
  return ExecutionScope(this, va_ranges, run_options);
}

}  // namespace gpu
}  // namespace xla
