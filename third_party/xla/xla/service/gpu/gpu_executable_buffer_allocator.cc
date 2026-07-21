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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_va_remap_allocator.h"
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

}  // namespace

std::unique_ptr<GpuExecutableBufferAllocator>
GpuExecutableBufferAllocator::Create(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor) {
  const DebugOptions::CommandBufferUpdateMode update_mode =
      debug_options != nullptr
          ? debug_options->xla_gpu_command_buffer_update_mode()
          : DebugOptions::ALWAYS_UPDATE;
  switch (update_mode) {
    case DebugOptions::ALWAYS_UPDATE:
      return std::make_unique<GpuExecutableBufferAllocator>(
          module_name, allocations, result_shape, debug_options,
          thunk_executor);
    case DebugOptions::SKIP_TEMP:
      return std::make_unique<GpuExecutableVaRemapAllocator>(
          module_name, allocations, result_shape, debug_options,
          thunk_executor);
    default:
      LOG(FATAL) << "Unsupported command buffer update mode: " << update_mode;
  }
}

void GpuExecutableBufferAllocator::ForEachCommandBufferAllocation(
    absl::Span<const BufferAllocation* const> allocations,
    const ThunkExecutor* thunk_executor,
    absl::FunctionRef<void(BufferAllocation::Index, const BufferAllocation&)>
        callback) {
  if (thunk_executor == nullptr) {
    return;
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
          callback(index, allocation);
        }
        return absl::OkStatus();
      }));
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
  AllocationIndexSet constants;
  ForEachCommandBufferAllocation(
      allocations_, thunk_executor,
      [&](BufferAllocation::Index index, const BufferAllocation& allocation) {
        if (allocation.is_constant()) {
          constants.insert(index);
        }
      });
  constant_alloc_indices_.assign(constants.begin(), constants.end());
  persistent_alloc_indices_ = constant_alloc_indices_;

  VLOG(3) << "Command buffer allocation policy: collected "
          << constant_alloc_indices_.size()
          << " constant allocation indices for module " << module_name_;
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::AllocateTransientBuffer(
    int device_ordinal, const BufferAllocation& allocation, int64_t buffer_size,
    se::DeviceAddressAllocator* memory_allocator) {
  ASSIGN_OR_RETURN(
      se::ScopedDeviceAddress<uint8_t> buffer,
      memory_allocator->Allocate(device_ordinal, buffer_size,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/allocation.color()));
  return buffer.Release();
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
    ASSIGN_OR_RETURN(buffer_address,
                     AllocateTransientBuffer(device_ordinal, allocation,
                                             buffer_size, memory_allocator));
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
  RETURN_IF_ERROR(Prepare(run_options, device_ordinal));

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
  return execute(owning_buffer_allocations,
                 absl::MakeConstSpan(owner_->constant_alloc_indices_));
}

absl::StatusOr<std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope>>
GpuExecutableBufferAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  return std::unique_ptr<ExecutionScope>(new ExecutionScope(this));
}

}  // namespace xla::gpu
