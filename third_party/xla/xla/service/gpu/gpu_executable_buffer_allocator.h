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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class ThunkExecutor;

// Owns executable-scoped buffer allocation state for one GpuExecutable.
class GpuExecutableBufferAllocator {
 private:
  struct VaRanges;

 public:
  struct ParameterBuffer {
    se::DeviceAddressBase buffer;
    int64_t parameter_number = 0;
    bool allow_null_buffer = false;
  };

  // Resolves the device address backing an entry-computation-parameter
  // allocation. Returning `allow_null_buffer` is used for skipped tuple
  // index-table allocations.
  using ParameterBufferResolver =
      absl::FunctionRef<absl::StatusOr<ParameterBuffer>(
          const BufferAllocation& allocation)>;

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

  using AllocationIndexSet = absl::btree_set<BufferAllocation::Index>;

  // Execution-scoped buffer allocation state. Command-buffer VA remapping is
  // inactive when `command_buffer_active()` is false.
  class ExecutionScope {
   public:
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    ExecutionScope(ExecutionScope&&) = default;
    ExecutionScope& operator=(ExecutionScope&&) = default;

    bool command_buffer_active() const { return va_ranges_ != nullptr; }

    // Builds the BufferAllocations for an execution. Entry-computation
    // parameter buffers are obtained from `get_parameter_buffer`; all other
    // allocations are resolved internally, including collective-memory
    // granularity rounding and alignment checking.
    absl::StatusOr<BufferAllocations> GenerateBufferAllocations(
        const ServiceExecutableRunOptions* run_options,
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

    // Copy-protection for an aliased output that was not donated at runtime:
    // allocates a fresh result buffer for the output at `index`, copies the
    // contents of the aliased buffer (allocation `allocation`) into it, and
    // redirects the aliased entry in `buffer_allocations` to the fresh buffer.
    // Returns the newly allocated result buffer.
    absl::StatusOr<se::DeviceAddressBase> AllocateCopyProtectedOutputBuffer(
        const ServiceExecutableRunOptions* run_options,
        BufferAllocations& buffer_allocations, const ShapeIndex& index,
        const BufferAllocation& allocation, int device_ordinal,
        se::DeviceAddressAllocator* memory_allocator,
        absl::FunctionRef<absl::Status(absl::Status)> allocation_error);

    absl::Status ExecuteWithBufferAllocations(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(const BufferAllocations&)> execute);

   private:
    friend class GpuExecutableBufferAllocator;

    explicit ExecutionScope(GpuExecutableBufferAllocator* owner);
    ExecutionScope(GpuExecutableBufferAllocator* owner, VaRanges* va_ranges,
                   const ServiceExecutableRunOptions* run_options);

    absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        const BufferAllocation& allocation,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
        int64_t arg_idx,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    absl::Status ExecuteWithVaRemapping(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(const BufferAllocations&)> execute);

    GpuExecutableBufferAllocator* owner_ = nullptr;
    VaRanges* va_ranges_ = nullptr;
    const ServiceExecutableRunOptions* run_options_ = nullptr;
  };

  static absl::StatusOr<AllocationIndexSet>
  CollectCommandBufferAllocationIndexes(
      ThunkExecutor* thunk_executor,
      absl::Span<const BufferAllocation* const> allocations,
      DebugOptions::CommandBufferUpdateMode update_mode);

  GpuExecutableBufferAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      DebugOptions::CommandBufferUpdateMode update_mode,
      AllocationIndexSet allocation_indexes);

  size_t command_buffer_allocation_count() const {
    return command_buffer_allocation_indexes_.size();
  }

  absl::StatusOr<ExecutionScope> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

 private:
  // State for VA remapping of command buffer allocations on a single executor.
  struct VaRanges {
    // Mutex to protect VA range operations (map/execute/unmap) for this
    // executor. This ensures only one thread can use the VA ranges at a time.
    absl::Mutex mutex;

    // Single large virtual address reservation covering all command buffer
    // allocations. nullptr until first use.
    std::unique_ptr<se::MemoryReservation> va_reservation;

    // Event used to synchronize VA range reuse. When the device has completed
    // the task that uses the VA range, it marks the event, letting the host
    // know the VA range can be remapped to other physical addresses.
    std::unique_ptr<se::Event> unmap_event;

    // RAII wrapper that keeps the VA->physical mapping active.
    // Reset (auto-unmapping) before each re-use of the VA range.
    std::optional<se::MemoryReservation::ScopedMapping> scoped_mapping;
  };

  std::string module_name_;
  std::vector<const BufferAllocation*> allocations_;
  Shape result_shape_;
  const DebugOptions* debug_options_ = nullptr;
  DebugOptions::CommandBufferUpdateMode update_mode_;
  AllocationIndexSet command_buffer_allocation_indexes_;

  // Separate mutex for VA ranges to avoid contention with executable module
  // handle state during VA remapping operations, which may synchronize with GPU
  // work.
  absl::Mutex va_ranges_mutex_;
  absl::node_hash_map<se::StreamExecutor*, VaRanges> module_va_ranges_
      ABSL_GUARDED_BY(va_ranges_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
