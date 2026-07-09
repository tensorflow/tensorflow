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
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace stream_executor {
class DeviceAddressVmmAllocator;
}  // namespace stream_executor

namespace xla {
namespace gpu {

class ThunkExecutor;

// Owns executable-scoped buffer allocation state for one GpuExecutable.
class GpuExecutableBufferAllocator {
 private:
  struct Remapping;

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

  // Per-run buffer allocation context created by `CreateExecutionScope`.
  // Callers first use it to build `BufferAllocations` from runtime parameters,
  // constants, temporary buffers, and output buffers, then use it to run the
  // executable with those allocations.
  //
  // The scope can provide an allocation-address policy even when VMM remapping
  // is inactive, for example for global constants.
  //
  // When command-buffer VA remapping is available, the scope also holds the
  // lock for the executable/executor remapping state. Selected command-buffer
  // allocations are backed by physical VMM allocations while execution sees
  // stable reserved VA addresses. Command-buffer VA remapping is inactive when
  // `va_remap_enabled()` is false.
  class ExecutionScope {
   public:
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    ExecutionScope(ExecutionScope&&) = default;
    ExecutionScope& operator=(ExecutionScope&&) = default;

    bool va_remap_enabled() const { return remapping_ != nullptr; }

    // Builds the BufferAllocations for an execution. Entry-computation
    // parameter buffers are obtained from `get_parameter_buffer`; all other
    // allocations are resolved internally, including collective-memory
    // granularity rounding, alignment checking, and command-buffer VA remapping
    // when enabled for this execution.
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
        absl::FunctionRef<absl::Status(
            const BufferAllocations&,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices)>
            execute);

   private:
    friend class GpuExecutableBufferAllocator;

    ExecutionScope(GpuExecutableBufferAllocator* owner, Remapping* remapping,
                   se::DeviceAddressVmmAllocator* vmm_allocator,
                   std::unique_ptr<absl::MutexLock> remap_lock);

    absl::Status PrepareReservation(
        const ServiceExecutableRunOptions* run_options, int device_ordinal,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    bool ShouldRemapAllocation(BufferAllocation::Index index) const;
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> AllocateBuffer(
        int device_ordinal, const BufferAllocation& allocation,
        int64_t buffer_size);
    absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        const BufferAllocation& allocation,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
        int64_t arg_idx,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    GpuExecutableBufferAllocator* owner_ = nullptr;
    Remapping* remapping_ = nullptr;
    se::DeviceAddressVmmAllocator* vmm_allocator_ = nullptr;
    std::unique_ptr<absl::MutexLock> remap_lock_;
  };

  GpuExecutableBufferAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      ThunkExecutor* thunk_executor);
  ~GpuExecutableBufferAllocator();

  size_t command_buffer_allocation_count() const {
    return persistent_alloc_indices_.size();
  }

  absl::StatusOr<ExecutionScope> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

 private:
  struct Remapping {
    absl::Mutex mutex;
    uint64_t granularity = 0;
    uint64_t total_size = 0;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_reservation_offset;
    std::unique_ptr<se::MemoryReservation> va_reservation;
    se::DeviceAddressVmmAllocator* vmm_allocator = nullptr;

    absl::StatusOr<uint64_t> GetReservationOffset(
        BufferAllocation::Index idx) const;
  };

  std::string module_name_;
  std::vector<const BufferAllocation*> allocations_;
  Shape result_shape_;
  const DebugOptions* debug_options_ = nullptr;

  // Sorted indices of command-buffer-referenced constant allocations. Their
  // global addresses are stable without VMM remapping.
  std::vector<BufferAllocation::Index> constant_alloc_indices_;

  // Indices of command-buffer-referenced temporary allocations assigned stable
  // addresses through VMM remapping.
  AllocationIndexSet va_remapped_alloc_indices_;

  // Sorted union of constant_alloc_indices_ and va_remapped_alloc_indices_.
  std::vector<BufferAllocation::Index> persistent_alloc_indices_;

  absl::Mutex remappings_mutex_;
  absl::node_hash_map<se::StreamExecutor*, Remapping> remappings_
      ABSL_GUARDED_BY(remappings_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
