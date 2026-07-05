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
  struct MemoryReservationAlias;
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

  struct CommandBufferAllocationIndexes {
    // Allocation indices with stable command-buffer-visible addresses.
    AllocationIndexSet persistent;

    // Allocation indices that need VMM-backed stable addresses for
    // command-buffer execution.
    AllocationIndexSet va_remapped;
  };

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
  // `command_buffer_active()` is false.
  class ExecutionScope {
   public:
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    ExecutionScope(ExecutionScope&&) = default;
    ExecutionScope& operator=(ExecutionScope&&) = default;

    bool command_buffer_active() const { return remapping_ != nullptr; }
    bool address_policy_active() const { return address_policy_active_; }

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

    ExecutionScope() = default;
    ExecutionScope(GpuExecutableBufferAllocator* owner, Remapping* remapping,
                   se::DeviceAddressVmmAllocator* vmm_allocator,
                   std::unique_ptr<absl::MutexLock> remap_lock,
                   bool address_policy_active);

    absl::Status PrepareReservation(
        const ServiceExecutableRunOptions* run_options, int device_ordinal,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    bool ShouldRemapAllocation(BufferAllocation::Index index) const;
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> AllocateBuffer(
        int device_ordinal, const BufferAllocation& allocation,
        int64_t buffer_size, bool return_reservation_address);
    absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        const BufferAllocation& allocation,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
        int64_t arg_idx,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    absl::Status UpdateAllocationAddressPolicy();
    std::optional<absl::Span<const BufferAllocation::Index>>
    GetPersistentAllocIndices() const;
    absl::StatusOr<BufferAllocations> BuildExecutionBufferAllocations(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal);
    absl::Status UnmapAliases(int device_ordinal);
    absl::StatusOr<MemoryReservationAlias> GetReservationAlias(
        BufferAllocation::Index idx) const;

    GpuExecutableBufferAllocator* owner_ = nullptr;
    Remapping* remapping_ = nullptr;
    se::DeviceAddressVmmAllocator* vmm_allocator_ = nullptr;
    std::unique_ptr<absl::MutexLock> remap_lock_;
    bool address_policy_active_ = false;
    absl::flat_hash_map<BufferAllocation::Index, MemoryReservationAlias>
        allocation_to_reservation_aliases_;
    std::vector<MemoryReservationAlias> aliases_to_unmap_;
  };

  static absl::StatusOr<CommandBufferAllocationIndexes>
  CollectCommandBufferAllocationIndexes(
      ThunkExecutor* thunk_executor,
      absl::Span<const BufferAllocation* const> allocations,
      DebugOptions::CommandBufferUpdateMode update_mode);

  GpuExecutableBufferAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      ThunkExecutor* thunk_executor,
      AllocationIndexSet returned_output_allocation_indexes);
  ~GpuExecutableBufferAllocator();

  size_t command_buffer_allocation_count() const {
    return command_buffer_persistent_allocation_indexes_.size();
  }

  const AllocationIndexSet& command_buffer_allocation_indexes() const {
    return command_buffer_persistent_allocation_indexes_;
  }

  absl::StatusOr<ExecutionScope> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

 private:
  struct MemoryReservationAlias {
    uint64_t reservation_offset = 0;
    uint64_t size = 0;
    se::DeviceAddressBase reservation_address;
  };

  struct Remapping {
    absl::Mutex mutex;
    uint64_t granularity = 0;
    uint64_t total_size = 0;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_reservation_offset;
    std::unique_ptr<se::MemoryReservation> va_reservation;
    se::DeviceAddressVmmAllocator* vmm_allocator = nullptr;
    std::optional<std::vector<BufferAllocation::Index>>
        policy_persistent_alloc_indices;
    std::optional<AllocationIndexSet> policy_va_remapped_index_set;

    absl::StatusOr<uint64_t> GetReservationOffset(
        BufferAllocation::Index idx) const;
  };

  std::string module_name_;
  std::vector<const BufferAllocation*> allocations_;
  Shape result_shape_;
  const DebugOptions* debug_options_ = nullptr;
  DebugOptions::CommandBufferUpdateMode update_mode_;
  AllocationIndexSet returned_output_allocation_indexes_;
  AllocationIndexSet command_buffer_persistent_allocation_indexes_;
  std::vector<BufferAllocation::Index> command_buffer_persistent_alloc_indices_;
  AllocationIndexSet command_buffer_va_remapped_allocation_indexes_;

  absl::Mutex remappings_mutex_;
  absl::node_hash_map<se::StreamExecutor*, Remapping> remappings_
      ABSL_GUARDED_BY(remappings_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
