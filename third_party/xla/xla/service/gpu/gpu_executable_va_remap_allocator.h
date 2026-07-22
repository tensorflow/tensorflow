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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_VA_REMAP_ALLOCATOR_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_VA_REMAP_ALLOCATOR_H_

#include <cstdint>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
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

// GpuExecutableBufferAllocator for the SKIP_TEMP command buffer update mode:
// command buffer VA remapping.
//
// When VA remapping is available for an execution, the
// command-buffer-referenced preallocated temp buffer allocations are backed
// by physical VMM allocations while execution sees stable reserved VA
// addresses, so recorded command buffers can skip updating them.
//
// When VA remapping is unavailable for an execution (no VMM allocator or
// nothing to remap), executions fall back to the base-class behavior.
class GpuExecutableVaRemapAllocator : public GpuExecutableBufferAllocator {
 public:
  GpuExecutableVaRemapAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      ThunkExecutor* thunk_executor);
  ~GpuExecutableVaRemapAllocator() override;

  // Creates the per-run execution scope. When VA remapping applies to this
  // execution, the returned scope holds the lock for the executable/executor
  // remapping state for the whole execution.
  absl::StatusOr<std::unique_ptr<ExecutionScope>> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator,
      int device_ordinal) override;

  // Test-only: adds `index` to the VA-remapped allocation set (and the
  // persistent set), as if it had been selected at construction time. Lets
  // unit tests exercise remapping of allocation kinds that no production
  // update mode selects yet.
  void AddVaRemappedAllocationForTesting(BufferAllocation::Index index);

 private:
  // Per-executor VA remapping state, owned by the allocator and shared by the
  // execution scopes created for that executor.
  struct Remapping {
    absl::Mutex mutex;
    uint64_t granularity = 0;
    uint64_t total_size = 0;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_reservation_offset;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_mapping_size;
    std::unique_ptr<se::MemoryReservation> va_reservation;
    se::DeviceAddressVmmAllocator* vmm_allocator = nullptr;

    absl::StatusOr<uint64_t> GetReservationOffset(
        BufferAllocation::Index idx) const;
    absl::StatusOr<uint64_t> GetMappingSize(BufferAllocation::Index idx) const;
  };

  // The VA-remapping ExecutionScope, defined in the .cc file.
  class VaRemapExecutionScope;

  // Indices of command-buffer-referenced temporary allocations assigned stable
  // addresses through VMM remapping.
  AllocationIndexSet va_remapped_alloc_indices_;

  absl::Mutex remappings_mutex_;
  absl::node_hash_map<se::StreamExecutor*, Remapping> remappings_
      ABSL_GUARDED_BY(remappings_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_VA_REMAP_ALLOCATOR_H_
