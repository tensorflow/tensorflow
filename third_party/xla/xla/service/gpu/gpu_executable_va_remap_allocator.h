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
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
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

// GpuExecutableBufferAllocator for the SKIP_TEMP and SKIP_PROFILED command
// buffer update modes: command buffer VA remapping.
//
// When VA remapping is available for an execution, selected command-buffer
// allocations are backed by physical VMM allocations while execution sees
// stable reserved VA addresses, so recorded command buffers can skip updating
// them:
//  - SKIP_TEMP remaps the command-buffer-referenced preallocated temp buffer
//    allocations, selected at construction time.
//  - SKIP_PROFILED automatically selects command-buffer-referenced
//    preallocated temp buffers, profiles the address stability of the other
//    non-constant, non-thread-local allocations over the first executions,
//    and then remaps the union of the temp buffers and stable profile
//    candidates.
//
// When VA remapping is unavailable for an execution (no VMM allocator,
// nothing to remap, or SKIP_PROFILED has neither automatically selected temp
// buffers nor stable profile candidates), executions fall back to the
// base-class behavior.
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

  // Test-only: adds `index` to the SKIP_PROFILED profile candidate set, as if
  // it had been collected from a command buffer thunk at construction time.
  void AddProfileCandidateAllocationForTesting(BufferAllocation::Index index);

 private:
  // Per-executor VA remapping state, owned by the allocator and shared by the
  // execution scopes created for that executor.
  struct Remapping {
    // SKIP_PROFILED per-executor profile state machine. The phase progresses
    // kProfiling -> (kActive | kDisabled) exactly once, which keeps the
    // persistent allocation indices passed to thunks consistent with the
    // one-way absent-to-present transition required by
    // Thunk::ExecuteParams::persistent_alloc_indices.
    enum class ProfilePhase {
      // Not a SKIP_PROFILED remapping (SKIP_TEMP).
      kInactive,
      // Observing allocation addresses; executions pass std::nullopt.
      kProfiling,
      // Profile transition done; the selected allocation set is VA-remapped.
      kActive,
      // No automatic or profiled allocations; executions pass only constants.
      kDisabled,
    };

    absl::Mutex mutex;
    uint64_t granularity = 0;
    uint64_t total_size = 0;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_reservation_offset;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_mapping_size;
    std::unique_ptr<se::MemoryReservation> va_reservation;
    se::DeviceAddressVmmAllocator* vmm_allocator = nullptr;

    ProfilePhase phase = ProfilePhase::kInactive;
    // Number of completed profiling observations.
    int64_t profiled_steps = 0;
    // Address of each profile candidate observed on the previous execution.
    absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>
        last_observed_address;
    // Candidates disqualified from remapping because their address changed
    // between executions or was null.
    absl::flat_hash_set<BufferAllocation::Index> unstable_alloc_indices;
    // Automatic temp buffers and profile candidates selected for VA remapping
    // by the profile transition.
    AllocationIndexSet profiled_va_remapped_alloc_indices;
    // Sorted union of constant allocation indices and
    // profiled_va_remapped_alloc_indices, set by the profile transition.
    std::vector<BufferAllocation::Index> profiled_persistent_alloc_indices;

    absl::StatusOr<uint64_t> GetReservationOffset(
        BufferAllocation::Index idx) const;
    absl::StatusOr<uint64_t> GetMappingSize(BufferAllocation::Index idx) const;
  };

  // The VA-remapping ExecutionScope, defined in the .cc file.
  class VaRemapExecutionScope;

  // Runs the SKIP_PROFILED transition for one executor: combines automatic
  // temp buffers with profiled candidates whose addresses stayed stable,
  // filters parameter allocations that cannot be Map()ed (not backed by
  // `vmm_allocator` or sharing an address with another parameter), and moves
  // the phase to kActive, or to kDisabled when nothing can be remapped.
  void TransitionProfiledRemapping(Remapping* remapping,
                                   se::DeviceAddressVmmAllocator* vmm_allocator,
                                   int device_ordinal);

  DebugOptions::CommandBufferUpdateMode update_mode_ = DebugOptions::SKIP_TEMP;

  // SKIP_TEMP and SKIP_PROFILED: indices of command-buffer-referenced
  // preallocated temp buffers selected for VMM remapping.
  AllocationIndexSet va_remapped_alloc_indices_;

  // SKIP_PROFILED: command-buffer-referenced non-constant, non-temp
  // allocations whose address stability is profiled during the first
  // executions.
  AllocationIndexSet profile_candidate_alloc_indices_;

  absl::Mutex remappings_mutex_;
  absl::node_hash_map<se::StreamExecutor*, Remapping> remappings_
      ABSL_GUARDED_BY(remappings_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_VA_REMAP_ALLOCATOR_H_
