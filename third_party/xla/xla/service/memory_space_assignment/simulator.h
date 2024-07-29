/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_

#include <cstdint>
#include <list>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape_util.h"

namespace xla {
namespace memory_space_assignment {

enum class MemoryTransferDirection {
  kUnsupported,
  kDefaultToAlternate,
  kAlternateToDefault,
};

// REQUIRES:
// * async_copy_like_start must be an async copy-start or slice-start
// instruction.
MemoryTransferDirection GetAsyncCopyLikeDirection(
    const HloInstruction* async_copy_like_start,
    int64_t alternate_memory_space);

// This struct is used to track the outstanding async copy like instructions and
// the remaining bytes required to be accessed.
struct OutstandingAsyncCopyLike {
  const HloInstruction* copy_like_start_inst;
  float remaining_bytes_to_transfer;
  bool operator==(const OutstandingAsyncCopyLike& other) const {
    return copy_like_start_inst == other.copy_like_start_inst &&
           remaining_bytes_to_transfer == other.remaining_bytes_to_transfer;
  }
};

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis,
                            int64_t alternate_memory_space)
      : cost_analysis_(cost_analysis),
        alternate_memory_space_(alternate_memory_space) {}

  // This constructor is used to inject the outstanding async copy queues for
  // testing purpose.
  explicit RuntimeSimulator(
      CostAnalysis* cost_analysis, int64_t alternate_memory_space,
      const std::list<OutstandingAsyncCopyLike>& outstanding_read_default_queue,
      const std::list<OutstandingAsyncCopyLike>&
          outstanding_write_default_queue)
      : cost_analysis_(cost_analysis),
        alternate_memory_space_(alternate_memory_space),
        outstanding_read_default_queue_(outstanding_read_default_queue),
        outstanding_write_default_queue_(outstanding_write_default_queue) {}

  ~RuntimeSimulator() = default;

  // This function provides a basic estimate without considering the overhead of
  // async copies.
  float SimulateElapsedTimeWithoutAsyncCopyLikes(
      const HloLiveRange& hlo_live_range,
      const AllocationSequence& allocations);

  // Returns the time to simulate the hlo_live_range, when we account for the
  // waiting time for async copy like instructions to finish.
  //
  // To simulate the overhead of async copy like instructions, we need to
  // maintain two queues to track the outstanding memory access requests that
  // read/write the default memory. When we simulate compute, we use any time
  // there is spare bandwidth to simulate async memory accesses to default
  // memory. If we get to an async copy like done, we must wait until it
  // finishes (potentially waiting for copies issued before it to finish.
  float SimulateElapsedTime(const HloModule* hlo_module,
                            const AllocationSequence& allocations);

  // This is an auxiliary function for simulating the execution
  // time for executing a copy-done instruction. It returns the
  // elapsed time (in seconds) for executing the copy-done instruction.
  //
  // This function also updates the passed in queues as we complete async copy
  // like instructions during the simulation.
  //
  // We simulate the shared bandwidth for default-alternate memory access.
  // For example, if the copy-done instruction is a default-write memory
  // process, and there are outstanding default-read memory processes in the
  // outstanding_read_default_queue, then we use half of the bandwidth to
  // process both requests in parallel. Otherwise, we use the full bandwidth to
  // process the default-write request.
  float SimulateAsyncCopyLikeDone(
      const HloInstruction* copy_like_done_instruction);

  const std::list<OutstandingAsyncCopyLike>& GetOutstandingReadDefaultQueue()
      const;

  const std::list<OutstandingAsyncCopyLike>& GetOutstandingWriteDefaultQueue()
      const;

  // This is an auxiliary function for simulating the execution
  // time for a compute instruction. It returns the elapsed time (in seconds)
  // for executing the compute instruction.
  //
  // Aside from returning the elapsed time, this function also updates the
  // outstanding memory request queues, by draining them when the compute
  // instruction is not occupying bandwidth.
  float SimulateComputeInstruction(
      const HloInstruction* compute_instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_memory,
      absl::Span<const ShapeIndex> outputs_in_alternate_memory);

 private:
  // This function parses the memory space assignment solution and initializes
  // the maps that record, for each instruction, which outputs and operands are
  // stored in alternate memory. These maps are used to estimate the runtime of
  // the HLO module.
  void InitializeAlternateMemoryMap(const AllocationSequence& allocations);
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
  // Members used for memory model simulation
  // This function updates the queue by updating the front request with the
  // processed bytes. If the request is completed (no remaining bytes to
  // process), the function returns the instruction and pop it from the queue.
  // Otherwise, it returns nullptr.
  const HloInstruction* RemoveBytesFromQueueIfNotEmpty(
      std::list<OutstandingAsyncCopyLike>& async_copy_like_queue,
      float processed_bytes);

  // This is an auxiliary function which simulates the process of draining
  // the memory access queues in a given amount of time (seconds). If both
  // outstanding_*_default_queues are non-empty, they share bandwidth. If one of
  // the queues is empty and the other is not, it gets the full bandwdith.
  void ProcessAsyncCopyLikesInIdleTime(float time);

  int64_t alternate_memory_space_;
  std::list<OutstandingAsyncCopyLike> outstanding_read_default_queue_;
  std::list<OutstandingAsyncCopyLike> outstanding_write_default_queue_;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      outputs_in_alternate_memory_map_;
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      operands_in_alternate_memory_map_;
};

}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
