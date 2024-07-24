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
// * async_copy must be an async copy-start instruction.
MemoryTransferDirection GetAsyncCopyDirection(const HloInstruction* async_copy,
                                              int64_t alternate_memory_space);

// This struct is used to track the outstanding async copy instructions and
// the remaining bytes required to be accessed.
struct OutstandingAsyncCopy {
  const HloInstruction* copy_start_inst;
  float remaining_bytes_to_transfer;
  bool operator==(const OutstandingAsyncCopy& other) const {
    return copy_start_inst == other.copy_start_inst &&
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
      const std::list<OutstandingAsyncCopy>& outstanding_read_default_queue,
      const std::list<OutstandingAsyncCopy>& outstanding_write_default_queue)
      : cost_analysis_(cost_analysis),
        alternate_memory_space_(alternate_memory_space),
        outstanding_read_default_queue_(outstanding_read_default_queue),
        outstanding_write_default_queue_(outstanding_write_default_queue) {}

  ~RuntimeSimulator() = default;

  // This function is used to predict the effectiveness of the memory space
  // assignment solution. Specifically, it returns the estimated execution time
  // (in seconds) of the HLO module for the given memory space assignment (i.e.,
  // ```allocations```).
  float ComputeEstimatedElapsedTime(const HloLiveRange& hlo_live_range,
                                    const AllocationSequence& allocations);

  // This is an auxiliary function for simulating the execution
  // time for executing a copy-done instruction. It returns the
  // elapsed time (in seconds) for executing the copy-done instruction.
  //
  // This function also updates the passed in queues as we complete async copies
  // during the simulation.
  //
  // We simulate the shared bandwidth for default-alternate memory access.
  // For example, if the copy-done instruction is a default-write memory
  // process, and there are outstanding default-read memory processes in the
  // outstanding_read_default_queue, then we use half of the bandwidth to
  // process both requests in parallel. Otherwise, we use the full bandwidth to
  // process the default-write request.
  float SimulateAsyncCopyDone(const HloInstruction* copy_done_instruction);

  const std::list<OutstandingAsyncCopy>& GetOutstandingReadDefaultQueue() const;

  const std::list<OutstandingAsyncCopy>& GetOutstandingWriteDefaultQueue()
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
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;

  // This function updates the queue by updating the front request with the
  // processed bytes. If the request is completed (no remaining bytes to
  // process), the function returns the instruction and pop it from the queue.
  // Otherwise, it returns nullptr.
  const HloInstruction* RemoveBytesFromQueueIfNotEmpty(
      std::list<OutstandingAsyncCopy>& async_copy_queue, float processed_bytes);

  // This is an auxiliary function which simulates the process of draining
  // the memory access queues in a given amount of time (seconds). If both
  // outstanding_*_default_queues are non-empty, they share bandwidth. If one of
  // the queues is empty and the other is not, it gets the full bandwdith.
  void ProcessAsyncCopiesInIdleTime(float time);
  // Members used for memory model simulation
  int64_t alternate_memory_space_;
  std::list<OutstandingAsyncCopy> outstanding_read_default_queue_;
  std::list<OutstandingAsyncCopy> outstanding_write_default_queue_;
};

}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
