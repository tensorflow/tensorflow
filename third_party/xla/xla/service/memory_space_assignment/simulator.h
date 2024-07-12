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

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"

namespace xla {
namespace memory_space_assignment {

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis)
      : cost_analysis_(cost_analysis) {}
  virtual ~RuntimeSimulator() = default;
  // This function is used to predict the effectiveness of the memory space
  // assignment solution. Specifically, it returns the estimated execution time
  // (in seconds) of the HLO module for the given memory space assignment (i.e.,
  // ```allocations```).
  // This function provides a basic estimate without considering the effect of
  // async copies.
  float SimulateElapsedTimeWithoutAsyncCopies(
      const HloLiveRange& hlo_live_range,
      const AllocationSequence& allocations);

  // TODO(b/352777140): The SimulateElapsedTimeWithoutAsyncCopies function
  // assumes that the async copies do not have any overhead.
  // However, in practice, the computation and the async copies may not overlap
  // perfectly, and the overhead of the async copies is an important factor to
  // evaluate the effectiveness of the memory space assignment solution. Thus,
  // we need to add a function to simulate the execution time with async copies.
  // The most important feature of the memory model is the bandwidth sharing, in
  // which the read-default-memory requests and write-default-memory requests
  // share the bandwidth. To implement this feature, we need to maintain two
  // queues for the read/write requests, and track whether we should share the
  // bandwidth to process the memory requests.

  // This is an auxiliary function which is used for simulating the execution
  // time of async copies. This function simulates the execution time of
  // transferring ```bytes_to_transfer``` bytes while sharing the bandwidth with
  // memory access requests in ```memory_access_queue_to_share_bandwidth```. The
  // bandwidth is shared equally: When the
  // memory_access_queue_to_share_bandwidth is not empty, we can only use half
  // of the bandwidth to transfer the request, and use the other half to
  // transfer the memory requests in the queue. When the queue is drained, we
  // can use the full bandwidth to transfer the request.
  static float SimulateAsyncCopyTransfer(
      float bytes_to_transfer,
      std::queue<const HloInstruction*>& memory_access_queue_to_share_bandwidth,
      absl::flat_hash_map<const HloInstruction*, float>&
          remaining_size_of_buffers,
      float default_memory_bytes_per_second);

  // This is an auxiliary function which simulates the process of draining the
  // memory access queue in a given time window. There are two queues which will
  // share the bandwidth: ```read_queue``` and ```write_queue``` which track the
  // memory access requests that read/write the default memory. When both of the
  // queues are not empty, the front requests from both queues equally share the
  // bandwidth. When one of the queue is empty, the other queue can use the full
  // bandwidth.

  static void ProcessAsyncCopyInTimeWindow(
      float time_windows, std::queue<const HloInstruction*>& read_queue,
      std::queue<const HloInstruction*>& write_queue,
      absl::flat_hash_map<const HloInstruction*, float>&
          remaining_size_of_buffers,
      float default_memory_bytes_per_second);

 private:
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
};
}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
