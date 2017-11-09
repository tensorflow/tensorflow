/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Allocator;
class AllocatorMemoryUsed;
class CostModelManager;
class Graph;
class NodeExecStats;
class StepStats;
class TrackingAllocator;

// Wraps NodeExecStats and adds allocation to it.
class NodeExecStatsWrapper {
 public:
  NodeExecStatsWrapper();
  // Owns 'stats'.
  NodeExecStatsWrapper(NodeExecStats* stats);

  // Destructor calls Finalize() to release the TrackingAllocators.
  ~NodeExecStatsWrapper() { Finalize(); }

  NodeExecStats* stats() { return stats_.get(); }

  // "Does not take ownership of the 'allocator'.
  // Transfers ownership of the 'tracking_allocator' to *this."
  void AddAllocation(Allocator* allocator,
                     TrackingAllocator* tracking_allocator);

 private:
  friend class StepStatsCollector;

  // Populates stats_ and releases TrackingAllocator.
  void Finalize();

  gtl::InlinedVector<std::pair<AllocatorMemoryUsed*, TrackingAllocator*>, 2>
      allocations_;
  std::unique_ptr<NodeExecStats> stats_;
};

// StepStatsCollector manages the collection of a StepStats object.
// The StepStats object holds multiple DeviceStats.
// Each DeviceStats object holds multiple NodeExecStats.
class StepStatsCollector {
 public:
  // Does not take ownership of `ss`.
  explicit StepStatsCollector(StepStats* ss);

  // BuildCostModel builds or updates a CostModel managed by cost_model_manager,
  // using the currently collected DeviceStats associated with the devices in
  // device_map.
  void BuildCostModel(
      CostModelManager* cost_model_manager,
      const std::unordered_map<string, const Graph*>& device_map);

  // Save saves nt to the DeviceStats object associated with device.
  // Should be called before Finalize.
  void Save(const string& device, NodeExecStats* nt);
  void Save(const string& device, NodeExecStatsWrapper* stats);

  // The following 2 Finalize methods populate the StepStats passed
  // from the constructor. Calling it more than once won't have any effect.
  // User shouldn't call Save() methods after Finalize.
  void Finalize();
  // swaps the content of StepStats* from constructor with 'ss'.
  void FinalizeAndSwap(StepStats* ss);

 private:
  void FinalizeInternal() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  typedef std::vector<std::unique_ptr<NodeExecStatsWrapper>> NodeExecStatsVec;
  // TODO(suharshs): Make this configurable if its not possible to find a value
  //                 that works for all cases.
  const uint64 kMaxCollectedNodes = 1 << 20;
  mutex mu_;
  bool finalized_ GUARDED_BY(mu_);
  std::unordered_map<string, NodeExecStatsVec> dev_stats_ GUARDED_BY(mu_);
  StepStats* step_stats_ GUARDED_BY(mu_);
  uint64 collectedNodes GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
