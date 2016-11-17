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

#include <unordered_map>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class CostModelManager;
class Graph;
class NodeExecStats;
class StepStats;

// StepStatsCollector manages the collection of a StepStats object.
// The StepStats object holds multiple DeviceStats.
// Each DeviceStats object holds multiple NodeExecStats.
class StepStatsCollector {
 public:
  explicit StepStatsCollector(StepStats* ss);

  // BuildCostModel builds or updates a CostModel managed by cost_model_manager,
  // using the currently collected DeviceStats associated with the devices in
  // device_map.
  void BuildCostModel(
      CostModelManager* cost_model_manager,
      const std::unordered_map<string, const Graph*>& device_map);

  // Save saves nt to the DeviceStats object associated with device.
  void Save(const string& device, NodeExecStats* nt);

  // Swap replaces the current step stats with ss.
  void Swap(StepStats* ss);

 private:
  // TODO(suharshs): Make this configurable if its not possible to find a value
  //                 that works for all cases.
  const uint64 kMaxCollectedNodes = 1 << 20;
  mutex mu_;
  StepStats* step_stats_ GUARDED_BY(mu_);
  uint64 collectedNodes GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
