/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class CostModel;
class Graph;
class Node;
class NodeExecStats;
class StepStats;

class StepStatsCollector {
 public:
  explicit StepStatsCollector(StepStats* ss,
                              CostModelManager* cost_model_manager = nullptr);

  void UpdateCostModel(const NodeExecStats* nt, const Graph* graph,
                       const Node* node);
  void Save(const string& device, NodeExecStats* nt);

  void Swap(StepStats* ss);

 private:
  friend class StepStatsMgr;
  mutex mu_;
  StepStats* step_stats_ GUARDED_BY(mu_);
  CostModelManager* cost_model_manager_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
