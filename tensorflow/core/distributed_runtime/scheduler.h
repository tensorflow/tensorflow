/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SCHEDULER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SCHEDULER_H_

#include <functional>
#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"

namespace tensorflow {

class SlackAnalysis {
 public:
  SlackAnalysis(const Graph* g, const CostModel* cost_model);

  ~SlackAnalysis() {}

  // Compute the earliest possible start time for each node, based on
  // a given cost model. 'asap_time' is indexed by node id.
  Microseconds ComputeAsap(std::vector<Microseconds>* asap_times);

  // Compute the latest possible start time for each node, based on
  // a given cost model. 'alap_time' is indexed by node id.
  Microseconds ComputeAlap(std::vector<Microseconds>* alap_times);

  // Compute the "slack" of each node. 'slacks' is indexed by node id.
  void ComputeSlack(std::vector<int64>* slacks);

 private:
  const Graph* graph_;
  const CostModel* cost_model_;

  TF_DISALLOW_COPY_AND_ASSIGN(SlackAnalysis);
};

class GreedyScheduler {
 public:
  struct Sim {
    int degree_parallelism;
    int num_running;
    std::vector<Node*> ready_nodes;
  };

  struct Event {
    Node* node;
    Microseconds time;
    bool is_completion;

    bool operator<(const Event& other) const { return time < other.time; }
  };

  GreedyScheduler(const DeviceSet* devices, const CostModel* cost_model,
                  const Graph* g, std::vector<int64>* priority);

  ~GreedyScheduler();

  // Computes the start time of each node given the priorities of
  // the nodes.
  Microseconds ComputeSchedule(std::vector<Microseconds>* start_times);

 private:
  // Returns the ready node with the highest priority for a sim.
  Node* GetNodeWithHighestPriority(const std::vector<Node*>& nodes);

  const DeviceSet* devices_;
  const CostModel* cost_model_;
  const Graph* graph_;
  std::vector<int64>* priority_;
  std::unordered_map<string, Sim*> device_states_;

  TF_DISALLOW_COPY_AND_ASSIGN(GreedyScheduler);
};

class PriorityScheduler {
 public:
  PriorityScheduler(const DeviceSet* devices, const CostModel* cost_model,
                    const Graph* g);

  ~PriorityScheduler() {}

  // Computes a schedule of the ideal start time for each node.
  // Returns the makespan (the total running time).
  Microseconds ComputeSchedule(std::vector<Microseconds>* start_times);

  // Computes a schedule and assigns priorities to the nodes based on
  // the schedule. Returns the makespan.
  Microseconds AssignPriorities(std::vector<int64>* priorities);

 private:
  const DeviceSet* devices_;
  const CostModel* cost_model_;
  const Graph* graph_;

  TF_DISALLOW_COPY_AND_ASSIGN(PriorityScheduler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SCHEDULER_H_
