/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_MEMORY_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_MEMORY_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// Infer the worst case memory usage for a given grappler item.
class GraphMemory {
 public:
  struct LiveTensor {
    string node;
    int output_id;
    size_t memory_used;
    Costs::Duration allocation_time;
    Costs::Duration deallocation_time;
  };
  struct MemoryUsage {
    int64 used_memory;
    std::vector<LiveTensor> live_tensors;
  };

  explicit GraphMemory(const GrapplerItem& item)
      : item_(item), unknown_usage_({-1, {}}) {}

  Status InferStatically(
      const std::unordered_map<string, DeviceProperties>& devices);
  Status InferDynamically(Cluster* cluster);

  // Worst case memory usage in bytes, or -1 if the usage is unknown. If there
  // are multiple devices, returns the highest per device memory usage.
  int64 GetWorstCaseMemoryUsage() const;

  // Returns the peak memory usage for the specified device.
  const MemoryUsage& GetPeakMemoryUsage(const string& device) const {
    auto it = peak_usage_.find(device);
    if (it == peak_usage_.end()) {
      return unknown_usage_;
    }
    return it->second;
  }

 private:
  void InferMemUsageForNodes(const std::vector<const NodeDef*>& nodes,
                             GraphProperties* properties, int64* worst_case,
                             int64* best_case) const;
  int64 InferMemUsageForNeighbors(
      const std::vector<OpInfo::TensorProperties>& props) const;

  void InferFromTrace(const StepStats& timeline);

  const GrapplerItem& item_;
  std::unordered_map<string, int64> worst_case_memory_usage_;
  std::unordered_map<string, MemoryUsage> peak_usage_;
  const MemoryUsage unknown_usage_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_MEMORY_H_
