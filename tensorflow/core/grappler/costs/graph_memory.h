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

#ifndef TENSORFLOW_GRAPPLER_COSTS_GRAPH_MEMORY_H_
#define TENSORFLOW_GRAPPLER_COSTS_GRAPH_MEMORY_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// Infer the worst case memory usage for a given grappler item.
class GraphMemory {
 public:
  explicit GraphMemory(const GrapplerItem& item)
      : item_(item), worst_case_memory_usage_(-1) {}

  Status InferStatically();
  Status InferDynamically(Cluster* cluster);
  Status InferFromGraphProperties(GraphProperties* properties);

  // Worst case memory usage in bytes, or -1 if the usage is unknown.
  int64 GetWorstCaseMemoryUsage() const { return worst_case_memory_usage_; }

  // Best case memory usage in bytes, or -1 if the usage is unknown.
  // This corresponds to the case where all the data is swapped out excepted
  // that which is needed for a single node to perform its computations.
  int64 GetBestCaseMemoryUsage() const { return best_case_memory_usage_; }

 private:
  void InferMemUsageForNodes(const std::vector<const NodeDef*>& nodes,
                             GraphProperties* properties, int64* worst_case,
                             int64* best_case) const;
  int64 InferMemUsageForNeighbors(
      const std::vector<OpInfo::TensorProperties>& props) const;

  // Inputs
  GrapplerItem item_;
  int64 worst_case_memory_usage_;
  int64 best_case_memory_usage_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_COSTS_GRAPH_MEMORY_H_
