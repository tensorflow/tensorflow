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

#ifndef TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_

#include <unordered_map>
#include <vector>
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// A TensorFlow model to optimize.
// Models are represented by the combination of a graph, one of more fetch
// nodes, and potentially a set of nodes to feed.
class GraphProperties {
 public:
  // Factory method for creating a GrapplerShapes from a MetaGraphDef.
  // Returns nullptr if the given meta_graph cannot be converted.
  explicit GraphProperties(const GrapplerItem& item) : item_(item) {}

  Status InferStatically();
  Status InferDynamically(Cluster* cluster);
  Status InferFromCostGraph(const CostGraphDef& cost_graph);

  bool HasOutputProperties(const string& name) const;
  std::vector<OpInfo::TensorProperties> GetInputProperties(
      const string& node_name) const;
  std::vector<OpInfo::TensorProperties> GetOutputProperties(
      const string& node_name) const;

 private:
  // Inputs
  GrapplerItem item_;
  std::map<string, std::vector<OpInfo::TensorProperties>> input_properties_;
  std::map<string, std::vector<OpInfo::TensorProperties>> output_properties_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
