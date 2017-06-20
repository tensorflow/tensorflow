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

#include "tensorflow/core/grappler/utils/topological_sort.h"
#include <deque>
#include <unordered_map>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

// Kahn's algorithm is implemented.
// For details, see https://en.wikipedia.org/wiki/Topological_sorting
void TopologicalSort(GraphDef* graph) {
  NodeMap node_map(graph);
  std::deque<const NodeDef*> ready_nodes;
  std::unordered_map<const NodeDef*, int> ready_inputs;
  for (const NodeDef& node : graph->node()) {
    if (node.input_size() == 0) {
      ready_nodes.push_back(&node);
    }
    if (node.op() == "Merge") {
      ready_inputs[&node] = 0;
      for (const auto& input : node.input()) {
        if (node_map.GetNode(input)->op() == "NextIteration") {
          ready_inputs[&node]++;
        }
      }
    } else {
      ready_inputs[&node] = 0;
    }
  }
  GraphDef sorted_graph;
  while (!ready_nodes.empty()) {
    auto ready_node = ready_nodes.front();
    *sorted_graph.add_node() = *ready_node;
    for (const auto& fanout : node_map.GetOutputs(ready_node->name())) {
      ready_inputs[fanout]++;
      if (ready_inputs[fanout] == fanout->input_size()) {
        ready_nodes.push_back(fanout);
      }
    }
    ready_nodes.pop_front();
  }
  if (sorted_graph.node_size() == graph->node_size()) {
    *graph = sorted_graph;
  }
}

}  // namespace grappler
}  // namespace tensorflow
