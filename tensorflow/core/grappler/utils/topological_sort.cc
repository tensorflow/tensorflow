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
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

// Kahn's algorithm is implemented.
// For details, see https://en.wikipedia.org/wiki/Topological_sorting
void TopologicalSort(GraphDef* graph) {
  OutputMap output_map(graph);
  std::vector<NodeDef*> ready_nodes;
  ready_nodes.reserve(graph->node_size());
  int front = 0;
  int back = 0;
  std::unordered_map<const NodeDef*, int> ready_inputs;
  for (int i = 0; i < graph->node_size(); i++) {
    auto node = graph->mutable_node(i);
    if (node->input_size() == 0) {
      ready_nodes.push_back(node);
      back++;
    }
    if (IsMerge(*node)) {
      ready_inputs[node] = 0;
      for (const auto& input : node->input()) {
        if (IsNextIteration(*output_map.GetNode(input))) {
          ready_inputs[node]++;
        }
      }
    } else {
      ready_inputs[node] = 0;
    }
  }

  while (front != back) {
    auto ready_node = ready_nodes[front];
    for (const auto& fanout_pair : output_map.GetOutputs(ready_node->name())) {
      auto fanout = fanout_pair.first;
      ready_inputs[fanout] += fanout_pair.second;
      if (ready_inputs[fanout] == fanout->input_size()) {
        ready_nodes.push_back(fanout);
        back++;
      }
    }
    front++;
  }

  if (back == graph->node_size()) {
    GraphDef new_graph;
    new_graph.mutable_node()->Reserve(graph->node_size());
    for (int i = 0; i < graph->node_size(); i++) {
      auto new_node = new_graph.add_node();
      new_node->Swap(ready_nodes[i]);
    }
    graph->mutable_node()->Swap(new_graph.mutable_node());
  } else {
    LOG(ERROR) << "The graph couldn't be sorted in topological order.";
  }
}

}  // namespace grappler
}  // namespace tensorflow
