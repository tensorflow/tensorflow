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
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

// Kahn's algorithm is implemented.
// For details, see https://en.wikipedia.org/wiki/Topological_sorting
Status ComputeTopologicalOrder(const GraphDef& graph,
                               std::vector<int>* ready_nodes) {
  SimpleGraphView graph_view;
  TF_RETURN_IF_ERROR(graph_view.Initialize(graph));

  ready_nodes->reserve(graph_view.num_nodes());

  int front = 0;
  int back = 0;
  std::vector<int> num_ready_inputs(graph_view.num_nodes(), 0);
  for (int i = 0; i < graph_view.num_nodes(); i++) {
    if (graph_view.inputs(i).empty()) {
      ready_nodes->push_back(i);
      back++;
    }
    if (IsMerge(graph.node(i))) {
      for (int input : graph_view.inputs(i)) {
        if (IsNextIteration(graph.node(input))) {
          num_ready_inputs[i]++;
        }
      }
    }
  }

  while (front != back) {
    int ready_node = (*ready_nodes)[front];
    for (int fanout : graph_view.outputs(ready_node)) {
      ++num_ready_inputs[fanout];
      if (num_ready_inputs[fanout] == graph_view.inputs(fanout).size()) {
        ready_nodes->push_back(fanout);
        ++back;
      }
    }
    ++front;
  }

  if (back != graph_view.num_nodes()) {
    return errors::InvalidArgument(
        "The graph couldn't be sorted in topological order.");
  }
  return Status::OK();
}

Status ComputeTopologicalOrder(
    const GraphDef& graph,
    std::unordered_map<const NodeDef*, int>* topo_order) {
  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(graph, &ready_nodes));
  topo_order->reserve(graph.node_size());
  for (int i = 0; i < ready_nodes.size(); ++i) {
    (*topo_order)[&graph.node(ready_nodes[i])] = i;
  }
  return Status::OK();
}

Status TopologicalSort(GraphDef* graph) {
  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(*graph, &ready_nodes));
  PermuteNodesInPlace(graph, &ready_nodes, /*invert_permutation=*/true);
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
