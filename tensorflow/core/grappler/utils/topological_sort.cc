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

#include <algorithm>
#include <deque>
#include <unordered_map>

#include "absl/types/span.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

namespace {

std::vector<GraphView::Edge> MakeEphemeralEdges(
    const absl::Span<const TopologicalDependency> extra_dependencies) {
  std::vector<GraphView::Edge> ephemeral_edges;
  ephemeral_edges.reserve(extra_dependencies.size());
  for (const auto& dep : extra_dependencies) {
    ephemeral_edges.emplace_back(
        GraphView::OutputPort(dep.from, Graph::kControlSlot),
        GraphView::InputPort(dep.to, Graph::kControlSlot));
  }
  return ephemeral_edges;
}

// Kahn's algorithm is implemented.
// For details, see https://en.wikipedia.org/wiki/Topological_sorting
Status ComputeTopologicalOrder(
    const GraphDef& graph,
    const absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<int>* ready_nodes) {
  GraphTopologyView graph_view;
  TF_RETURN_IF_ERROR(graph_view.InitializeFromGraph(
      graph, MakeEphemeralEdges(extra_dependencies)));

  // Keep track of how many inputs are ready for the given node.
  std::vector<int> num_ready_inputs(graph.node_size(), 0);

  // We'll push index of ready nodes to this output vector.
  ready_nodes->reserve(graph.node_size());

  int front = 0;
  int back = 0;

  for (int i = 0; i < graph.node_size(); i++) {
    if (graph_view.GetFanin(i).empty()) {
      ready_nodes->push_back(i);
      back++;
    }
    if (IsMerge(graph.node(i))) {
      for (int input : graph_view.GetFanin(i)) {
        if (IsNextIteration(graph.node(input))) {
          num_ready_inputs[i]++;
        }
      }
    }
  }

  while (front != back) {
    int ready_node = (*ready_nodes)[front];
    for (int fanout : graph_view.GetFanout(ready_node)) {
      ++num_ready_inputs[fanout];
      const int max_size = graph_view.GetFanin(fanout).size();
      if (num_ready_inputs[fanout] == max_size) {
        ready_nodes->push_back(fanout);
        ++back;
      }
    }
    ++front;
  }

  if (back != graph_view.num_nodes()) {
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "The graph couldn't be sorted in topological order. Stalled "
                 "at node = "
              << graph.node(back).DebugString();
      for (int i = 0; i < graph_view.num_nodes(); ++i) {
        const int max_size = graph_view.GetFanin(i).size();
        if (num_ready_inputs[i] != max_size) {
          VLOG(1) << "Node not ready: " << graph.node(i).DebugString();
        }
      }
    }
    return errors::InvalidArgument(
        "The graph couldn't be sorted in topological order.");
  }
  return Status::OK();
}

}  // namespace

Status ComputeTopologicalOrder(
    const GraphDef& graph,
    const absl::Span<const TopologicalDependency> extra_dependencies,
    std::vector<const NodeDef*>* topo_order) {
  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(
      ComputeTopologicalOrder(graph, extra_dependencies, &ready_nodes));

  topo_order->reserve(ready_nodes.size());
  for (int ready_node_idx : ready_nodes) {
    topo_order->emplace_back(&graph.node(ready_node_idx));
  }

  return Status::OK();
}

Status ComputeTopologicalOrder(const GraphDef& graph,
                               std::vector<const NodeDef*>* topo_order) {
  return ComputeTopologicalOrder(graph, {}, topo_order);
}

Status ReversedTopologicalSort(GraphDef* graph) {
  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(*graph, {}, &ready_nodes));
  std::reverse(ready_nodes.begin(), ready_nodes.end());
  PermuteNodesInPlace(graph, &ready_nodes, /*invert_permutation=*/true);
  return Status::OK();
}

Status TopologicalSort(GraphDef* graph) {
  std::vector<int> ready_nodes;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(*graph, {}, &ready_nodes));
  PermuteNodesInPlace(graph, &ready_nodes, /*invert_permutation=*/true);
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
