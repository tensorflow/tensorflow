/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/traversal.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"

namespace tensorflow {
namespace grappler {

namespace {

struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {}

  // Index of the node in the graph ∊ [0, num_nodes).
  int node;
  // `True` if visited all the input/output nodes (pushed all input/output nodes
  // to the stack).
  bool children_visited;
  // Index of the node in the graph, from which we entered the `node`.
  int src;
};

enum class NodeState { kNotVisited, kVisiting, kDone };

}  // namespace

void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  const TraversalDirection direction,
                  const DfsPredicates& predicates,
                  const DfsCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());

  for (const NodeDef* node : from) {
    const absl::optional<int> node_idx = graph_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value()) << "Illegal start node: " << node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }

  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();

    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;

    // Skip nodes that we should not enter.
    if (predicates.enter && !predicates.enter(graph_view.GetNode(w.node))) {
      state = NodeState::kDone;
      continue;
    }

    // We've processed all the children of this node.
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(graph_view.GetNode(w.node));
      }
      continue;
    }

    // Loop detected.
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(graph_view.GetNode(w.src),
                               graph_view.GetNode(w.node));
      }
      continue;
    }

    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(graph_view.GetNode(w.node));
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.emplace_back(w.node, true, w.src);

    // Check if we can continue traversal from the current node.
    if (predicates.advance && !predicates.advance(graph_view.GetNode(w.node))) {
      continue;
    }

    // Now enqueue the fanin/fanout nodes.
    if (direction == TraversalDirection::kFollowInputs) {
      for (const int fanin : graph_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    } else {
      for (const int fanout : graph_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}

void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsCallbacks& callbacks) {
  DfsTraversal(graph_view, from, direction, {}, callbacks);
}

}  // namespace grappler
}  // namespace tensorflow
