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

namespace tensorflow {
namespace grappler {

namespace {

template <typename GraphViewType>
void ReverseDfsInternal(
    const GraphViewType& graph_view, const std::vector<const NodeDef*>& from,
    const std::function<void(const NodeDef*)>& pre_order,
    const std::function<void(const NodeDef*)>& post_order,
    const std::function<void(const NodeDef*, const NodeDef*)>& on_back_edge) {
  // Stack of work to do.
  struct StackElem {
    const NodeDef* node;
    bool children_visited;
    const NodeDef* src;
  };
  std::vector<StackElem> stack;

  stack.reserve(from.size());
  for (const NodeDef* node : from) {
    stack.push_back(StackElem{node, false});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  absl::flat_hash_map<const NodeDef*, NodeState> node_state;
  while (!stack.empty()) {
    StackElem w = stack.back();
    stack.pop_back();

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.push_back(StackElem{w.node, true, w.src});

    // Now enqueue the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      stack.push_back(StackElem{fanin.node, false, w.node});
    }
  }
}

}  // namespace

void ReverseDfs(
    const GraphView& graph_view, const std::vector<const NodeDef*>& from,
    const std::function<void(const NodeDef*)>& pre_order,
    const std::function<void(const NodeDef*)>& post_order,
    const std::function<void(const NodeDef*, const NodeDef*)>& on_back_edge) {
  ReverseDfsInternal<GraphView>(graph_view, from, pre_order, post_order,
                                on_back_edge);
}

void ReverseDfs(
    const MutableGraphView& graph_view, const std::vector<const NodeDef*>& from,
    const std::function<void(const NodeDef*)>& pre_order,
    const std::function<void(const NodeDef*)>& post_order,
    const std::function<void(const NodeDef*, const NodeDef*)>& on_back_edge) {
  ReverseDfsInternal<MutableGraphView>(graph_view, from, pre_order, post_order,
                                       on_back_edge);
}

}  // namespace grappler
}  // namespace tensorflow
