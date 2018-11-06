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

#ifndef TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_

#include "tensorflow/core/grappler/graph_view.h"

namespace tensorflow {
namespace grappler {

// A utility class to simplify the traversal of a GraphDef that, unlike
// GraphView, supports updating the graph.  Note that you should not modify the
// graph separately, because the view will get out of sync.

class MutableGraphView : public internal::GraphViewInternal<GraphDef, NodeDef> {
 public:
  explicit MutableGraphView(GraphDef* graph) : GraphViewInternal(graph) {
    for (NodeDef& node : *graph->mutable_node()) AddUniqueNodeOrDie(&node);
    for (NodeDef& node : *graph->mutable_node()) AddFanouts(&node);
  }

  // Lookup fanouts/fanins using immutable ports.
  using GraphViewInternal::GetFanout;
  const absl::flat_hash_set<InputPort>& GetFanout(
      const GraphView::OutputPort& port) const;

  using GraphViewInternal::GetFanin;
  absl::flat_hash_set<OutputPort> GetFanin(
      const GraphView::InputPort& port) const;

  using GraphViewInternal::GetRegularFanin;
  const OutputPort GetRegularFanin(const GraphView::InputPort& port) const;

  // Adds a new node to graph and updates the view. Returns a pointer to the
  // node in graph.
  NodeDef* AddNode(NodeDef&& node);

  // Updates all fanouts (input ports fetching output tensors) from `from_node`
  // to the `to_node`, including control dependencies.
  //
  // Example: We have 2 nodes that use `bar` node output tensors as inputs:
  //   1. foo1(bar:0, bar:1, other:0, ^bar)
  //   2. foo2(bar:1, other:1)
  //
  // After calling ForwardOutputs(bar, new_bar):
  //   1. foo1(new_bar:0, new_bar:1, other:0, ^new_bar)
  //   2. foo2(new_bar:1, other:1)
  void UpdateFanouts(absl::string_view from_node, absl::string_view to_node);

  // Deletes nodes from the graph.
  void DeleteNodes(const std::set<string>& nodes_to_delete);

 private:
  // Updates all fanouts (input ports fetching output tensors) from `from_node`
  // to the `to_node`, including control dependencies.
  //
  // Example: We have 2 nodes that use `bar` node output tensors as inputs:
  //   1. foo1(bar:0, bar:1, other:0, ^bar)
  //   2. foo2(bar:1, other:1)
  //
  // After calling ForwardOutputs(bar, new_bar):
  //   1. foo1(new_bar:0, new_bar:1, other:0, ^new_bar)
  //   2. foo2(new_bar:1, other:1)
  //
  // IMPORTANT: If `from_node` or `to_node` is not in the underlying graph, the
  // behavior is undefined.
  void UpdateFanouts(NodeDef* from_node, NodeDef* to_node);

  // Remove fanouts of the deleted node from internal state (including control
  // dependencies).
  void RemoveFanouts(NodeDef* deleted_node);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
