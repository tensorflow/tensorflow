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

#include <set>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/platform/types.h"

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

  // Add fanin to node `node_name`. If the node or fanin do not exist in the
  // graph, nothing will be modified in the graph. If fanin is a control
  // dependency, existing control dependencies will be checked first before
  // adding. Otherwise fanin will be added after existing non control dependency
  // inputs.
  //
  // This will return true iff the node is modified. If a control dependency
  // already exists, the node will not be modified.
  bool AddFanin(absl::string_view node_name, const TensorId& fanin);

  // Remove fanin from node `node_name`. If the node or fanin do not exist in
  // the graph, nothing will be modified in the graph. If there are multiple
  // inputs that match the fanin, all of them will be removed.
  //
  // This will return true iff the node is modified. If no inputs match the
  // fanin, the node will not be modified.
  bool RemoveFanin(absl::string_view node_name, const TensorId& fanin);

  // Remove all fanins from node `node_name`. Control dependencies will be
  // retained if keep_controlling_fanins is true.
  //
  // This will return true iff the node is modified.
  bool RemoveAllFanins(absl::string_view node_name,
                       bool keep_controlling_fanins);

  // Replace all fanins `from_fanin` with `to_fanin` in node `node_name`. If
  // the fanins or node do not exist, nothing will be modified in the graph.
  //
  // This will return true iff the node is modified.
  bool UpdateFanin(absl::string_view node_name, const TensorId& from_fanin,
                   const TensorId& to_fanin);

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

  // Remove fanins of the deleted node from internal state. Control dependencies
  // are retained iff keep_controlling_fanins is true.
  void RemoveFaninsInternal(NodeDef* deleted_node,
                            bool keep_controlling_fanins);

  // Add fanin to node. If the node or fanin do not exist in the graph, nothing
  // will be modified in the graph. If fanin is a control dependency, existing
  // control dependencies will be checked first before adding. Otherwise fanin
  // will be added after existing non control dependency inputs.
  //
  // This will return true iff the node is modified. If a control dependency
  // already exists, the node will not be modified.
  bool AddFanin(NodeDef* node, const TensorId& fanin);

  // Remove any fanin in node that matches to a fanin in fanins.
  bool RemoveFanins(NodeDef* node, absl::Span<const TensorId> fanins);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
