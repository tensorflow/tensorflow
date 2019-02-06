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
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

const char kMutableGraphViewCtrl[] = "ConstantFoldingCtrl";

// A utility class to simplify the traversal of a GraphDef that, unlike
// GraphView, supports updating the graph.  Note that you should not modify the
// graph separately, because the view will get out of sync.

class MutableGraphView : public internal::GraphViewInternal<GraphDef, NodeDef> {
 public:
  explicit MutableGraphView(GraphDef* graph) : GraphViewInternal(graph) {
    for (NodeDef& node : *graph->mutable_node()) AddUniqueNodeOrDie(&node);
    for (NodeDef& node : *graph->mutable_node()) AddAndDedupFanouts(&node);
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

  // Adds all nodes from the `subgraph` to the underlying graph and updates the
  // view. `subgraph` doesn't have to be a valid graph definition on it's own,
  // it can have edges to the nodes that are not in it, however after adding
  // it to the underlying graph, final graph must be valid.
  //
  // If subgraph function library is not empty, all new functions will be added
  // to the graph. Functions that appear with the same name in both subgraph and
  // the graph represented by *this, must have identical function definitions.
  //
  // IMPORTANT: All nodes and functions of the given subgraph moved into the
  // underlying graph, which leaves subgraph in valid but undefined state.
  Status AddSubgraph(GraphDef&& subgraph);

  // Updates node `node_name` op, device, and attributes. This will clear any
  // existing attributes. If it is not possible to update the node or if the
  // node does not exist, an error will be returned and nothing will be modified
  // in the graph.
  Status UpdateNode(absl::string_view node_name, absl::string_view op,
                    absl::string_view device,
                    absl::Span<const std::pair<string, AttrValue>> attrs);

  // Updates all fanouts (input ports fetching output tensors) from
  // `from_node_name` to the `to_node_name`, including control dependencies.
  //
  // Example: We have 3 nodes that use `bar` node output tensors as inputs:
  //   1. foo1(bar:0, bar:1, other:0)
  //   2. foo2(bar:1, other:1)
  //   3. foo3(other:2, ^bar)
  //
  // After calling ForwardOutputs(bar, new_bar):
  //   1. foo1(new_bar:0, new_bar:1, other:0)
  //   2. foo2(new_bar:1, other:1)
  //   3. foo3(other:2, ^new_bar)
  Status UpdateFanouts(absl::string_view from_node_name,
                       absl::string_view to_node_name);

  // Adds regular fanin `fanin` to node `node_name`. If the node or fanin do not
  // exist in the graph, nothing will be modified in the graph. Otherwise fanin
  // will be added after existing non control dependency fanins. Control
  // dependencies will be deduped. To add control dependencies, use
  // AddControllingFanin.
  Status AddRegularFanin(absl::string_view node_name, const TensorId& fanin);

  // Adds regular fanin `fanin` to node `node_name` at port `port`. If the node
  // or fanin do not exist in the graph, nothing will be modified in the graph.
  // Otherwise fanin will be inserted at port `port`. Control dependencies will
  // be deduped. To add control dependencies, use AddControllingFanin.
  //
  // If the port is not a valid port (less than 0 or greater than the number of
  // regular fanins), this will result in an error and the node will not be
  // modified.
  Status AddRegularFaninByPort(absl::string_view node_name, int port,
                               const TensorId& fanin);

  // Adds control dependency `fanin` to the target node named `node_name`. To
  // add regular fanins, use AddRegularFanin.
  //
  // Case 1: If the fanin is not a Switch node, the control dependency is simply
  // added to the target node:
  //
  //   fanin -^> target node.
  //
  // Case 2: If the fanin is a Switch node, we cannot anchor a control
  // dependency on it, because unlike other nodes, only one of its outputs will
  // be generated when the node is activated. In this case, we try to find an
  // Identity/IdentityN node in the fanout of the relevant port of the Switch
  // and add it as a fanin to the target node. If no such Identity/IdentityN
  // node can be found, a new Identity node will be created. In both cases, we
  // end up with:
  //
  //   fanin -> Identity{N} -^> target node.
  //
  // If the control dependency being added is redundant (control dependency
  // already exists or control dependency can be deduped from regular fanins),
  // this will not result in an error and the node will not be modified.
  Status AddControllingFanin(absl::string_view node_name,
                             const TensorId& fanin);

  // Removes regular fanin `fanin` from node `node_name`. If the node or fanin
  // do not exist in the graph, nothing will be modified in the graph. If there
  // are multiple inputs that match the fanin, all of them will be removed. To
  // remove controlling fanins, use RemoveControllingFanin.
  //
  // If the fanin being removed doesn't exist in the node's inputs, this will
  // not result in an error and the node will not be modified.
  Status RemoveRegularFanin(absl::string_view node_name, const TensorId& fanin);

  // Removes regular fanin at port `port` from node `node_name`. If the node
  // does not exist in the graph, nothing will be modified in the graph.
  // To remove controlling fanins, use RemoveControllingFanin.
  //
  // If the port is not a valid port (less than 0 or greater than the last index
  // of the regular fanins), this will result in an error and the node will not
  // be modified.
  Status RemoveRegularFaninByPort(absl::string_view node_name, int port);

  // Removes control dependency `fanin_node_name` from the target node named
  // `node_name`. If the node or fanin do not exist in the graph, nothing will
  // be modified in the graph. To remove regular fanins, use RemoveRegualrFanin.
  //
  // If the fanin being removed doesn't exist in the node's inputs, this will
  // not result in an error and the node will not be modified.
  Status RemoveControllingFanin(absl::string_view node_name,
                                absl::string_view fanin_node_name);

  // Removes all fanins from node `node_name`. Control dependencies will be
  // retained if keep_controlling_fanins is true.
  //
  // If no fanins are removed, this will not result in an error and the node
  // will not be modified.
  Status RemoveAllFanins(absl::string_view node_name,
                         bool keep_controlling_fanins);

  // Replaces all fanins `from_fanin` with `to_fanin` in node `node_name`. If
  // the fanins or node do not exist, nothing will be modified in the graph.
  // Control dependencies will be deduped.
  //
  // If the fanin being updated doesn't exist in the node's inputs, this will
  // not result in an error and the node will not be modified.
  Status UpdateFanin(absl::string_view node_name, const TensorId& from_fanin,
                     const TensorId& to_fanin);

  // Replaces fanin at port `port` in node `node_name` with fanin `fanin`. If
  // the fanins or node do not exist, nothing will be modified in the graph.
  // Control dependencies will be deduped.
  //
  // If the port is not a valid port (less than 0 or greater than the last index
  // of the regular fanins), this will result in an error and the node will not
  // be modified.
  Status UpdateRegularFaninByPort(absl::string_view node_name, int port,
                                  const TensorId& fanin);

  // Swaps fanins at ports `from_port` and `to_port` in node `node_name`. If the
  // node does not exist, nothing will be modified in the graph.
  //
  // If the ports are not a valid port (less than 0 or greater than the last
  // index of the regular fanins), this will result in an error and the node
  // will not be modified.
  Status SwapRegularFaninsByPorts(absl::string_view node_name, int from_port,
                                  int to_port);

  // Updates all regular fanins to equivalent controlling fanins. If it is not
  // possible, an error will be returned and nothing will be modified in the
  // graph.
  Status UpdateAllRegularFaninsToControlling(absl::string_view node_name);

  // Deletes nodes from the graph. If a node can't be safely removed,
  // specifically if a node still has fanouts, an error will be returned. Nodes
  // that can't be found are ignored.
  Status DeleteNodes(const absl::flat_hash_set<string>& nodes_to_delete);

 private:
  // Adds fanouts for fanins of node to graph, while deduping control
  // dependencies from existing control dependencies and regular fanins. Note,
  // node inputs will be mutated if control dependencies can be deduped.
  void AddAndDedupFanouts(NodeDef* node);

  // Finds next output port smaller than fanin.port_id and update. The
  // max_regular_output_port is only updated if fanin.port_id is the same as the
  // current max_regular_output_port and if the fanouts set is empty. If there
  // are no regular outputs, max_regular_output_port will be erased.
  void UpdateMaxRegularOutputPortForRemovedFanin(
      const OutputPort& fanin,
      const absl::flat_hash_set<InputPort>& fanin_fanouts);

  // Updates max regular output port for newly added fanin by checking the
  // current max and updating if the newly added fanin is of a larger port.
  void UpdateMaxRegularOutputPortForAddedFanin(const OutputPort& fanin);

  // Updates all fanouts (input ports fetching output tensors) from `from_node`
  // to the `to_node`, including control dependencies.
  //
  // Example: We have 3 nodes that use `bar` node output tensors as inputs:
  //   1. foo1(bar:0, bar:1, other:0)
  //   2. foo2(bar:1, other:1)
  //   3. foo3(other:2, ^bar)
  //
  // After calling ForwardOutputs(bar, new_bar):
  //   1. foo1(new_bar:0, new_bar:1, other:0)
  //   2. foo2(new_bar:1, other:1)
  //   3. foo3(other:2, ^new_bar)
  //
  // IMPORTANT: If `from_node` or `to_node` is not in the underlying graph, the
  // behavior is undefined.
  Status UpdateFanoutsInternal(NodeDef* from_node, NodeDef* to_node);

  // Adds fanin to node. If fanin is a control dependency, existing control
  // dependencies will be checked first before adding. Otherwise fanin will be
  // added after existing non control dependency inputs.
  bool AddFaninInternal(NodeDef* node, const OutputPort& fanin);

  // Finds control dependency node to be used based on fanin. If fanin is not a
  // Switch node, fanin.node is simply returned. Otherwise this will try to find
  // a candidate Identity node consuming fanin, as the control dependency. If it
  // is not possible or will introduce a self loop, an error message will be
  // set. If nullptr is returned with no error
  // GetOrCreateIdentityConsumingSwitch should be called to generate the new
  // Identity node.
  NodeDef* GetControllingFaninToAdd(absl::string_view node_name,
                                    const OutputPort& fanin, string* error_msg);

  // Finds a generated Identity node consuming Switch node `fanin.node` at port
  // `fanin.port_id`. If such a node does not exist, a new Identity node will be
  // created.
  NodeDef* GetOrCreateIdentityConsumingSwitch(const OutputPort& fanin);

  // Removes all instances of regular fanin `fanin` from node `node`.
  bool RemoveRegularFaninInternal(NodeDef* node, const OutputPort& fanin);

  // Removes controlling fanin `fanin_node` from node if such controlling fanin
  // exists.
  bool RemoveControllingFaninInternal(NodeDef* node, NodeDef* fanin_node);

  // Checks if nodes to be deleted are missing or have any fanouts that will
  // remain in the graph. If node is removed in either case, the graph will
  // enter an invalid state.
  Status CheckNodesCanBeDeleted(
      const absl::flat_hash_set<string>& nodes_to_delete);

  // Removes fanins of the deleted node from internal state. Control
  // dependencies are retained iff keep_controlling_fanins is true.
  void RemoveFaninsInternal(NodeDef* deleted_node,
                            bool keep_controlling_fanins);

  // Removes fanouts of the deleted node from internal state.
  void RemoveFanoutsInternal(NodeDef* deleted_node);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
