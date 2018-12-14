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

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

const absl::flat_hash_set<MutableGraphView::InputPort>&
MutableGraphView::GetFanout(const GraphView::OutputPort& port) const {
  return GetFanout(MutableGraphView::OutputPort(const_cast<NodeDef*>(port.node),
                                                port.port_id));
}

absl::flat_hash_set<MutableGraphView::OutputPort> MutableGraphView::GetFanin(
    const GraphView::InputPort& port) const {
  return GetFanin(MutableGraphView::InputPort(const_cast<NodeDef*>(port.node),
                                              port.port_id));
}

const MutableGraphView::OutputPort MutableGraphView::GetRegularFanin(
    const GraphView::InputPort& port) const {
  return GetRegularFanin(MutableGraphView::InputPort(
      const_cast<NodeDef*>(port.node), port.port_id));
}

NodeDef* MutableGraphView::AddNode(NodeDef&& node) {
  auto* node_in_graph = graph()->add_node();
  *node_in_graph = std::move(node);

  AddUniqueNodeOrDie(node_in_graph);

  AddFanouts(node_in_graph);
  return node_in_graph;
}

void MutableGraphView::UpdateFanouts(absl::string_view from_node,
                                     absl::string_view to_node) {
  NodeDef* from_node_ptr = GetNode(from_node);
  NodeDef* to_node_ptr = GetNode(to_node);
  if (from_node_ptr && to_node_ptr) {
    UpdateFanouts(from_node_ptr, to_node_ptr);
  } else if (!from_node_ptr) {
    LOG(WARNING) << absl::Substitute(
        "Can't update fanouts from '$0' to '$1', from node was not found.",
        from_node, to_node);
  } else {
    LOG(WARNING) << absl::Substitute(
        "Can't update fanouts from '$0' to '$1', to node was not found.",
        from_node, to_node);
  }
}

void MutableGraphView::UpdateFanouts(NodeDef* from_node, NodeDef* to_node) {
  VLOG(2) << absl::Substitute("Update fanouts from '$0' to '$1'.",
                              from_node->name(), to_node->name());

  // Update internal state with the new output_port->input_port edge.
  const auto add_edge = [this](const OutputPort& output_port,
                               const InputPort& input_port) {
    fanouts()[output_port].insert(input_port);
  };

  // Remove invalidated edge from the internal state.
  const auto remove_edge = [this](const OutputPort& output_port,
                                  const InputPort& input_port) {
    fanouts()[output_port].erase(input_port);
  };

  // First we update regular fanouts. For the regular fanouts
  // `input_port:port_id` is the input index in NodeDef.

  auto regular_edges =
      GetFanoutEdges(*from_node, /*include_controlled_edges=*/false);

  // Maximum index of the `from_node` output tensor that is still used as an
  // input to some other node.
  int keep_max_regular_output_port = -1;

  for (const Edge& edge : regular_edges) {
    const OutputPort output_port = edge.src;
    const InputPort input_port = edge.dst;

    // If the `to_node` reads from the `from_node`, skip this edge (see
    // AddAndUpdateFanoutsWithoutSelfLoops test for an example).
    if (input_port.node == to_node) {
      keep_max_regular_output_port =
          std::max(keep_max_regular_output_port, input_port.port_id);
      continue;
    }

    // Update input at destination node.
    input_port.node->set_input(
        input_port.port_id,
        output_port.port_id == 0
            ? to_node->name()
            : absl::StrCat(to_node->name(), ":", output_port.port_id));

    // Remove old edge between the `from_node` and the fanout node.
    remove_edge(output_port, input_port);
    // Add an edge between the `to_node` and new fanout node.
    add_edge(OutputPort(to_node, output_port.port_id), input_port);
  }

  // For the control fanouts we do not know the input index in a NodeDef,
  // so we have to traverse all control inputs.

  auto control_fanouts =
      GetFanout(GraphView::OutputPort(from_node, Graph::kControlSlot));
  if (control_fanouts.empty()) return;

  const string from_control_input = absl::StrCat("^", from_node->name());
  const string to_control_input = absl::StrCat("^", to_node->name());

  for (const InputPort& control_port : control_fanouts) {
    // Node can't be control dependency of itself.
    if (control_port.node == to_node) continue;

    // Find and update input corresponding to control dependency.
    NodeDef* node = control_port.node;
    for (int i = node->input_size() - 1; i >= 0; --i) {
      const string& input = node->input(i);
      if (!IsControlInput(input)) break;  // we reached regular inputs
      if (input == from_control_input) {
        node->set_input(i, to_control_input);
      }
    }

    // Remove old edge between the `from_node` and the fanout node.
    remove_edge(OutputPort(from_node, Graph::kControlSlot), control_port);
    // Add an edge between the `to_node` and new fanout node.
    add_edge(OutputPort(to_node, Graph::kControlSlot), control_port);
  }

  // Because we update all regular fanouts of `from_node`, we can just copy
  // the value `num_regular_outputs`.
  max_regular_output_port()[to_node] = max_regular_output_port()[from_node];

  // Check if all fanouts were updated to read from the `to_node`.
  if (keep_max_regular_output_port >= 0) {
    max_regular_output_port()[from_node] = keep_max_regular_output_port;
  } else {
    max_regular_output_port().erase(from_node);
  }
}

void MutableGraphView::DeleteNodes(const std::set<string>& nodes_to_delete) {
  for (const string& node_name_to_delete : nodes_to_delete)
    RemoveFanouts(nodes().at(node_name_to_delete));
  for (const string& node_name_to_delete : nodes_to_delete)
    nodes().erase(node_name_to_delete);
  EraseNodesFromGraph(nodes_to_delete, graph());
}

void MutableGraphView::RemoveFanouts(NodeDef* deleted_node) {
  for (int i = 0; i < deleted_node->input_size(); ++i) {
    TensorId tensor_id = ParseTensorName(deleted_node->input(i));
    OutputPort fanin(nodes()[tensor_id.node()], tensor_id.index());

    InputPort input;
    input.node = deleted_node;
    if (tensor_id.index() < 0)
      input.port_id = Graph::kControlSlot;
    else
      input.port_id = i;

    fanouts()[fanin].erase(input);
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
