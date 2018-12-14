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

#include <algorithm>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

namespace {

bool IsTensorIdPortValid(const TensorId& tensor_id) {
  return tensor_id.index() >= Graph::kControlSlot;
}

}  // namespace

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

bool MutableGraphView::AddFanin(NodeDef* node, const TensorId& fanin) {
  NodeDef* fanin_node = GetNode(fanin.node());
  if (fanin_node == nullptr) {
    return false;
  }

  int num_non_controlling_fanins =
      NumFanins(*node, /*include_controlling_nodes=*/false);
  InputPort input;
  input.node = node;
  input.port_id = fanin.index() == Graph::kControlSlot
                      ? Graph::kControlSlot
                      : num_non_controlling_fanins;

  OutputPort fanin_port(fanin_node, fanin.index());

  if (!gtl::InsertIfNotPresent(&fanouts()[fanin_port], input)) {
    return false;
  }
  node->add_input(TensorIdToString(fanin));
  if (fanin.index() > Graph::kControlSlot) {
    int node_input_size = node->input_size() - 1;
    // If there are control dependencies in node, move newly inserted fanin to
    // be before such control dependencies.
    if (num_non_controlling_fanins < node_input_size) {
      node->mutable_input()->SwapElements(node_input_size,
                                          num_non_controlling_fanins);
    }
  }
  return true;
}

bool MutableGraphView::AddFanin(absl::string_view node_name,
                                const TensorId& fanin) {
  if (!IsTensorIdPortValid(fanin)) {
    return false;
  }
  NodeDef* node = GetNode(node_name);
  if (node == nullptr) {
    return false;
  }
  return AddFanin(node, fanin);
}

bool MutableGraphView::RemoveFanins(NodeDef* node,
                                    absl::Span<const TensorId> fanins) {
  bool modified = false;
  auto mutable_inputs = node->mutable_input();
  int curr_pos = 0;
  int num_inputs = node->input_size();
  for (int i = 0; i < num_inputs; ++i) {
    TensorId tensor_id = ParseTensorName(node->input(i));
    bool remove_fanin =
        std::find(fanins.begin(), fanins.end(), tensor_id) != fanins.end();
    bool update_fanin = !remove_fanin && modified;
    if (remove_fanin || update_fanin) {
      OutputPort fanin(nodes()[tensor_id.node()], tensor_id.index());

      InputPort input;
      input.node = node;
      input.port_id =
          tensor_id.index() == Graph::kControlSlot ? Graph::kControlSlot : i;

      if (remove_fanin) {
        fanouts()[fanin].erase(input);
      } else {
        // Shift inputs to be retained.
        if (tensor_id.index() > Graph::kControlSlot) {
          fanouts()[fanin].erase(input);
          fanouts()[fanin].insert(InputPort(node, i));
        }
        mutable_inputs->SwapElements(i, curr_pos++);
      }

      modified = true;
    } else {
      // Skip inputs to be retained until first modification.
      curr_pos++;
    }
  }
  if (modified) {
    mutable_inputs->DeleteSubrange(curr_pos, num_inputs - curr_pos);
  }
  return modified;
}

bool MutableGraphView::RemoveFanin(absl::string_view node_name,
                                   const TensorId& fanin) {
  if (!IsTensorIdPortValid(fanin)) {
    return false;
  }
  NodeDef* node = GetNode(node_name);
  if (node == nullptr) {
    return false;
  }
  return RemoveFanins(node, {fanin});
}

bool MutableGraphView::RemoveAllFanins(absl::string_view node_name,
                                       bool keep_controlling_fanins) {
  NodeDef* node = GetNode(node_name);
  if (node == nullptr || node->input().empty()) {
    return false;
  }
  RemoveFaninsInternal(node, keep_controlling_fanins);
  if (keep_controlling_fanins) {
    int num_non_controlling_fanins =
        NumFanins(*node, /*include_controlling_nodes=*/false);
    if (num_non_controlling_fanins == 0) {
      return false;
    } else if (num_non_controlling_fanins < node->input_size()) {
      node->mutable_input()->DeleteSubrange(0, num_non_controlling_fanins);
    } else {
      node->clear_input();
    }
  } else {
    node->clear_input();
  }
  return true;
}

bool MutableGraphView::UpdateFanin(absl::string_view node_name,
                                   const TensorId& from_fanin,
                                   const TensorId& to_fanin) {
  if (from_fanin == to_fanin || !IsTensorIdPortValid(from_fanin) ||
      !IsTensorIdPortValid(to_fanin)) {
    return false;
  }
  NodeDef* node = GetNode(node_name);
  if (node == nullptr) {
    return false;
  }

  bool is_from_fanin_control = from_fanin.index() == Graph::kControlSlot;
  bool is_to_fanin_control = to_fanin.index() == Graph::kControlSlot;
  // When replacing a non control dependency fanin with a control dependency, or
  // vice versa, remove and add, so ports can be updated properly in fanout(s).
  if (is_from_fanin_control || is_to_fanin_control) {
    bool modified = RemoveFanins(node, {from_fanin});
    if (!HasFanin(*node, to_fanin)) {
      modified |= AddFanin(node, to_fanin);
    }
    return modified;
  }

  // In place mutation, requires no shifting of ports.
  NodeDef* from_fanin_node = GetNode(from_fanin.node());
  NodeDef* to_fanin_node = GetNode(to_fanin.node());
  if (from_fanin_node == nullptr || to_fanin_node == nullptr) {
    return false;
  }

  string to_fanin_string = TensorIdToString(to_fanin);
  int num_inputs = node->input_size();
  bool modified = false;
  for (int i = 0; i < num_inputs; ++i) {
    if (ParseTensorName(node->input(i)) == from_fanin) {
      OutputPort from_fanin_port(from_fanin_node, from_fanin.index());
      InputPort old_input;
      old_input.node = node;
      old_input.port_id =
          from_fanin.index() == Graph::kControlSlot ? Graph::kControlSlot : i;
      fanouts()[from_fanin_port].erase(old_input);

      OutputPort to_fanin_port(to_fanin_node, to_fanin.index());
      InputPort new_input;
      new_input.node = node;
      new_input.port_id =
          to_fanin.index() == Graph::kControlSlot ? Graph::kControlSlot : i;
      fanouts()[to_fanin_port].insert(new_input);

      node->set_input(i, to_fanin_string);
      modified = true;
    }
  }

  return modified;
}

void MutableGraphView::DeleteNodes(const std::set<string>& nodes_to_delete) {
  for (const string& node_name_to_delete : nodes_to_delete)
    RemoveFaninsInternal(nodes().at(node_name_to_delete),
                         /*keep_controlling_fanins=*/false);
  for (const string& node_name_to_delete : nodes_to_delete)
    nodes().erase(node_name_to_delete);
  EraseNodesFromGraph(nodes_to_delete, graph());
}

void MutableGraphView::RemoveFaninsInternal(NodeDef* deleted_node,
                                            bool keep_controlling_fanins) {
  for (int i = 0; i < deleted_node->input_size(); ++i) {
    TensorId tensor_id = ParseTensorName(deleted_node->input(i));
    if (keep_controlling_fanins && tensor_id.index() < 0) {
      break;
    }
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
