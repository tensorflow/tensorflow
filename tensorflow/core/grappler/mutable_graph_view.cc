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
#include "absl/strings/string_view.h"
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

// Determines if node is an Identity where it's first regular input is a Switch
// node.
bool IsIdentityConsumingSwitch(const MutableGraphView& graph,
                               const NodeDef& node) {
  if ((IsIdentity(node) || IsIdentityNSingleInput(node)) &&
      node.input_size() > 0) {
    TensorId tensor_id = ParseTensorName(node.input(0));
    if (tensor_id.index() == Graph::kControlSlot) return false;
    NodeDef* input_node = graph.GetNode(tensor_id.node());
    return IsSwitch(*input_node);
  }
  return false;
}

// Determines if node input can be deduped by regular inputs when used as a
// control dependency. Specifically, if a node is an Identity that leads to a
// Switch node, when used as a control dependency, that control dependency
// should not be deduped even though the same node is used as a regular input.
bool CanDedupControlWithRegularInput(const MutableGraphView& graph,
                                     const NodeDef& control_node) {
  return !IsIdentityConsumingSwitch(graph, control_node);
}

// Determines if node input can be deduped by regular inputs when used as a
// control dependency. Specifically, if a node is an Identity that leads to a
// Switch node, when used as a control dependency, that control dependency
// should not be deduped even though the same node is used as a regular input.
bool CanDedupControlWithRegularInput(const MutableGraphView& graph,
                                     absl::string_view control_node_name) {
  NodeDef* control_node = graph.GetNode(control_node_name);
  return CanDedupControlWithRegularInput(graph, *control_node);
}

}  // namespace

void MutableGraphView::AddAndDedupFanouts(NodeDef* node) {
  absl::flat_hash_set<absl::string_view> fanins;
  absl::flat_hash_set<absl::string_view> controlling_fanins;
  int pos = 0;
  const int last_idx = node->input_size() - 1;
  int last_pos = last_idx;
  while (pos <= last_pos) {
    TensorId tensor_id = ParseTensorName(node->input(pos));
    absl::string_view input_node_name = tensor_id.node();
    bool is_control_input = IsControlInput(tensor_id);
    bool can_dedup_control_with_regular_input =
        CanDedupControlWithRegularInput(*this, input_node_name);
    bool can_dedup_control =
        is_control_input && (can_dedup_control_with_regular_input ||
                             (!can_dedup_control_with_regular_input &&
                              controlling_fanins.contains(input_node_name)));
    if (!gtl::InsertIfNotPresent(&fanins, input_node_name) &&
        can_dedup_control) {
      node->mutable_input()->SwapElements(pos, last_pos--);
    } else {
      OutputPort output(nodes()[input_node_name], tensor_id.index());

      if (is_control_input) {
        fanouts()[output].emplace(node, Graph::kControlSlot);
      } else {
        max_regular_output_port()[output.node] =
            std::max(max_regular_output_port()[output.node], output.port_id);
        fanouts()[output].emplace(node, pos);
      }
      ++pos;
    }
    if (is_control_input) {
      controlling_fanins.insert(input_node_name);
    }
  }

  if (last_pos < last_idx) {
    node->mutable_input()->DeleteSubrange(last_pos + 1, last_idx - last_pos);
  }
}

void MutableGraphView::UpdateMaxRegularOutputPortForRemovedFanin(
    const OutputPort& fanin,
    const absl::flat_hash_set<InputPort>& fanin_fanouts) {
  int max_port = max_regular_output_port()[fanin.node];
  if (!fanin_fanouts.empty() || max_port != fanin.port_id) {
    return;
  }
  bool updated_max_port = false;
  for (int i = fanin.port_id - 1; i >= 0; --i) {
    OutputPort fanin_port(fanin.node, i);
    if (!fanouts()[fanin_port].empty()) {
      max_regular_output_port()[fanin.node] = i;
      updated_max_port = true;
      break;
    }
  }
  if (!updated_max_port) {
    max_regular_output_port().erase(fanin.node);
  }
}

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

  AddAndDedupFanouts(node_in_graph);
  return node_in_graph;
}

bool MutableGraphView::UpdateFanouts(absl::string_view from_node,
                                     absl::string_view to_node) {
  NodeDef* from_node_ptr = GetNode(from_node);
  NodeDef* to_node_ptr = GetNode(to_node);
  if (from_node_ptr && to_node_ptr) {
    return UpdateFanoutsInternal(from_node_ptr, to_node_ptr);
  } else if (!from_node_ptr) {
    LOG(WARNING) << absl::Substitute(
        "Can't update fanouts from '$0' to '$1', from node was not found.",
        from_node, to_node);
  } else {
    LOG(WARNING) << absl::Substitute(
        "Can't update fanouts from '$0' to '$1', to node was not found.",
        from_node, to_node);
  }
  return false;
}

bool MutableGraphView::UpdateFanoutsInternal(NodeDef* from_node,
                                             NodeDef* to_node) {
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

  bool modified = false;
  for (const Edge& edge : regular_edges) {
    const OutputPort output_port = edge.src;
    const InputPort input_port = edge.dst;

    // If the `to_node` reads from the `from_node`, skip this edge (see
    // AddAndUpdateFanoutsWithoutSelfLoops test for an example).
    if (input_port.node == to_node) {
      keep_max_regular_output_port =
          std::max(keep_max_regular_output_port, output_port.port_id);
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
    // Dedup control dependency.
    if (CanDedupControlWithRegularInput(*this, *to_node)) {
      RemoveControllingFaninInternal(input_port.node, to_node);
    }
    modified = true;
  }

  // For the control fanouts we do not know the input index in a NodeDef,
  // so we have to traverse all control inputs.

  auto control_fanouts =
      GetFanout(GraphView::OutputPort(from_node, Graph::kControlSlot));

  const string from_control_input = absl::StrCat("^", from_node->name());
  const string to_control_input = absl::StrCat("^", to_node->name());

  for (const InputPort& control_port : control_fanouts) {
    // Node can't be control dependency of itself.
    if (control_port.node == to_node) continue;

    NodeDef* node = control_port.node;
    modified |= RemoveControllingFaninInternal(node, from_node);
    modified |= AddFaninInternal(node, {to_node, Graph::kControlSlot});
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

  return modified;
}

bool MutableGraphView::AddFaninInternal(NodeDef* node,
                                        const OutputPort& fanin) {
  int num_non_controlling_fanins =
      NumFanins(*node, /*include_controlling_nodes=*/false);
  bool input_is_control = fanin.port_id == Graph::kControlSlot;
  bool can_dedup_control_with_regular_input =
      CanDedupControlWithRegularInput(*this, *fanin.node);
  // Don't add duplicate control dependencies.
  if (input_is_control) {
    const int start =
        can_dedup_control_with_regular_input ? 0 : num_non_controlling_fanins;
    for (int i = start; i < node->input_size(); ++i) {
      if (ParseTensorName(node->input(i)).node() == fanin.node->name()) {
        return false;
      }
    }
  }

  InputPort input;
  input.node = node;
  input.port_id =
      input_is_control ? Graph::kControlSlot : num_non_controlling_fanins;

  node->add_input(TensorIdToString({fanin.node->name(), fanin.port_id}));
  if (fanin.port_id > Graph::kControlSlot) {
    int node_input_size = node->input_size() - 1;
    // If there are control dependencies in node, move newly inserted fanin to
    // be before such control dependencies.
    if (num_non_controlling_fanins < node_input_size) {
      node->mutable_input()->SwapElements(node_input_size,
                                          num_non_controlling_fanins);
    }
  }

  fanouts()[fanin].insert(input);
  if (max_regular_output_port()[fanin.node] < fanin.port_id) {
    max_regular_output_port()[fanin.node] = fanin.port_id;
  }

  // Dedup control dependencies.
  if (!input_is_control && can_dedup_control_with_regular_input) {
    RemoveControllingFaninInternal(node, fanin.node);
  }

  return true;
}

bool MutableGraphView::AddFaninInternal(NodeDef* node, const TensorId& fanin) {
  NodeDef* fanin_node = GetNode(fanin.node());
  if (fanin_node == nullptr) {
    return false;
  }
  return AddFaninInternal(node, {fanin_node, fanin.index()});
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
  return AddFaninInternal(node, fanin);
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

      auto& fanouts_set = fanouts()[fanin];
      if (remove_fanin) {
        fanouts_set.erase(input);
      } else {
        // Shift inputs to be retained.
        if (tensor_id.index() > Graph::kControlSlot) {
          fanouts_set.erase(input);
          fanouts_set.insert(InputPort(node, i));
        }
        mutable_inputs->SwapElements(i, curr_pos++);
      }
      UpdateMaxRegularOutputPortForRemovedFanin(fanin, fanouts_set);

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

  // When replacing a non control dependency fanin with a control dependency, or
  // vice versa, remove and add, so ports can be updated properly in fanout(s).
  if (from_fanin.index() == Graph::kControlSlot ||
      to_fanin.index() == Graph::kControlSlot) {
    bool modified = RemoveFanins(node, {from_fanin});
    modified |= AddFaninInternal(node, to_fanin);
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
  absl::flat_hash_set<InputPort>* from_fanin_port_fanouts = nullptr;
  absl::flat_hash_set<InputPort>* to_fanin_port_fanouts = nullptr;
  for (int i = 0; i < num_inputs; ++i) {
    if (ParseTensorName(node->input(i)) == from_fanin) {
      InputPort old_input;
      old_input.node = node;
      old_input.port_id =
          from_fanin.index() == Graph::kControlSlot ? Graph::kControlSlot : i;
      if (from_fanin_port_fanouts == nullptr) {
        OutputPort from_fanin_port(from_fanin_node, from_fanin.index());
        from_fanin_port_fanouts = &fanouts()[from_fanin_port];
      }
      from_fanin_port_fanouts->erase(old_input);

      InputPort new_input;
      new_input.node = node;
      new_input.port_id =
          to_fanin.index() == Graph::kControlSlot ? Graph::kControlSlot : i;
      if (to_fanin_port_fanouts == nullptr) {
        OutputPort to_fanin_port(to_fanin_node, to_fanin.index());
        to_fanin_port_fanouts = &fanouts()[to_fanin_port];
      }
      to_fanin_port_fanouts->insert(new_input);

      node->set_input(i, to_fanin_string);
      modified = true;
    }
  }

  // Dedup control dependencies and update max regular output ports.
  if (modified) {
    UpdateMaxRegularOutputPortForRemovedFanin(
        {from_fanin_node, from_fanin.index()}, *from_fanin_port_fanouts);
    if (max_regular_output_port()[to_fanin_node] < to_fanin.index()) {
      max_regular_output_port()[to_fanin_node] = to_fanin.index();
    }
    if (CanDedupControlWithRegularInput(*this, *to_fanin_node)) {
      RemoveControllingFaninInternal(node, to_fanin_node);
    }
  }

  return modified;
}

bool MutableGraphView::AddControllingFanin(absl::string_view node_name,
                                           const TensorId& fanin) {
  NodeDef* node = GetNode(node_name);
  if (node == nullptr) {
    return false;
  }
  NodeDef* fanin_node = GetNode(fanin.node());
  if (fanin_node == nullptr) {
    return false;
  }
  if (fanin.index() == Graph::kControlSlot) {
    return AddFaninInternal(node, {fanin_node, Graph::kControlSlot});
  }

  if (!IsSwitch(*fanin_node)) {
    return AddFaninInternal(node, {fanin_node, Graph::kControlSlot});
  } else {
    // We can't anchor control dependencies directly on the switch node: unlike
    // other nodes only one of the outputs of the switch node will be generated
    // when the switch node is executed, and we need to make sure the control
    // dependency is only triggered when the corresponding output is triggered.
    // We start by looking for an identity node connected to the output of the
    // switch node, and use it to anchor the control dependency.
    auto fanouts = GetFanouts(*fanin_node, /*include_controlled_nodes=*/false);
    for (auto fanout : fanouts) {
      if (IsIdentity(*fanout.node) || IsIdentityNSingleInput(*fanout.node)) {
        if (ParseTensorName(fanout.node->input(0)) == fanin) {
          return AddFaninInternal(node, {fanout.node, Graph::kControlSlot});
        }
      }
    }
    // We haven't found an existing node where we can anchor the control
    // dependency: add a new identity node.
    string ctrl_dep_name = AddPrefixToNodeName(
        absl::StrCat(fanin.node(), "_", fanin.index()), kMutableGraphViewCtrl);

    NodeDef* ctrl_dep_node = GetNode(ctrl_dep_name);
    if (ctrl_dep_node == nullptr) {
      NodeDef new_node;
      new_node.set_name(ctrl_dep_name);
      new_node.set_op("Identity");
      new_node.set_device(fanin_node->device());
      (*new_node.mutable_attr())["T"].set_type(
          fanin_node->attr().at("T").type());
      new_node.add_input(TensorIdToString(fanin));
      ctrl_dep_node = AddNode(std::move(new_node));
    }
    return AddFaninInternal(node, {ctrl_dep_node, Graph::kControlSlot});
  }
}

bool MutableGraphView::RemoveControllingFaninInternal(NodeDef* node,
                                                      NodeDef* fanin_node) {
  for (int i = node->input_size() - 1; i >= 0; --i) {
    TensorId tensor_id = ParseTensorName(node->input(i));
    if (tensor_id.index() > Graph::kControlSlot) {
      break;
    }
    if (tensor_id.node() == fanin_node->name()) {
      fanouts()[{fanin_node, Graph::kControlSlot}].erase(
          {node, Graph::kControlSlot});
      node->mutable_input()->SwapElements(i, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
      return true;
    }
  }
  return false;
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

    auto& fanouts_set = fanouts()[fanin];
    fanouts_set.erase(input);
    UpdateMaxRegularOutputPortForRemovedFanin(fanin, fanouts_set);
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
