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

#include "tensorflow/core/grappler/optimizers/model_pruner.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

bool IsTrivialIdentity(const utils::MutableNodeView& node_view) {
  if (node_view.NumControllingFanins() > 0) {
    // Node is driven by control dependency.
    return false;
  }
  if (node_view.NumControlledFanouts() > 0) {
    // Node drives control dependency.
    return false;
  }
  for (const auto& regular_fanin : node_view.GetRegularFanins()) {
    if (IsSwitch(*regular_fanin.node_view()->node())) {
      // Node is driven by switch.
      return false;
    }
  }
  for (const auto& regular_fanouts : node_view.GetRegularFanouts()) {
    for (const auto& regular_fanout : regular_fanouts) {
      if (IsMerge(*regular_fanout.node_view()->node())) {
        // Node feeds merge.
        return false;
      }
    }
  }
  return true;
}

bool IsTrivialOp(const utils::MutableNodeView& node_view) {
  // Remove the stop gradient nodes since they serve no purpose once the graph
  // is built. Also remove Identity ops.
  const auto* node = node_view.node();
  if (IsStopGradient(*node)) {
    return true;
  }
  if (IsIdentity(*node) || IsIdentityNSingleInput(*node)) {
    return IsTrivialIdentity(node_view);
  }
  const bool no_fanins = node_view.NumRegularFanins() == 0 &&
                         node_view.NumControllingFanins() == 0;
  if (IsNoOp(*node) && no_fanins) {
    return true;
  }
  // Const nodes are always executed before anything else, so if they only
  // have control outputs we can remove them.
  if (IsConstant(*node) && no_fanins && node_view.NumRegularFanouts() == 0) {
    return true;
  }
  return IsAddN(*node) && node_view.NumRegularFanins() <= 1;
}

bool RemovalIncreasesEdgeCount(const utils::MutableNodeView& node_view) {
  int in_degree =
      node_view.NumRegularFanins() + node_view.NumControllingFanins();
  int out_degree =
      node_view.NumRegularFanouts() + node_view.NumControlledFanouts();
  return in_degree * out_degree > in_degree + out_degree;
}

bool IsOutputPortRefValue(const NodeDef& node, int port_id,
                          const OpRegistryInterface& op_registry) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = op_registry.LookUp(node.op(), &op_reg_data);
  if (s.ok()) {
    DataType output_type;
    s = OutputTypeForNode(node, op_reg_data->op_def, port_id, &output_type);
    if (s.ok() && IsRefType(output_type)) {
      return true;
    }
  }
  return false;
}

bool CanRemoveNode(const utils::MutableNodeView& node_view,
                   const absl::flat_hash_set<absl::string_view>& function_names,
                   const OpRegistryInterface& op_registry) {
  const auto* node = node_view.node();
  const bool no_fanins = node_view.NumRegularFanins() == 0 &&
                         node_view.NumControllingFanins() == 0;
  if (IsNoOp(*node) && no_fanins) {
    return true;
  }
  if (IsConstant(*node) && no_fanins && node_view.NumRegularFanouts() == 0) {
    return true;
  }
  if (RemovalIncreasesEdgeCount(node_view)) {
    return false;
  }
  const string& device = node->device();
  for (const auto& regular_fanin : node_view.GetRegularFanins()) {
    auto* fanin_node = regular_fanin.node_view()->node();
    if (fanin_node->device() != device) {
      // Node is driven by a different device.
      return false;
    } else if (function_names.contains(fanin_node->op())) {
      // Node input is a function call.
      return false;
    } else if (IsOutputPortRefValue(*fanin_node, regular_fanin.index(),
                                    op_registry)) {
      return false;
    }
  }
  for (const auto& controlling_fanin : node_view.GetControllingFanins()) {
    if (controlling_fanin.node_view()->GetDevice() != device) {
      // Node is driven by a different device.
      return false;
    }
  }
  for (const auto& regular_fanouts : node_view.GetRegularFanouts()) {
    for (const auto& regular_fanout : regular_fanouts) {
      if (function_names.contains(regular_fanout.node_view()->GetOp())) {
        // Node output is a function call.
        return false;
      }
    }
  }
  return true;
}

// ForwardFanins forwards fanins of a node to be removed to its fanouts. This
// currently is specific to nodes defined in `IsTrivialOp` under the assumption
// they can have at most one regular fanin (at index 0) and one regular fanout
// index (at 0).
//
// The forwarding is as follows:
// * If the node to be removed has a regular fanin (at index 0), that fanin will
//   be forwarded by replacing the fanin (consisting of the node being removed)
//   in each regular fanout (at index 0) of the node being removed with the
//   node being removed's regular fanin (at index 0). If the node being removed
//   also has controlling fanins, those controlling fanins are added to each
//   regular fanout (at index 0) as a controlling fanin.
// * If the node to be removed has controlled fanouts, each controlling fanin of
//   the node is added and the node to be removed (as a control dependency) is
//   removed from each controlled fanout. If the node also has a regular fanin
//   (at index 0), that fanin as a control dependency is added to each
//   controlled fanout.
// TODO(lyandy): Move this to a shared util for GraphView.
Status ForwardFanins(utils::MutableGraphView* graph_view, int node_to_delete,
                     absl::flat_hash_set<int>* mutated_fanouts) {
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  auto* node_view = graph_view->GetNode(node_to_delete);
  std::vector<absl::string_view> controlling_fanin_names;
  controlling_fanin_names.reserve(node_view->NumControllingFanins());
  for (const auto& controlling_fanin : node_view->GetControllingFanins()) {
    controlling_fanin_names.push_back(controlling_fanin.node_view()->GetName());
  }
  const auto& node_regular_fanin_0 = node_view->GetRegularFanin(0);
  const bool has_regular_fanin_0 = node_view->NumRegularFanins() >= 1;
  const string regular_fanin_0_name =
      has_regular_fanin_0 ? node_regular_fanin_0.node_view()->GetName() : "";

  // Forward to regular fanouts.
  if (has_regular_fanin_0) {
    TensorId tensor_id(regular_fanin_0_name, node_regular_fanin_0.index());
    for (const auto& fanout : node_view->GetRegularFanout(0)) {
      auto* fanout_node_view = fanout.node_view();
      mutation->AddOrUpdateRegularFanin(fanout_node_view, fanout.index(),
                                        tensor_id);
      for (const auto& controlling_fanin : controlling_fanin_names) {
        mutation->AddControllingFanin(fanout_node_view, controlling_fanin);
      }
      mutated_fanouts->emplace(fanout.node_index());
    }
  }

  // Forward to controlled fanouts.
  for (const auto& controlled_fanout : node_view->GetControlledFanouts()) {
    auto* fanout_node_view = controlled_fanout.node_view();
    mutation->RemoveControllingFanin(fanout_node_view, node_view->GetName());
    if (has_regular_fanin_0) {
      mutation->AddControllingFanin(fanout_node_view, regular_fanin_0_name);
    }
    for (const auto& controlling_fanin : controlling_fanin_names) {
      mutation->AddControllingFanin(fanout_node_view, controlling_fanin);
    }
    mutated_fanouts->emplace(controlled_fanout.node_index());
  }

  return mutation->Apply();
}

absl::flat_hash_map<int, std::vector<bool>> IdentityNTerminalPorts(
    const utils::MutableGraphView& graph_view,
    absl::Span<const int> terminal_nodes) {
  // Determines which ports for IdentityN nodes (that can be rewritten) lead to
  // a terminal node.
  std::vector<utils::MutableFanoutView> to_visit;
  to_visit.reserve(graph_view.NumNodes());
  // Set terminal nodes as visited so terminal nodes that may be IdentityN don't
  // get pruned later on.
  absl::flat_hash_set<int> visited(terminal_nodes.begin(),
                                   terminal_nodes.end());
  for (const auto& terminal_node : terminal_nodes) {
    const auto* node = graph_view.GetNode(terminal_node);
    for (const auto& regular_fanin : node->GetRegularFanins()) {
      to_visit.push_back(regular_fanin);
    }
    for (const auto& controlling_fanin : node->GetRegularFanins()) {
      to_visit.push_back(controlling_fanin);
    }
  }

  absl::flat_hash_set<utils::MutableFanoutView> identity_n_fanouts;
  while (!to_visit.empty()) {
    const auto curr = to_visit.back();
    to_visit.pop_back();
    const auto* curr_node = curr.node_view();
    if (visited.contains(curr_node->node_index())) {
      continue;
    }
    // For IdentityN nodes, only traverse up through the port that comes from a
    // terminal node along with control inputs. The IdentityN node is not marked
    // as visited so other node input traversals can go through the other ports
    // of the IdentityN node.
    if (IsIdentityN(*curr_node->node())) {
      if (!identity_n_fanouts.contains(curr)) {
        identity_n_fanouts.emplace(curr);
        const int pos = curr.index();
        if (pos >= 0) {
          to_visit.push_back(curr_node->GetRegularFanin(pos));
        }
        for (const auto& controlling_fanin :
             curr_node->GetControllingFanins()) {
          if (!identity_n_fanouts.contains(controlling_fanin)) {
            to_visit.push_back(controlling_fanin);
          }
        }
      }
    } else {
      for (const auto& regular_fanin : curr_node->GetRegularFanins()) {
        to_visit.push_back(regular_fanin);
      }
      for (const auto& controlling_fanin : curr_node->GetRegularFanins()) {
        to_visit.push_back(controlling_fanin);
      }
      visited.emplace(curr_node->node_index());
    }
  }

  absl::flat_hash_map<int, std::vector<bool>> identity_n_ports;
  for (const auto& fanout : identity_n_fanouts) {
    if (fanout.index() == Graph::kControlSlot) {  // Exclude control inputs.
      continue;
    }
    const auto* fanout_node_view = fanout.node_view();
    auto& ports = identity_n_ports[fanout_node_view->node_index()];
    if (ports.empty()) {
      ports.resize(fanout_node_view->NumRegularFanins());
    }
    ports[fanout.index()] = true;
  }

  return identity_n_ports;
}

string AddNewIdentityFromIdentityN(utils::MutableGraphView* graph_view,
                                   const utils::MutableNodeView& node_view,
                                   const AttrValue* node_type_attr, int pos) {
  // TODO(lyandy): Migrate over to GrapplerOptimizerStage and use
  // OptimizedNodeName for new node name.
  string new_node_name =
      strings::StrCat(node_view.GetName(), "-", pos, "-grappler-ModelPruner");
  if (graph_view->HasNode(new_node_name)) {
    return "";
  }

  // This node will be pruned away so there is no need to add/forward control
  // dependencies.
  NodeDef new_node;
  new_node.set_name(new_node_name);
  new_node.set_op("Identity");
  new_node.set_device(node_view.GetDevice());
  new_node.add_input(node_view.node()->input(pos));

  AttrValue attr;
  attr.set_type(node_type_attr->list().type(pos));
  new_node.mutable_attr()->insert({"T", attr});

  Status status;
  graph_view->GetMutationBuilder()->AddNode(std::move(new_node), &status);
  if (!status.ok()) {
    return "";
  }

  return new_node_name;
}

Status ForwardFaninToFanouts(utils::MutableGraphView* graph_view,
                             utils::MutableNodeView* node_view, int pos,
                             const TensorId& fanin) {
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  auto& fanouts_at_pos = node_view->GetRegularFanout(pos);
  for (auto& fanout : fanouts_at_pos) {
    mutation->AddOrUpdateRegularFanin(fanout.node_view(), fanout.index(),
                                      fanin);
  }
  return Status::OK();
}

Status RewriteIdentityNAndInputsOutputs(
    utils::MutableGraphView* graph_view, utils::MutableNodeView* node_view,
    const std::vector<bool>& terminal_ports) {
  // Rewrite IdentityN node and associated inputs and outputs. For inputs and
  // outputs that don't lead to a terminal node, a new Identity node is created
  // and those inputs and outputs are rewritten to use the new Identity node as
  // their outputs and inputs respectively. For the remaining nodes, the ouputs
  // have their inputs updated with the adjusted port, from the IdentityN node
  // having less inputs.
  auto* identity_n_types = node_view->GetAttr("T");
  if (identity_n_types == nullptr) {
    return errors::Internal("IdentityN node '", node_view->GetName(),
                            "' is missing attribute 'T'");
  }
  AttrValue types(*identity_n_types);

  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  int new_idx = 0;
  const int num_regular_fanins = node_view->NumRegularFanins();
  for (int i = 0; i < num_regular_fanins; i++) {
    if (terminal_ports[i]) {
      if (i > new_idx) {
        const auto& fanin_at_i = node_view->GetRegularFanin(i);
        mutation->AddOrUpdateRegularFanin(
            node_view, new_idx,
            {fanin_at_i.node_view()->GetName(), fanin_at_i.index()});
        TF_RETURN_IF_ERROR(ForwardFaninToFanouts(
            graph_view, node_view, i, {node_view->GetName(), new_idx}));
        types.mutable_list()->mutable_type()->SwapElements(i, new_idx);
      }
      new_idx++;
    } else {
      string identity = AddNewIdentityFromIdentityN(graph_view, *node_view,
                                                    identity_n_types, i);
      if (identity.empty()) {
        // Fail early when creating Identity from IdentityN errors.
        return errors::Internal(
            "Could not create Identity node from IdentityN node '",
            node_view->GetName(), "' at port ", i);
      }
      TF_RETURN_IF_ERROR(
          ForwardFaninToFanouts(graph_view, node_view, i, {identity, 0}));
    }
  }

  if (new_idx < num_regular_fanins) {
    for (int i = new_idx; i < num_regular_fanins; ++i) {
      mutation->RemoveRegularFanin(node_view, i);
    }
    types.mutable_list()->mutable_type()->Truncate(new_idx);
    mutation->AddOrUpdateNodeAttr(node_view, "T", types);
    return mutation->Apply();
  }

  return Status::OK();
}

std::vector<int> GetTerminalNodeIndices(
    const utils::MutableGraphView& graph_view,
    const absl::flat_hash_set<absl::string_view>& nodes_to_preserve,
    Status* s) {
  std::vector<int> node_indices;
  node_indices.reserve(nodes_to_preserve.size());

  for (const auto& node_to_preserve : nodes_to_preserve) {
    const auto* node = graph_view.GetNode(node_to_preserve);
    if (node != nullptr) {
      node_indices.push_back(node->node_index());
    } else {
      *s = errors::Internal("Could not find node with name '", node_to_preserve,
                            "'");
      return {};
    }
  }
  *s = Status::OK();
  return node_indices;
}

Status SplitIdentityNInputs(
    utils::MutableGraphView* graph_view,
    const absl::flat_hash_set<absl::string_view>& nodes_to_preserve,
    bool* updated_graph) {
  Status status;
  std::vector<int> terminal_nodes =
      GetTerminalNodeIndices(*graph_view, nodes_to_preserve, &status);
  TF_RETURN_IF_ERROR(status);

  // For inputs of IdentityN nodes that do not lead to a terminal node, remove
  // them from IdentityN and create new individual Identity nodes. This will
  // allow ModelPruner to possibly remove nodes in the transitive fanin of the
  // newly created Identity nodes.
  auto terminal_ports = IdentityNTerminalPorts(*graph_view, terminal_nodes);
  for (auto const& terminal : terminal_ports) {
    auto* node = graph_view->GetNode(terminal.first);

    const int num_regular_fanins = node->NumRegularFanins();
    auto* t_attr = node->GetAttr("T");
    if (t_attr == nullptr || t_attr->list().type_size() != num_regular_fanins ||
        terminal.second.size() != num_regular_fanins) {
      continue;
    }

    TF_RETURN_IF_ERROR(
        RewriteIdentityNAndInputsOutputs(graph_view, node, terminal.second));
    *updated_graph = true;
  }

  return Status::OK();
}

std::vector<bool> ComputeTransitiveFanin(
    const utils::MutableGraphView& graph_view,
    absl::Span<const int> terminal_nodes) {
  absl::flat_hash_map<absl::string_view, int> name_to_send;
  for (const auto& node : graph_view.GetNodes()) {
    if (node.GetOp() == "_Send") {
      name_to_send[node.GetAttr("tensor_name")->s()] = node.node_index();
    }
  }

  std::vector<int> queue;
  queue.insert(queue.end(), terminal_nodes.begin(), terminal_nodes.end());

  std::vector<bool> result;
  result.resize(graph_view.NumNodes(), false);

  while (!queue.empty()) {
    const int node_index = queue.back();
    queue.pop_back();
    if (result[node_index]) {
      // The node has already been visited.
      continue;
    }
    result[node_index] = true;
    const auto* node = graph_view.GetNode(node_index);
    for (const auto& regular_fanin : node->GetRegularFanins()) {
      queue.push_back(regular_fanin.node_index());
    }
    for (const auto& controlling_fanin : node->GetControllingFanins()) {
      queue.push_back(controlling_fanin.node_index());
    }
    if (node->GetOp() == "_Recv") {
      auto it = name_to_send.find(node->GetAttr("tensor_name")->s());
      if (it != name_to_send.end()) {
        queue.push_back(it->second);
      }
      // Subgraph after partitioning may have either _Send or _Recv, not both.
    }
  }

  return result;
}

// TODO(lyandy): Move this to a shared util for GraphView.
Status PruneUnreachableNodes(
    utils::MutableGraphView* graph_view,
    const absl::flat_hash_set<absl::string_view>& nodes_to_preserve) {
  Status status;
  std::vector<int> terminal_nodes =
      GetTerminalNodeIndices(*graph_view, nodes_to_preserve, &status);
  TF_RETURN_IF_ERROR(status);
  std::vector<bool> nodes_to_keep =
      ComputeTransitiveFanin(*graph_view, terminal_nodes);
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    if (!nodes_to_keep[i]) {
      mutation->RemoveNode(graph_view->GetNode(i));
    }
  }

  return mutation->Apply();
}

// TODO(lyandy): Move this to a shared util for GraphView.
Status DedupNodeControlDependencies(utils::MutableGraphView* graph_view,
                                    int node_index) {
  auto* node_view = graph_view->GetNode(node_index);

  std::vector<bool> regular_nodes;
  regular_nodes.resize(graph_view->NumNodes());
  for (const auto& regular_fanin : node_view->GetRegularFanins()) {
    regular_nodes[regular_fanin.node_view()->node_index()] = true;
  }

  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  for (const auto& controlling_fanin : node_view->GetControllingFanins()) {
    auto* controlling_fanin_node = controlling_fanin.node_view();
    if (regular_nodes[controlling_fanin_node->node_index()]) {
      mutation->RemoveControllingFanin(node_view,
                                       controlling_fanin_node->GetName());
    }
  }

  return mutation->Apply();
}

Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* optimized_graph) {
  const auto preserve_set = item.NodesToPreserve();
  const absl::flat_hash_set<absl::string_view> nodes_to_preserve(
      preserve_set.begin(), preserve_set.end());

  // Prune all the nodes that won't be executed, ie all the nodes that aren't in
  // the fanin of a fetch node. If fetch nodes aren't specified, we'll assume
  // the whole graph might be executed.
  GraphDef graph = item.graph;
  Status status;
  utils::MutableGraphView graph_view(&graph, &status);
  TF_RETURN_IF_ERROR(status);
  if (!nodes_to_preserve.empty()) {
    TF_RETURN_IF_ERROR(PruneUnreachableNodes(&graph_view, nodes_to_preserve));
    bool did_split_identity_n = false;
    TF_RETURN_IF_ERROR(SplitIdentityNInputs(&graph_view, nodes_to_preserve,
                                            &did_split_identity_n));
    if (did_split_identity_n) {
      TF_RETURN_IF_ERROR(PruneUnreachableNodes(&graph_view, nodes_to_preserve));
    }
    // TODO(lyandy): Remove sorting once ArithmeticOptimizer
    // (MinimizeBroadcasts) is migrated over to using utils::GraphView.
    TF_RETURN_IF_ERROR(
        graph_view.SortTopologically(/*ignore_cycles=*/true, {}));
  }

  absl::flat_hash_set<absl::string_view> function_names;
  function_names.reserve(graph.library().function_size());
  for (const auto& function : graph.library().function()) {
    function_names.insert(function.signature().name());
  }
  OpRegistryInterface* op_registry = OpRegistry::Global();

  // Check if we can further prune the graph, by removing the trivial ops.
  const int num_nodes = graph_view.NumNodes();
  std::vector<int> nodes_to_delete;
  nodes_to_delete.reserve(num_nodes);
  for (auto& node : graph_view.GetNodes()) {
    if (!IsTrivialOp(node)) {
      continue;
    }

    // Don't remove nodes that must be preserved.
    if (nodes_to_preserve.contains(node.GetName())) {
      continue;
    }

    // - Don't remove nodes that drive control dependencies.
    // - Don't remove nodes that are driven by control dependencies either since
    //   we can't ensure (yet) that we won't increase the number of control
    //   dependency edges by deleting them (for example, removing a node driven
    //   by 10 control edges and driving 10 control edges would result in the
    //   creation of 100 edges).
    // - Don't modify nodes that are connected to functions since that can
    //   result in inlining failures later on.
    // - Don't prune nodes that are driven by another device since these could
    //   be used to reduce cross device communication.
    // - Don't remove nodes that receive reference values, as those can be
    //   converting references to non-references. It is important to preserve
    //   these non-references since the partitioner will avoid sending
    //   non-references across partitions more than once.
    if (CanRemoveNode(node, function_names, *op_registry)) {
      nodes_to_delete.push_back(node.node_index());
    }
  }

  if (!nodes_to_delete.empty()) {
    absl::flat_hash_set<int> mutated_fanouts;
    for (const int node_to_delete : nodes_to_delete) {
      TF_RETURN_IF_ERROR(
          ForwardFanins(&graph_view, node_to_delete, &mutated_fanouts));
    }

    for (const int mutated_fanout : mutated_fanouts) {
      TF_RETURN_IF_ERROR(
          DedupNodeControlDependencies(&graph_view, mutated_fanout));
    }

    if (!item.fetch.empty()) {
      utils::Mutation* mutation = graph_view.GetMutationBuilder();
      for (const int node_to_delete : nodes_to_delete) {
        mutation->RemoveNode(graph_view.GetNode(node_to_delete));
      }
      TF_RETURN_IF_ERROR(mutation->Apply());
      VLOG(1) << "Pruned " << num_nodes - graph.node_size()
              << " nodes from the graph. The graph now contains "
              << graph.node_size() << " nodes.";
    }

    if (graph.node_size() > item.graph.node_size()) {
      return errors::Internal("Pruning increased graph size.");
    }
  }

  *optimized_graph = graph;

  return Status::OK();
}

void ModelPruner::Feedback(Cluster* cluster, const GrapplerItem& item,
                           const GraphDef& optimized_graph, double result) {
  // Nothing to do for ModelPruner.
}

}  // end namespace grappler
}  // end namespace tensorflow
