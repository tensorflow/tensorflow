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

#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

bool IsTrivialOp(const NodeDef& node, const GraphRewriter& rewriter) {
  // Remove the stop gradient nodes since they serve no purpose once the graph
  // is built. Also remove Identity ops.
  if (IsStopGradient(node)) {
    return true;
  }
  if (IsIdentity(node) || IsIdentityNSingleInput(node)) {
    return !(rewriter.FeedsMerge(node) || rewriter.IsDrivenBySwitch(node) ||
             rewriter.IsDrivenByControlDependency(node) ||
             rewriter.DrivesControlDependency(node));
  }

  return IsAddN(node) && NumNonControlInputs(node) <= 1;
}

absl::flat_hash_map<string, absl::flat_hash_set<int>> IdentityNTerminalPorts(
    const NodeMap& node_map, const std::vector<string>& terminal_nodes,
    int graph_size) {
  // Determines which ports for IdentityN nodes (that can be rewritten) lead to
  // a terminal node.
  std::vector<string> to_visit;
  to_visit.reserve(graph_size);
  // Set terminal nodes as visited so terminal nodes that may be IdentityN don't
  // get pruned later on.
  absl::flat_hash_set<string> visited(terminal_nodes.begin(),
                                      terminal_nodes.end());
  for (string terminal_node : terminal_nodes) {
    NodeDef* node = node_map.GetNode(terminal_node);
    if (node == nullptr) {
      continue;
    }
    for (string input : node->input()) {
      to_visit.push_back(input);
    }
  }

  absl::flat_hash_set<string> identity_n_fanouts;
  while (!to_visit.empty()) {
    string curr = to_visit.back();
    to_visit.pop_back();
    NodeDef* curr_node = node_map.GetNode(curr);
    if (curr_node == nullptr ||
        visited.find(curr_node->name()) != visited.end()) {
      continue;
    }
    // For IdentityN nodes, only traverse up through the port that comes from a
    // terminal node along with control inputs. The IdentityN node is not marked
    // as visited so other node input traversals can go through the other ports
    // of the IdentityN node.
    if (IsIdentityN(*curr_node)) {
      if (identity_n_fanouts.find(curr) == identity_n_fanouts.end()) {
        identity_n_fanouts.emplace(curr);
        int pos = NodePositionIfSameNode(curr, curr_node->name());
        if (pos >= 0) {
          to_visit.push_back(curr_node->input(pos));
        }
        for (const string& input : curr_node->input()) {
          if (IsControlInput(input) &&
              identity_n_fanouts.find(input) == identity_n_fanouts.end()) {
            to_visit.push_back(input);
          }
        }
      }
    } else {
      for (const string& input : curr_node->input()) {
        to_visit.push_back(input);
      }
      visited.emplace(curr_node->name());
    }
  }

  absl::flat_hash_map<string, absl::flat_hash_set<int>> identity_n_ports;
  for (const auto& fanout : identity_n_fanouts) {
    int pos;
    string node_name = ParseNodeName(fanout, &pos);
    if (node_name.empty() || pos < 0) {  // Exclude control inputs.
      continue;
    }
    if (identity_n_ports.find(node_name) == identity_n_ports.end()) {
      identity_n_ports[node_name] = {pos};
    } else {
      identity_n_ports[node_name].emplace(pos);
    }
  }

  return identity_n_ports;
}

string NewIdentityFromIdentityN(int pos, const NodeDef& identity_n,
                                GraphDef* graph, NodeMap* node_map) {
  // TODO(lyandy): Migrate over to GrapplerOptimizerStage and use
  // OptimizedNodeName for new node name.
  string new_node_name =
      strings::StrCat(identity_n.name(), "-", pos, "-grappler-ModelPruner");
  if (node_map->NodeExists(new_node_name)) {
    return "";
  }
  NodeDef* new_node = graph->add_node();
  Status status = NodeDefBuilder(new_node_name, "Identity")
                      .Input(identity_n.input(pos), 0,
                             identity_n.attr().at("T").list().type(pos))
                      .Device(identity_n.device())
                      .Finalize(new_node);
  if (!status.ok()) {
    return "";
  }
  node_map->AddNode(new_node->name(), new_node);
  node_map->AddOutput(NodeName(new_node->input(0)), new_node->name());
  return new_node->name();
}

Status RewriteIdentityNAndInputsOutputs(
    NodeDef* node, int num_non_control_inputs,
    const absl::flat_hash_set<int>& terminal_ports, GraphDef* graph,
    NodeMap* node_map) {
  // Rewrite IdentityN node and associated inputs and outputs. For inputs and
  // outputs that don't lead to a terminal node, a new Identity node is created
  // and those inputs and outputs are rewritten to use the new Identity node as
  // their outputs and inputs respectively. For the remaining nodes, the ouputs
  // have their inputs updated with the adjusted port, from the IdentityN node
  // having less inputs.
  struct NodeOutputUpdate {
    string input;
    string output;
  };

  absl::flat_hash_map<int, int> terminal_input_pos;
  absl::flat_hash_map<int, string> new_identities;
  int new_idx = 0;
  for (int i = 0; i < num_non_control_inputs; i++) {
    if (terminal_ports.find(i) != terminal_ports.end()) {
      terminal_input_pos[i] = new_idx++;
    } else {
      string identity = NewIdentityFromIdentityN(i, *node, graph, node_map);
      if (identity.empty()) {
        // Fail early when creating Identity from IdentityN errors.
        return errors::Internal(
            "Could not create Identity node from IdentityN node ", node->name(),
            " at port ", i);
      }
      new_identities[i] = identity;
    }
  }

  std::vector<NodeOutputUpdate> updates;
  for (NodeDef* output : node_map->GetOutputs(node->name())) {
    for (int i = 0; i < output->input_size(); i++) {
      string input = output->input(i);
      if (IsControlInput(input)) {
        continue;
      }
      TensorId input_tensor = ParseTensorName(input);
      if (input_tensor.node() == node->name()) {
        if (terminal_ports.find(input_tensor.index()) == terminal_ports.end()) {
          // Replace input that does not lead to a terminal node with newly
          // created identity.
          string new_identity = new_identities[input_tensor.index()];
          output->set_input(i, new_identity);
          updates.push_back({new_identity, output->name()});
        } else {
          // Update input ports that lead to a terminal node from splitting
          // inputs.
          int new_pos = terminal_input_pos[input_tensor.index()];
          string updated_input_name =
              new_pos > 0 ? strings::StrCat(node->name(), ":", new_pos)
                          : node->name();
          output->set_input(i, updated_input_name);
        }
      }
    }
  }

  for (NodeOutputUpdate update : updates) {
    node_map->AddOutput(update.input, update.output);
  }

  // Update inputs and types by removing inputs that were split away from
  // main IdentityN node.
  const int num_inputs = node->input_size();
  int curr_pos = 0;
  auto mutable_inputs = node->mutable_input();
  auto mutable_types =
      node->mutable_attr()->at("T").mutable_list()->mutable_type();
  for (int i = 0; i < num_non_control_inputs; i++) {
    if (terminal_input_pos.find(i) != terminal_input_pos.end()) {
      mutable_inputs->SwapElements(i, curr_pos);
      mutable_types->SwapElements(i, curr_pos);
      curr_pos++;
    }
  }
  mutable_types->Truncate(curr_pos);
  // Control inputs.
  for (int i = num_non_control_inputs; i < num_inputs; i++) {
    mutable_inputs->SwapElements(i, curr_pos++);
  }
  mutable_inputs->DeleteSubrange(curr_pos, num_inputs - curr_pos);

  return Status::OK();
}

Status SplitIdentityNInputs(GraphDef* graph,
                            const std::vector<string>& terminal_nodes,
                            bool* updated_graph) {
  // For inputs of IdentityN nodes that do not lead to a terminal node, remove
  // them from IdentityN and create new individual Identity nodes. This will
  // allow ModelPruner to possibly remove nodes in the transitive fanin of the
  // newly created Identity nodes.
  NodeMap node_map(graph);

  for (auto const& terminal :
       IdentityNTerminalPorts(node_map, terminal_nodes, graph->node_size())) {
    NodeDef* node = node_map.GetNode(terminal.first);
    if (node == nullptr) {
      continue;
    }

    const int num_non_control_inputs = NumNonControlInputs(*node);
    if (node->attr().count("T") == 0 ||
        node->attr().at("T").list().type_size() != num_non_control_inputs ||
        terminal.second.size() >= num_non_control_inputs) {
      continue;
    }

    TF_RETURN_IF_ERROR(RewriteIdentityNAndInputsOutputs(
        node, num_non_control_inputs, terminal.second, graph, &node_map));
    *updated_graph = true;
  }

  return Status::OK();
}

Status SetTransitiveFaninGraph(const GraphDef& input_graph,
                               GraphDef* output_graph,
                               const std::vector<string>& terminal_nodes) {
  // Determines transitive fanin nodes from terminal nodes and add them to the
  // output graph.
  bool ill_formed = false;
  std::vector<const NodeDef*> keep =
      ComputeTransitiveFanin(input_graph, terminal_nodes, &ill_formed);
  if (ill_formed) {
    // Some graph edges are invalid, or some of the feeds/fetch don't exist:
    // let's be conservative and preserve the graph as is.
    return errors::InvalidArgument("Invalid input graph.");
  }
  // Try to keep the nodes ordered somewhat topologically since this helps
  // further optimizations perform better.
  output_graph->mutable_node()->Reserve(keep.size());
  for (int i = keep.size() - 1; i >= 0; --i) {
    *output_graph->add_node() = *keep[i];
  }

  return Status::OK();
}

Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* pruned_graph) {
  const std::unordered_set<string> nodes_to_preserve = item.NodesToPreserve();

  // Prune all the nodes that won't be executed, ie all the nodes that aren't in
  // the fanin of a fetch node. If fetch nodes aren't specified, we'll assume
  // the whole graph might be executed.
  GrapplerItem runnable_item;
  if (!nodes_to_preserve.empty()) {
    std::vector<string> terminal_nodes(nodes_to_preserve.begin(),
                                       nodes_to_preserve.end());
    std::sort(terminal_nodes.begin(), terminal_nodes.end());
    TF_RETURN_IF_ERROR(SetTransitiveFaninGraph(item.graph, &runnable_item.graph,
                                               terminal_nodes));
    bool did_split_identity_n = false;
    TF_RETURN_IF_ERROR(SplitIdentityNInputs(
        &runnable_item.graph, terminal_nodes, &did_split_identity_n));
    if (did_split_identity_n) {
      GraphDef fanin_split_identity_n_graph;
      TF_RETURN_IF_ERROR(SetTransitiveFaninGraph(
          runnable_item.graph, &fanin_split_identity_n_graph, terminal_nodes));
      runnable_item.graph.Swap(&fanin_split_identity_n_graph);
    }
  } else {
    runnable_item = item;
  }

  GraphRewriter rewriter(runnable_item);

  // Check if we can further prune the graph, by removing the trivial ops.
  std::unordered_set<const NodeDef*> nodes_to_delete;
  for (auto& node : runnable_item.graph.node()) {
    if (!IsTrivialOp(node, rewriter)) {
      continue;
    }

    // Don't remove nodes that must be preserved.
    if (nodes_to_preserve.find(node.name()) != nodes_to_preserve.end()) {
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
    if (!rewriter.RemovalIncreasesEdgeCount(node) &&
        !rewriter.IsConnectedToFunction(node) &&
        !rewriter.IsDrivenByAnotherDevice(node) &&
        !rewriter.ReceivesRefValue(node)) {
      nodes_to_delete.insert(&node);
    }
  }

  pruned_graph->Clear();
  *pruned_graph->mutable_library() = item.graph.library();
  *pruned_graph->mutable_versions() = item.graph.versions();

  if (nodes_to_delete.empty()) {
    pruned_graph->mutable_node()->Swap(runnable_item.graph.mutable_node());
    return Status::OK();
  }

  const bool fetches_are_known = !item.fetch.empty();
  pruned_graph->mutable_node()->Reserve(runnable_item.graph.node_size());
  for (auto& node : runnable_item.graph.node()) {
    if (!fetches_are_known ||
        nodes_to_delete.find(&node) == nodes_to_delete.end()) {
      NodeDef* new_node = pruned_graph->add_node();
      *new_node = node;
      new_node->clear_input();
      rewriter.ForwardInputs(node, nodes_to_delete, new_node);
    }
  }
  VLOG(1) << "Pruned " << nodes_to_delete.size()
          << " nodes from the graph. The graph now contains "
          << pruned_graph->node_size() << " nodes.";
  CHECK_LE(pruned_graph->node_size(), item.graph.node_size());

  return Status::OK();
}

void ModelPruner::Feedback(Cluster* cluster, const GrapplerItem& item,
                           const GraphDef& pruned_graph, double result) {
  // Nothing to do for ModelPruner.
}

}  // end namespace grappler
}  // end namespace tensorflow
