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

#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

// Builds a map from the &graph->node(i) to i.
absl::flat_hash_map<const NodeDef*, int> BuildNodeToIdx(const GraphDef& graph) {
  // Set up &node -> index map.
  absl::flat_hash_map<const NodeDef*, int> node_to_idx;
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& node = graph.node(i);
    node_to_idx[&node] = i;
  }
  return node_to_idx;
}

}  // namespace

bool DependencyOptimizer::SafeToRemoveIdentity(const NodeDef& node) const {
  if (!IsIdentity(node) && !IsIdentityN(node)) {
    return true;
  }

  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (!fetch_nodes_known_) {
    // The output values of this node may be needed.
    return false;
  }

  MutableGraphView::OutputPort port = graph_view_->GetRegularFanin(
      MutableGraphView::InputPort(const_cast<NodeDef*>(&node), 0));
  NodeDef* input = port.node;
  CHECK(input != nullptr) << "node = " << node.name()
                          << " input = " << node.input(0);
  // Don't remove Identity nodes corresponding to Variable reads or following
  // Recv.
  if (IsVariable(*input) || IsRecv(*input)) {
    return false;
  } else if (IsSwitch(*input)) {
    // Don't turn Identity nodes following Switch into NoOp or remove them
    // if it requires anchoring a control dependencies the Switch node, which
    // is not valid.
    MutableGraphView::OutputPort control_port(const_cast<NodeDef*>(&node),
                                              Graph::kControlSlot);
    if (!graph_view_->GetFanout(control_port).empty()) {
      return false;
    }
  }
  bool node_has_multiple_inputs =
      graph_view_->NumFanins(node, /*include_controlling_nodes=*/true) > 1;
  for (auto consumer :
       graph_view_->GetFanouts(node, /*include_controlled_nodes=*/true)) {
    if (node_has_multiple_inputs && IsMerge(*consumer.node)) {
      return false;
    }
    if (IsSwitch(*input)) {
      if (graph_view_->HasFanin(*consumer.node,
                                {node.name(), Graph::kControlSlot})) {
        return false;
      }
    }
  }
  return true;
}

bool DependencyOptimizer::SafeToConvertToNoOp(const NodeDef& node) const {
  if (!fetch_nodes_known_ ||
      nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (IsMerge(node) || IsSwitch(node) || ModifiesFrameInfo(node) ||
      !IsFreeOfSideEffect(node)) {
    return false;
  }
  if (node.op().rfind("Submodel", 0) == 0) {
    return false;
  }
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok() || op_def->output_arg_size() == 0) {
    return false;
  }
  const std::unordered_set<string> do_not_rewrite_ops{
      "Assert",     "CheckNumerics",         "_Retval",
      "_Arg",       "_ParallelConcatUpdate", "TPUExecute",
      "TPUCompile", "ControlTrigger"};
  if (do_not_rewrite_ops.find(node.op()) != do_not_rewrite_ops.end()) {
    return false;
  }
  if (!SafeToRemoveIdentity(node)) {
    return false;
  }
  if (graph_view_->NumFanouts(node, /*include_controlled_nodes=*/false) > 0) {
    // The output values of this node may be needed.
    return false;
  }
  return true;
}

int DependencyOptimizer::NumEdgesIfBypassed(
    const NodeDef& node, int num_fanins,
    const absl::flat_hash_set<MutableGraphView::Edge>& fanout_edges) const {
  const bool is_multi_input_identity_n =
      IsIdentityN(node) && !IsIdentityNSingleInput(node);
  const int num_fanouts = fanout_edges.size();

  if (is_multi_input_identity_n) {
    // multi-input identity_n with input/output control dependencies will likely
    // increase number of edges after optimization.
    int num_edges_if_bypassed(0);
    int num_non_controlling_fanins =
        graph_view_->NumFanins(node, /*include_controlling_nodes=*/false);
    num_edges_if_bypassed += num_non_controlling_fanins;
    num_edges_if_bypassed +=
        (num_fanins - num_non_controlling_fanins) * num_fanouts;

    for (const auto& fanout : fanout_edges) {
      if (fanout.dst.port_id == Graph::kControlSlot) {
        num_edges_if_bypassed += num_fanins;
      } else {
        ++num_edges_if_bypassed;
      }
    }
    return num_edges_if_bypassed;
  } else {
    return num_fanins * num_fanouts;
  }
}

bool DependencyOptimizer::BypassingNodeIsBeneficial(
    const NodeDef& node,
    const absl::flat_hash_set<MutableGraphView::OutputPort>& fanins,
    const absl::flat_hash_set<MutableGraphView::Edge>& fanout_edges) const {
  const bool is_identity = IsIdentity(node) || IsIdentityNSingleInput(node);
  const bool is_multi_input_identity_n =
      IsIdentityN(node) && !IsIdentityNSingleInput(node);
  const int num_outputs = fanout_edges.size();
  const int num_inputs = fanins.size();

  if (NumEdgesIfBypassed(node, num_inputs, fanout_edges) >
      num_inputs + num_outputs) {
    return false;
  }

  // Make sure that we don't increase the number of edges that cross
  // device boundaries.
  if ((num_inputs == 1 && num_outputs > 1 &&
       fanins.begin()->node->device() != node.device()) ||
      (num_inputs > 1 && num_outputs == 1 &&
       fanout_edges.begin()->dst.node->device() != node.device())) {
    return false;
  }

  // TODO(rmlarsen): Not all device crossings are equally expensive.
  // Assign a cost to each based on device affinity and compute a
  // cost before and after.
  const string& node_dev = node.device();
  int num_cross_in = 0;
  for (const auto& fanin : fanins) {
    num_cross_in += static_cast<int>(fanin.node->device() != node_dev);
  }
  int num_cross_out = 0;
  for (const auto& fanout : fanout_edges) {
    num_cross_out += static_cast<int>(fanout.dst.node->device() != node_dev);
  }

  if ((is_identity || is_multi_input_identity_n) && num_cross_in > 0 &&
      num_cross_out > 0) {
    // This identity node follows a device crossing, so it might be
    // following a _Recv node after partioning. Do not remove such nodes,
    // unless they only have consumers on the same device as themselves.
    return false;
  }

  // Make sure we do not increase the number of device crossings.
  const int num_cross_before = num_cross_in + num_cross_out;
  int num_cross_after = 0;
  for (const auto& fanin : fanins) {
    for (const auto& fanout : fanout_edges) {
      num_cross_after +=
          static_cast<int>(fanin.node->device() != fanout.dst.node->device());
    }
  }
  if (num_cross_after > num_cross_before) {
    return false;
  }
  return true;
}

void DependencyOptimizer::OptimizeNode(NodeDef* node,
                                       SetVector<NodeDef*>* nodes_to_simplify,
                                       std::set<string>* nodes_to_delete) {
  const string node_name = node->name();
  const bool is_noop = IsNoOp(*node);
  const bool is_identity = IsIdentity(*node) || IsIdentityNSingleInput(*node);
  const bool is_multi_input_identity =
      IsIdentityN(*node) && !IsIdentityNSingleInput(*node);
  // Constant nodes with no input control dependency are always executed early,
  // so we can prune all their output control dependencies.
  if (IsConstant(*node) &&
      graph_view_->NumFanins(*node, /*include_controlling_nodes=*/true) == 0) {
    for (const auto& fanout :
         graph_view_->GetFanouts(*node, /*include_controlled_nodes=*/true)) {
      if (graph_view_->RemoveFanin(fanout.node->name(),
                                   {node_name, Graph::kControlSlot})) {
        nodes_to_simplify->PushBack(fanout.node);
      }
    }

    if (graph_view_->NumFanouts(*node, /*include_controlled_nodes=*/true) ==
            0 &&
        fetch_nodes_known_ &&
        nodes_to_preserve_.find(node_name) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node->name());
    }
    return;
  }

  // Change ops that only have control dependencies as outputs to NoOps.
  if (!is_noop && SafeToConvertToNoOp(*node)) {
    VLOG(1) << "***** Replacing  " << node_name << " (" << node->op()
            << ") with NoOp.";
    // The outputs of this node are not consumed. Replace its inputs with
    // control dependencies and replace the op itself with the NoOp op.
    int num_non_controlling_fanins =
        graph_view_->NumFanins(*node, /*include_controlling_nodes=*/false);
    if (num_non_controlling_fanins > 0) {
      absl::flat_hash_set<string> non_controlling_fanins(
          node->input().begin(),
          node->input().begin() + num_non_controlling_fanins);
      graph_view_->RemoveAllFanins(node_name,
                                   /*keep_controlling_fanins=*/true);
      for (const string& fanin : non_controlling_fanins) {
        TensorId tensor_id = ParseTensorName(fanin);
        graph_view_->AddControllingFanin(node_name, tensor_id);
        nodes_to_simplify->PushBack(graph_view_->GetNode(tensor_id.node()));
      }
    }
    node->set_op("NoOp");
    node->clear_attr();
    nodes_to_simplify->PushBack(node);
    return;
  }

  // Remove NoOp nodes if the product of their fan-in and fan-out is less than
  // or equal to the sum of the fan-in and fan-out. The non-trivial rewrites
  // take the following form:
  //
  // Case a)
  //    x --^> +------+                x --^> +---+
  //    y --^> | NoOp | --^> a   ==>   y --^> | a |
  //    ...    |      |                  ...  |   |
  //    z --^> +------+                z --^> +---+
  //
  // Case b)
  //           +------+ --^> a         +---+ --^> a
  //    x --^> | NoOp | --^> b  ==>    | x | --^> b
  //           |      | ...            |   | ...
  //           +------+ --^> c         +---+ --^> c
  // Case c)
  //           +------+                x ---^> a
  //    x --^> | NoOp | --^> a  ==>      \/
  //    y --^> |      | --^> b           /\
  //           +------+                y ---^> b
  //
  // We only apply this optimization if we don't increase the number of control
  // edges across device boundaries, e.g. in cases a) and b) if NoOp and
  // a and x, respectively, are on the same device. Control edges across device
  // boundaries require inter-device communication (Send/Recv pairs to be
  // inserted in the graph), which is very costly.
  //
  // We also remove identity nodes, subject to the same constraints on number of
  // resulting control edges and device boundary crossings:
  //
  // Case a)
  //          +----------+ ---> a       +---+ ---> a
  //    x --> | Identity | --^> b  ==>  | x | --^> b
  //          |          | ...          |   | ...
  //          +----------+ --^> c       +---+ --^> c
  //
  // Case b)
  //    x ---> +----------+ ---> a      x ---> +---+
  //    y --^> | Identity |        ==>  y --^> | a |
  //    ...    |          |               ...  |   |
  //    z --^> +----------+             z --^> +---+
  //
  // Case c)
  //           +----------+             x ---> +---+
  //    x ---> | Identity | ---> a ==>   \--^> | a |
  //    y --^> |          | --^> b       /\    +---+
  //           +----------+             y --^> b

  if (is_noop || ((is_identity || is_multi_input_identity) &&
                  SafeToRemoveIdentity(*node))) {
    auto fanins =
        graph_view_->GetFanins(*node, /*include_controlling_nodes=*/true);
    auto fanout_edges =
        graph_view_->GetFanoutEdges(*node, /*include_controlled_edges=*/true);

    if (!BypassingNodeIsBeneficial(*node, fanins, fanout_edges)) {
      return;
    }

    int num_non_controlling_fanins =
        graph_view_->NumFanins(*node, /*include_controlling_nodes=*/false);
    VLOG(1) << "***** Rerouting input around\n" << node->DebugString();
    // Now remove the node and re-wire its inputs to its outputs.
    for (auto fanout_edge : fanout_edges) {
      bool updated_consumer = false;
      NodeDef* consumer = fanout_edge.dst.node;
      VLOG(1) << "consumer before:\n" << consumer->DebugString();
      if (fanout_edge.src.port_id == Graph::kControlSlot) {
        updated_consumer = graph_view_->RemoveFanin(
            consumer->name(), {node_name, Graph::kControlSlot});
        if (updated_consumer) {
          // Add all non controlling fanins of node as controlling fanins to
          // consumer.
          for (int i = 0; i < num_non_controlling_fanins; ++i) {
            TensorId input = ParseTensorName(node->input(i));
            if (graph_view_->AddControllingFanin(consumer->name(), input)) {
              nodes_to_simplify->PushBack(graph_view_->GetNode(input.node()));
            }
          }
        }
      } else {
        updated_consumer = graph_view_->UpdateFanin(
            consumer->name(), {node_name, fanout_edge.src.port_id},
            ParseTensorName(node->input(fanout_edge.src.port_id)));
      }
      if (updated_consumer) {
        // Forward all controlling fanins of node to consumer.
        for (int i = num_non_controlling_fanins; i < node->input_size(); ++i) {
          TensorId input = ParseTensorName(node->input(i));
          if (graph_view_->AddFanin(consumer->name(), input)) {
            nodes_to_simplify->PushBack(graph_view_->GetNode(input.node()));
          }
        }
        nodes_to_simplify->PushBack(consumer);
      }
      VLOG(1) << "consumer after:\n" << consumer->DebugString();
    }
    if (fetch_nodes_known_ &&
        nodes_to_preserve_.find(node_name) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node_name);

      // Disconnect the node from its inputs to enable further optimizations.
      graph_view_->RemoveAllFanins(node_name,
                                   /*keep_controlling_fanins=*/false);
    }
  }
}

Status DependencyOptimizer::OptimizeDependencies() {
  SetVector<NodeDef*> nodes_to_simplify;
  std::set<string> nodes_to_delete;
  for (int i = 0; i < graph_view_->graph()->node_size(); ++i) {
    NodeDef* node = graph_view_->graph()->mutable_node(i);
    if (IsNoOp(*node) || IsIdentity(*node) || IsIdentityN(*node) ||
        IsConstant(*node) || SafeToConvertToNoOp(*node)) {
      nodes_to_simplify.PushBack(node);
    }
  }
  while (!nodes_to_simplify.Empty()) {
    NodeDef* node_to_simplify = nodes_to_simplify.PopBack();
    // Discard nodes that were marked for deletion already.
    while (nodes_to_delete.find(node_to_simplify->name()) !=
           nodes_to_delete.end()) {
      node_to_simplify = nodes_to_simplify.PopBack();
    }
    OptimizeNode(node_to_simplify, &nodes_to_simplify, &nodes_to_delete);
  }

  if (fetch_nodes_known_) {
    VLOG(1) << "Deleted " << nodes_to_delete.size() << " out of "
            << graph_view_->graph()->node_size() << " nodes.";
    graph_view_->DeleteNodes(nodes_to_delete);
  }
  return Status::OK();
}

Status DependencyOptimizer::TransitiveReduction() {
  // PRECONDITION: optimized_graph_ must be sorted topologically.
  GraphDef* graph = graph_view_->graph();
  auto node_to_idx = BuildNodeToIdx(*graph);
  const int num_nodes = graph->node_size();
  // Set up a compressed version of the graph to save a constant factor in the
  // expensive algorithm below. Also cache the set of control outputs and the
  // highest index of a target of any control output from each node.
  int num_controls = 0;
  std::vector<gtl::InlinedVector<int, 4>> inputs(num_nodes);
  std::vector<gtl::InlinedVector<std::pair<int, int>, 2>> control_outputs(
      num_nodes);
  for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const NodeDef& node = graph->node(node_idx);
    if (ModifiesFrameInfo(node) || !HasOpDef(node)) {
      // Ignore function nodes and nodes that modify frame info.
      continue;
    }
    for (int input_slot = 0; input_slot < node.input_size(); ++input_slot) {
      const string& input = node.input(input_slot);
      const NodeDef* input_node = graph_view_->GetNode(NodeName(input));
      if (ModifiesFrameInfo(*input_node) || IsMerge(*input_node)) {
        // Ignore edges from nodes that modify frame info and from Merge nodes,
        // because we cannot know which of it's input paths executes.
        continue;
      }
      const int input_node_idx = node_to_idx[input_node];
      inputs[node_idx].push_back(input_node_idx);
      if (IsControlInput(input)) {
        ++num_controls;
        control_outputs[input_node_idx].emplace_back(node_idx, input_slot);
      }
    }
  }

  // Run the longest path in DAG algorithm for each source node that has control
  // outputs. If, for any target node of a control output, there exists a path
  // of length > 1, we can drop that control dependency.
  int num_controls_removed = 0;
  std::vector<int> longest_distance(num_nodes);
  // Map from target_index -> set of (input_slot, source_index), representing
  // the control edges to remove. We sort them in reverse order by input slot,
  // such that when we swap them out so we don't clobber the
  // node(target).input() repeated field.
  typedef std::pair<int, int> InputSlotAndSource;
  std::unordered_map<
      int, std::set<InputSlotAndSource, std::greater<InputSlotAndSource>>>
      control_edges_to_remove;
  for (int source = 0; source < num_nodes; ++source) {
    int highest_control_target = -1;
    for (const auto& control_output : control_outputs[source]) {
      if (control_output.first > highest_control_target) {
        highest_control_target = control_output.first;
      }
    }
    if (highest_control_target <= source) {
      continue;
    }
    std::fill(longest_distance.begin() + source,
              longest_distance.begin() + highest_control_target + 1, 0);
    for (int target = source + 1; target <= highest_control_target; ++target) {
      for (int input : inputs[target]) {
        // If the input node is before source in the topo order, no path
        // source -> input -> target can exits and we can skip it.
        // Also only extend a path from the source itself or from nodes that
        // have a path from source, indicated by longest_distance[input] > 0.
        if (input == source ||
            (input > source && longest_distance[input] > 0)) {
          // If source -> input -> target is longer than the longest
          // path so far from source -> target, update the longest_distance.
          int candidate_longest_distance = longest_distance[input] + 1;
          if (candidate_longest_distance > longest_distance[target]) {
            longest_distance[target] = candidate_longest_distance;
          }
        }
      }
    }

    // If the longest path from source to target of a control dependency is
    // longer than 1, there exists an alternate path, and we can eliminate the
    // redundant direct control dependency.
    for (const auto& control_output : control_outputs[source]) {
      const int target = control_output.first;
      if (longest_distance[target] > 1) {
        const int input_slot = control_output.second;
        control_edges_to_remove[target].emplace(input_slot, source);
      }
    }
  }

  for (const auto& it : control_edges_to_remove) {
    const int target = it.first;
    const NodeDef& target_node = graph->node(target);
    const string target_node_name = target_node.name();
    for (const InputSlotAndSource& slot_and_source : it.second) {
      const int input_slot = slot_and_source.first;
      const TensorId tensor_id = ParseTensorName(target_node.input(input_slot));
      graph_view_->RemoveFanin(target_node_name, tensor_id);
      ++num_controls_removed;
    }
  }
  VLOG(1) << "Removed " << num_controls_removed << " out of " << num_controls
          << " control dependencies";
  return Status::OK();
}

// Suppose there are cross-device control inputs to node C from multiple nodes
// that are located on another device, e.g., we have control edges:
// A->C, B->C
// where A and B are on device X and C is on device Y.
// We can reduce cross-device communication by introducing an intermediate
// NoOp node C' on device X and rewriting the control edges to:
// A->C', B->C', C'->C
void DependencyOptimizer::GroupCrossDeviceControlEdges() {
  const int num_nodes = graph_view_->graph()->node_size();
  for (int i = 0; i < num_nodes; ++i) {
    NodeDef* node = graph_view_->graph()->mutable_node(i);
    if (node->device().empty()) continue;

    // Creates new noop nodes for devices on which multiple control inputs are
    // located.

    // Map keyed by device name to the newly introduced Noop node for that
    // device. A nullptr value means that we have only seen a single node on
    // that device.
    std::map<string, NodeDef*> noops;
    int num_noops = 0;
    auto controlling_fanins = graph_view_->GetFanin(
        MutableGraphView::InputPort(node, Graph::kControlSlot));
    for (const auto& controlling_fanin : controlling_fanins) {
      const NodeDef* fanin_node = controlling_fanin.node;
      if (!fanin_node->device().empty() &&
          fanin_node->device() != node->device()) {
        auto emplace_result = noops.emplace(fanin_node->device(), nullptr);
        if (!emplace_result.second && emplace_result.first->second == nullptr) {
          // This is the second cross-device control input from the same
          // device. Creates an intermediate noop node on that device.
          string group_name;
          NodeDef* noop;
          // Creates a fresh node name; there may be conflicting names from
          // a previous iteration of the optimizer.
          do {
            group_name = AddPrefixToNodeName(
                node->name(),
                strings::StrCat("GroupCrossDeviceControlEdges_", num_noops));
            noop = graph_view_->GetNode(group_name);
            ++num_noops;
          } while (noop != nullptr);
          NodeDef new_node;
          new_node.set_name(group_name);
          new_node.set_device(fanin_node->device());
          new_node.set_op("NoOp");
          emplace_result.first->second =
              graph_view_->AddNode(std::move(new_node));
        }
      }
    }

    // Reroute existing control edges to go via the newly introduced NoOp nodes.
    for (const auto& controlling_fanin : controlling_fanins) {
      auto it = noops.find(controlling_fanin.node->device());
      if (it != noops.end() && it->second != nullptr) {
        TensorId tensor_id(controlling_fanin.node->name(), Graph::kControlSlot);
        graph_view_->RemoveFanin(node->name(), tensor_id);
        graph_view_->AddFanin(it->second->name(), tensor_id);
      }
    }
    for (const auto& entry : noops) {
      if (entry.second) {
        graph_view_->AddFanin(node->name(),
                              {entry.second->name(), Graph::kControlSlot});
      }
    }
  }
}

Status DependencyOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  graph_view_.reset(new MutableGraphView(optimized_graph));

  const int num_iterations = 2;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    Status topo_sort_status;
    // Perform topological sort to prepare the graph for transitive reduction.
    topo_sort_status = TopologicalSort(optimized_graph);
    if (topo_sort_status.ok()) {
      // Remove redundant control dependencies.
      TF_RETURN_IF_ERROR(TransitiveReduction());
    } else {
      LOG(ERROR) << "Iteration = " << iteration
                 << ", topological sort failed with message: "
                 << topo_sort_status.error_message();
    }
    // Turn nodes with only control outputs into NoOps, prune NoOp and Identity
    // nodes.
    TF_RETURN_IF_ERROR(OptimizeDependencies());

    GroupCrossDeviceControlEdges();
  }

  return Status::OK();
}

void DependencyOptimizer::Feedback(Cluster* /*cluster*/,
                                   const GrapplerItem& /*item*/,
                                   const GraphDef& /*optimized_graph*/,
                                   double /*result*/) {
  // Nothing to do for DependencyOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
