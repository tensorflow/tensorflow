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

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

bool RemoveInput(NodeDef* node, const string& input, NodeMap* node_map) {
  bool removed_input = false;
  int pos = 0;
  while (pos < node->input_size()) {
    if (node->input(pos) == input) {
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
      node_map->RemoveOutput(NodeName(input), node->name());
      removed_input = true;
    } else {
      ++pos;
    }
  }
  return removed_input;
}

// Remove duplicate control inputs.
void PruneControlInputs(NodeDef* node) {
  std::unordered_set<string> inputs;
  int pos = 0;
  while (pos < node->input_size()) {
    const string& input = node->input(pos);
    if (!inputs.insert(NodeName(input)).second && IsControlInput(input)) {
      VLOG(1) << "**** Removing duplicate control input: " << input
              << " from node " << node->DebugString();
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    } else {
      ++pos;
    }
  }
}

}  // namespace

bool DependencyOptimizer::SafeToRemoveIdentity(const NodeDef& node) {
  if (!IsIdentity(node)) {
    return true;
  }
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (!fetch_nodes_known_) {
    // The output values of this node may be needed.
    return false;
  }
  const NodeDef* input = node_map_->GetNode(NodeName(node.input(0)));
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
    if (StringPiece(node.name()).starts_with(kConstantFoldingCtrl)) {
      // TODO(rmlarsen): Try to remove this artificial contraint.
      return false;
    }
    for (auto consumer : node_map_->GetOutputs(node.name())) {
      for (const string& consumer_input : consumer->input()) {
        if (consumer_input == AsControlDependency(node.name())) {
          return false;
        }
      }
    }
  }
  return true;
}

bool DependencyOptimizer::SafeToConvertToNoOp(const NodeDef& node) {
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (!fetch_nodes_known_ || NumNonControlOutputs(node, *node_map_) > 0) {
    // The output values of this node may be needed.
    return false;
  }
  if (IsMerge(node) || IsSwitch(node)) {
    return false;
  }
  if (ModifiesFrameInfo(node)) {
    return false;
  }
  if (!IsFreeOfSideEffect(node)) {
    return false;
  }
  if (node.op() == "ControlTrigger") {
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

  if (!SafeToRemoveIdentity(node)) {
    return false;
  }

  const std::unordered_set<string> do_not_rewrite_ops{
      "Assert",     "CheckNumerics",         "_Retval",
      "_Arg",       "_ParallelConcatUpdate", "_TPUExecute",
      "_TPUCompile"};
  return do_not_rewrite_ops.find(node.op()) == do_not_rewrite_ops.end();
}

void DependencyOptimizer::OptimizeNode(int node_idx,
                                       SetVector<int>* nodes_to_simplify,
                                       std::set<int>* nodes_to_delete) {
  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;
  NodeDef* node = optimized_graph_->mutable_node(node_idx);
  const bool is_noop = IsNoOp(*node);
  const bool is_identity = IsIdentity(*node);
  const string node_name = node->name();
  // Constant nodes with no input control dependency are always executed early,
  // so we can prune all their output control dependencies.
  if (IsConstant(*node) && node->input_size() == 0) {
    const std::set<NodeDef*> output_nodes = node_map_->GetOutputs(node_name);
    for (NodeDef* fanout : output_nodes) {
      bool optimize_fanout = false;
      bool data_connection = false;
      for (int i = fanout->input_size() - 1; i >= 0; --i) {
        int pos;
        string input_name = ParseNodeName(fanout->input(i), &pos);
        if (input_name == node_name) {
          if (pos < 0) {
            fanout->mutable_input()->SwapElements(i, fanout->input_size() - 1);
            fanout->mutable_input()->RemoveLast();
            optimize_fanout = true;
          } else {
            data_connection = true;
          }
        }
      }
      if (optimize_fanout) {
        nodes_to_simplify->PushBack(node_to_idx_[fanout]);
        if (!data_connection) {
          node_map_->RemoveOutput(node_name, fanout->name());
        }
      }
    }
    if (node_map_->GetOutputs(node_name).empty() && fetch_nodes_known_ &&
        nodes_to_preserve_.find(node_name) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node_to_idx_[node]);
    }
    return;
  }

  // Change ops that only have control dependencies as outputs to NoOps.
  if (!is_noop && SafeToConvertToNoOp(*node)) {
    VLOG(1) << "***** Replacing  " << node_name << " (" << node->op()
            << ") with NoOp.";
    // The outputs of this node are not consumed. Replace its inputs with
    // control dependencies and replace the op itself with the NoOp op.
    std::unordered_set<string> ctrl_inputs;
    int pos = 0;
    while (pos < node->input_size()) {
      const string old_input = node->input(pos);
      if (IsControlInput(old_input)) {
        if (!ctrl_inputs.insert(old_input).second) {
          // We found a duplicate control input. Remove it.
          node->mutable_input()->SwapElements(pos, node->input_size() - 1);
          node->mutable_input()->RemoveLast();
        } else {
          ++pos;
        }
        continue;
      }
      const string ctrl_input = ConstantFolding::AddControlDependency(
          old_input, optimized_graph_, node_map_.get());
      if (ctrl_inputs.insert(ctrl_input).second) {
        node->set_input(pos, ctrl_input);
        node_map_->UpdateInput(node_name, old_input, ctrl_input);
        const NodeDef* old_input_node = node_map_->GetNode(old_input);
        nodes_to_simplify->PushBack(node_to_idx_[old_input_node]);
      }
      ++pos;
    }
    node->set_op("NoOp");
    node->clear_attr();
    nodes_to_simplify->PushBack(node_to_idx_[node]);
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

  if (is_noop || (is_identity && is_aggressive)) {
    const auto& output_node_set = node_map_->GetOutputs(node_name);
    const std::vector<NodeDef*> output_nodes(output_node_set.begin(),
                                             output_node_set.end());
    const int num_outputs = output_nodes.size();
    const int num_inputs = node->input_size();

    if (num_inputs * num_outputs > num_inputs + num_outputs) {
      return;
    }
    std::vector<NodeDef*> input_nodes;
    for (int i = 0; i < num_inputs; ++i) {
      NodeDef* input_node = node_map_->GetNode(node->input(i));
      CHECK_NE(input_node, nullptr);
      input_nodes.push_back(input_node);
    }

    // Make sure that we don't increase the number of edges that cross
    // device boundaries.
    if ((num_inputs == 1 && num_outputs > 1 &&
         input_nodes[0]->device() != node->device()) ||
        (num_inputs > 1 && num_outputs == 1 &&
         output_nodes[0]->device() != node->device())) {
      return;
    }
    if (num_inputs == 2 && num_outputs == 2) {
      const string& noop_dev = node->device();
      const string& in0_dev = input_nodes[0]->device();
      const string& in1_dev = input_nodes[1]->device();
      const string& out0_dev = output_nodes[0]->device();
      const string& out1_dev = output_nodes[1]->device();
      const int num_cross_before = static_cast<int>(in0_dev != noop_dev) +
                                   static_cast<int>(in1_dev != noop_dev) +
                                   static_cast<int>(out0_dev != noop_dev) +
                                   static_cast<int>(out1_dev != noop_dev);
      const int num_cross_after = static_cast<int>(in0_dev != out0_dev) +
                                  static_cast<int>(in0_dev != out1_dev) +
                                  static_cast<int>(in1_dev != out0_dev) +
                                  static_cast<int>(in1_dev != out1_dev);
      if (num_cross_after > num_cross_before) {
        return;
      }
      // To avoid potentially removing Identity nodes following _Recv nodes,
      // we require that no device crossings occur in that case.
      // TODO(rmlarsen): See if we can relax this condition.
      if (is_identity && (num_cross_after > 0 || num_cross_before > 0)) {
        return;
      }
    }
    if (is_identity && !SafeToRemoveIdentity(*node)) {
      return;
    }

    VLOG(1) << "***** Rerouting input around\n" << node->DebugString();
    // Now remove the node and re-wire its inputs to its outputs.
    for (auto consumer : output_nodes) {
      bool updated_consumer = false;
      VLOG(1) << "consumer before:\n" << consumer->DebugString();
      for (int i = 0; i < num_inputs; ++i) {
        const NodeDef* input = input_nodes[i];
        // Forward dependency from input to consumer if it doesn't already
        // depend on it.
        if (is_identity && i == 0) {
          // Replace regular input from Identity node.
          bool found_input = false;
          string new_input;
          const string& input_to_forward = node->input(0);
          CHECK(!IsControlInput(input_to_forward));
          for (int j = 0; j < consumer->input_size(); ++j) {
            const string& old_input = consumer->input(j);
            if (old_input == node_name) {
              new_input = input_to_forward;
              node_map_->UpdateInput(consumer->name(), old_input, new_input);
              consumer->set_input(j, new_input);
              found_input = true;
            } else if (old_input == AsControlDependency(NodeName(node_name))) {
              new_input = AsControlDependency(NodeName(input_to_forward));
              node_map_->UpdateInput(consumer->name(), old_input, new_input);
              consumer->set_input(j, new_input);
              found_input = true;
            }
          }
          CHECK(found_input);
          updated_consumer = true;
        } else {
          // Forward dependency from input to consumer if it doesn't already
          // depend on it.
          if (node_map_->GetOutputs(input->name()).count(consumer) == 0) {
            consumer->add_input(AsControlDependency(input->name()));
            node_map_->AddOutput(input->name(), consumer->name());
            nodes_to_simplify->PushBack(node_to_idx_[input]);
            updated_consumer = true;
          }
        }
      }
      // Remove dependency on node from consumer.
      updated_consumer |= RemoveInput(consumer, AsControlDependency(node_name),
                                      node_map_.get());
      if (updated_consumer) {
        nodes_to_simplify->PushBack(node_to_idx_[consumer]);
      }
      VLOG(1) << "consumer after:\n" << consumer->DebugString();
    }
    node_map_->RemoveOutputs(node_name);
    if (fetch_nodes_known_ &&
        nodes_to_preserve_.find(node_name) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node_idx);

      // Disconnect the node from its inputs to enable further optimizations.
      node_map_->RemoveInputs(node_name);
      node->clear_input();
    }
  }
}

void DependencyOptimizer::CleanControlInputs() {
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    PruneControlInputs(optimized_graph_->mutable_node(i));
  }
}

void DependencyOptimizer::DeleteNodes(const std::set<int>& nodes_to_delete) {
  int last = optimized_graph_->node_size() - 1;
  for (auto it = nodes_to_delete.rbegin(); it != nodes_to_delete.rend(); ++it) {
    const int index = *it;
    optimized_graph_->mutable_node()->SwapElements(index, last);
    last--;
  }
  optimized_graph_->mutable_node()->DeleteSubrange(last + 1,
                                                   nodes_to_delete.size());
  // Rebuild the NodeMap which was invalidated by the node swapping above.
  node_map_.reset(new NodeMap(optimized_graph_));
  BuildNodeToIdx();
}

Status DependencyOptimizer::OptimizeDependencies() {
  SetVector<int> nodes_to_simplify;
  std::set<int> nodes_to_delete;
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    const NodeDef& node = optimized_graph_->node(i);
    if (IsNoOp(node) || IsIdentity(node) || IsConstant(node) ||
        SafeToConvertToNoOp(node)) {
      nodes_to_simplify.PushBack(i);
    }
  }
  while (!nodes_to_simplify.Empty()) {
    int node_to_simplify = nodes_to_simplify.PopBack();
    // Discard nodes that were marked for deletion already.
    while (nodes_to_delete.find(node_to_simplify) != nodes_to_delete.end()) {
      node_to_simplify = nodes_to_simplify.PopBack();
    }
    OptimizeNode(node_to_simplify, &nodes_to_simplify, &nodes_to_delete);
  }

  if (fetch_nodes_known_) {
    VLOG(1) << "Deleted " << nodes_to_delete.size() << " out of "
            << optimized_graph_->node_size() << " nodes.";
    DeleteNodes(nodes_to_delete);
  }
  return Status::OK();
}

Status DependencyOptimizer::TransitiveReduction() {
  // PRECONDITION: optimized_graph_ must be sorted topologically.
  const int num_nodes = optimized_graph_->node_size();
  // Set up a compressed version of the graph to save a constant factor in the
  // expensive algorithm below. Also cache the set of control outputs and the
  // highest index of a target of any control output from each node.
  int num_controls = 0;
  std::vector<gtl::InlinedVector<int, 4>> inputs(num_nodes);
  std::vector<gtl::InlinedVector<std::pair<int, int>, 2>> control_outputs(
      num_nodes);
  for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const NodeDef& node = optimized_graph_->node(node_idx);
    if (ModifiesFrameInfo(node) || !HasOpDef(node)) {
      // Ignore function nodes and nodes that modify frame info.
      continue;
    }
    for (int input_slot = 0; input_slot < node.input_size(); ++input_slot) {
      const string& input = node.input(input_slot);
      const NodeDef* input_node = node_map_->GetNode(input);
      if (ModifiesFrameInfo(*input_node) || IsMerge(*input_node)) {
        // Ignore edges from nodes that modify frame info and from Merge nodes,
        // because we cannot know which of it's input paths executes.
        continue;
      }
      const int input_node_idx = node_to_idx_[input_node];
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
        //        VLOG(1) << "Removing edge from:\n"
        //                << optimized_graph_->node(source).DebugString() <<
        //                "\n\nto:\n\n"
        //                << optimized_graph_->node(target).DebugString();
      }
    }
  }

  for (const auto& it : control_edges_to_remove) {
    const int target = it.first;
    NodeDef* target_node = optimized_graph_->mutable_node(target);
    for (const InputSlotAndSource& slot_and_source : it.second) {
      const int input_slot = slot_and_source.first;
      const int source = slot_and_source.second;
      const NodeDef& source_node = optimized_graph_->node(source);
      CHECK_LT(input_slot, target_node->input_size());
      target_node->mutable_input()->SwapElements(input_slot,
                                                 target_node->input_size() - 1);
      node_map_->RemoveOutput(source_node.name(), target_node->name());
      target_node->mutable_input()->RemoveLast();
      ++num_controls_removed;
    }
  }
  VLOG(1) << "Removed " << num_controls_removed << " out of " << num_controls
          << " control dependencies";
  return Status::OK();
}

void DependencyOptimizer::BuildNodeToIdx() {
  // Set up &node -> index map.
  node_to_idx_.clear();
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    const NodeDef& node = optimized_graph_->node(i);
    node_to_idx_[&node] = i;
  }
}

Status DependencyOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  optimized_graph_ = optimized_graph;
  *optimized_graph_ = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  CleanControlInputs();

  const int num_iterations = 2;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    Status topo_sort_status;
    // Perform topological sort to prepare the graph for transitive reduction.
    topo_sort_status = TopologicalSort(optimized_graph_);

    // Set up index-based graph datastructures to speed up analysis steps below.
    node_map_.reset(new NodeMap(optimized_graph_));
    BuildNodeToIdx();

    if (topo_sort_status.ok()) {
      // Remove redundant control dependencies.
      TF_RETURN_IF_ERROR(TransitiveReduction());
    } else {
      LOG(ERROR) << topo_sort_status.error_message();
    }
    // Turn nodes with only control outputs into NoOps, prune NoOp and Identity
    // nodes.
    TF_RETURN_IF_ERROR(OptimizeDependencies());

    // Dedup control inputs.
    CleanControlInputs();
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
