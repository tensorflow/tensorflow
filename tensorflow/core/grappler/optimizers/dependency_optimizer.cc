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

int RemoveInput(NodeDef* node, const string& input, NodeMap* node_map) {
  int num_removed = 0;
  int pos = 0;
  while (pos < node->input_size()) {
    if (node->input(pos) == input) {
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
      node_map->RemoveOutput(NodeName(input), node->name());
    } else {
      ++pos;
    }
    ++num_removed;
  }
  return num_removed;
}

// Remove dulicate control inputs.
void PruneControlInputs(NodeDef* node) {
  std::unordered_set<string> inputs;
  int pos = 0;
  while (pos < node->input_size()) {
    const string& input = node->input(pos);
    // TODO(rmlarsen): Remove control inputs that also appears as a regular
    // inputs. Currently, doing so breaks testControlFlowStrictness in
    // python/framework/function_test.
    //    if (!inputs.insert(NodeName(input)).second && IsControlInput(input)) {
    if (IsControlInput(input) && !inputs.insert(input).second) {
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

  // TODO(rmlarsen): We have to skip Identity nodes to make an obsolete test in
  // python/training/session_manager_test.py pass. See if we can fix or get rid
  // of that test.
  const std::unordered_set<string> do_not_rewrite_ops{
      "Assert", "CheckNumerics",         "Identity",    "_Retval",
      "_Arg",   "_ParallelConcatUpdate", "_TPUExecute", "_TPUCompile"};
  return do_not_rewrite_ops.find(node.op()) == do_not_rewrite_ops.end();
}

void DependencyOptimizer::OptimizeNode(int node_idx,
                                       SetVector<int>* nodes_to_simplify,
                                       std::set<int>* nodes_to_delete) {
  NodeDef* node = optimized_graph_->mutable_node(node_idx);

  // Constant nodes with no input control dependency are always executed early,
  // so we can prune all their output control dependencies.
  if (IsConstant(*node) && node->input_size() == 0) {
    const std::set<NodeDef*> output_nodes = node_map_->GetOutputs(node->name());
    for (NodeDef* fanout : output_nodes) {
      bool optimize_fanout = false;
      bool data_connection = false;
      for (int i = fanout->input_size() - 1; i >= 0; --i) {
        int pos;
        string input_name = ParseNodeName(fanout->input(i), &pos);
        if (input_name == node->name()) {
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
          node_map_->RemoveOutput(node->name(), fanout->name());
        }
      }
    }
    if (node_map_->GetOutputs(node->name()).empty() && fetch_nodes_known_ &&
        nodes_to_preserve_.find(node->name()) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node_to_idx_[node]);
    }

    return;
  }

  // Change ops that only have control dependencies as outputs to NoOps.
  if (node->op() != "NoOp" && SafeToConvertToNoOp(*node)) {
    VLOG(1) << "***** Replacing  " << node->name() << " (" << node->op()
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
        node_map_->UpdateInput(node->name(), old_input, ctrl_input);
        const NodeDef* old_input_node = node_map_->GetNode(old_input);
        nodes_to_simplify->PushBack(node_to_idx_[old_input_node]);
      }
      ++pos;
    }
    node->set_op("NoOp");
    node->clear_attr();
  }

  // Remove NoOp nodes if their fan-in or fan-out is less than 2.
  // The non-trivial rewrites take the following form:
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
  if (node->op() == "NoOp") {
    const auto output_nodes = node_map_->GetOutputs(node->name());
    const int num_outputs = output_nodes.size();
    const int num_inputs = node->input_size();

    if (num_inputs * num_outputs > num_inputs + num_outputs) {
      return;
    }
    VLOG(1) << "***** Rerouting input around " << node->name();
    std::vector<NodeDef*> input_nodes;
    for (int i = 0; i < num_inputs; ++i) {
      NodeDef* tmp = node_map_->GetNode(node->input(i));
      CHECK_NE(tmp, nullptr);
      input_nodes.push_back(tmp);
    }

    for (auto consumer : output_nodes) {
      bool updated_consumer = false;
      VLOG(1) << "***** Considering consumer  " << consumer->name() << "\n"
              << consumer->DebugString();
      for (int i = 0; i < num_inputs; ++i) {
        const NodeDef* input = input_nodes[i];
        // Forward dependency from input to consumer if it doesn't already
        // depend on it.
        if (node_map_->GetOutputs(input->name()).count(consumer) == 0) {
          consumer->add_input(AsControlDependency(input->name()));
          updated_consumer = true;
          node_map_->AddOutput(input->name(), consumer->name());
          nodes_to_simplify->PushBack(node_to_idx_[input]);
        }
      }
      // Remove dependency on node from consumer.
      updated_consumer |= RemoveInput(
          consumer, AsControlDependency(node->name()), node_map_.get());
      if (updated_consumer) {
        VLOG(1) << "***** Updated consumer  " << consumer->name() << " ("
                << consumer->op() << ")";
        nodes_to_simplify->PushBack(node_to_idx_[consumer]);
      }
    }

    node_map_->RemoveOutputs(node->name());
    if (fetch_nodes_known_ &&
        nodes_to_preserve_.find(node->name()) == nodes_to_preserve_.end()) {
      // Mark the node for deletion.
      nodes_to_delete->insert(node_idx);

      // Unconnect the node from its inputs to enable further optimizations.
      node_map_->RemoveInputs(node->name());
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
    if (node.op() == "NoOp" || IsConstant(node) || SafeToConvertToNoOp(node)) {
      nodes_to_simplify.PushBack(i);
    }
  }
  while (!nodes_to_simplify.Empty()) {
    OptimizeNode(nodes_to_simplify.PopBack(), &nodes_to_simplify,
                 &nodes_to_delete);
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
    if (ModifiesFrameInfo(node)) {
      // Ignore nodes that modify frame info.
      continue;
    }
    for (int input_slot = 0; input_slot < node.input_size(); ++input_slot) {
      const string& input = node.input(input_slot);
      const NodeDef* input_node = node_map_->GetNode(input);
      if (ModifiesFrameInfo(*input_node)) {
        // Ignore edges from nodes that modify frame info.
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
  for (int source = 0; source < num_nodes; ++source) {
    int highest_control_target = -1;
    for (const auto& control_output : control_outputs[source]) {
      if (control_output.first > highest_control_target) {
        highest_control_target = control_output.first;
      }
    }
    if (highest_control_target < source) {
      continue;
    }
    std::fill(longest_distance.begin() + source,
              longest_distance.begin() + highest_control_target + 1, 0);
    for (int target = source + 1; target <= highest_control_target; ++target) {
      for (int input : inputs[target]) {
        // If the input node is before source in the topo order, no path
        // source -> input -> target can exits and we can skip it.
        if (input >= source) {
          // If source -> input -> target is longer than the longest
          // path so far from source -> target, update the longest_distance.
          int candidate_longest_distance = longest_distance[input] + 1;
          if (candidate_longest_distance > longest_distance[target]) {
            longest_distance[target] = candidate_longest_distance;
          }
        }
      }
    }

    // If the longest path from the source to the target of a control dependency
    // is longer than 1, there exists an alternate path, and we can eliminate
    // the control dependency since it is redundant.
    for (const auto& control_output : control_outputs[source]) {
      const int target = control_output.first;
      if (longest_distance[target] > 1) {
        const int input_slot = control_output.second;
        // We modify the node inplace here. This is safe because there can
        // only be one control edge from a given source to a given target.
        const NodeDef& source_node = optimized_graph_->node(source);
        NodeDef* target_node = optimized_graph_->mutable_node(target);
        target_node->mutable_input()->SwapElements(
            input_slot, target_node->input_size() - 1);
        node_map_->RemoveOutput(source_node.name(), target_node->name());
        target_node->mutable_input()->RemoveLast();
        ++num_controls_removed;
      }
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

  VLOG(1) << "Graph before optimization:\n" << optimized_graph_->DebugString();
  CleanControlInputs();
  const int num_iterations = opt_level_ == RewriterConfig::AGGRESSIVE ? 2 : 1;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    Status topo_sort_status;
    if (opt_level_ == RewriterConfig::AGGRESSIVE) {
      // Prepare the graph for transitive reduction if enabled.
      topo_sort_status = TopologicalSort(optimized_graph_);
    }

    node_map_.reset(new NodeMap(optimized_graph_));
    BuildNodeToIdx();

    // Remove redundant control dependencies, iteration 1.
    if (opt_level_ == RewriterConfig::AGGRESSIVE) {
      if (topo_sort_status.ok()) {
        TF_RETURN_IF_ERROR(TransitiveReduction());
      } else {
        LOG(ERROR) << topo_sort_status.error_message();
      }
      VLOG(1) << "Graph after transitive reduction:\n"
              << optimized_graph_->DebugString();
    }

    // Turn nodes without non-control outputs into NoOps, prune NoOps.
    TF_RETURN_IF_ERROR(OptimizeDependencies());
    VLOG(1) << "Graph after NoOp conversion & pruning:\n"
            << optimized_graph_->DebugString();
  }
  VLOG(1) << "Graph after optimization:\n" << optimized_graph_->DebugString();

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
