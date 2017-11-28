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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
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
      node_map->RemoveOutput(node->name(), NodeName(input));
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

string DependencyOptimizer::TryOptimizeDependencies(
    NodeDef* node, SetVector<NodeDef*>* nodes_to_simplify) {
  // Change ops that only have control dependencies as outputs to NoOps.
  if (node->op() != "NoOp" && SafeToConvertToNoOp(*node)) {
    VLOG(1) << "***** Replacing  " << node->name() << " (" << node->op()
            << ") with NoOp.";
    // The outputs of this node are not consumed. Replace its inputs with
    // control dependencies and replace the op itself with the NoOp op.
    std::unordered_set<string> ctrl_inputs;
    int pos = 0;
    while (pos < node->input_size()) {
      const string& old_input = node->input(pos);
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
        auto old_input_node = node_map_->GetNode(old_input);
        nodes_to_simplify->PushBack(old_input_node);
      }
      ++pos;
    }
    node->set_op("NoOp");
    node->clear_attr();
    nodes_to_simplify->PushBack(node);
    return "";
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
  if (node->op() == "NoOp" &&
      nodes_to_preserve_.find(node->name()) == nodes_to_preserve_.end()) {
    const auto output_nodes = node_map_->GetOutputs(node->name());
    const int num_outputs = output_nodes.size();
    const int num_inputs = node->input_size();
    if (num_inputs > 1 && num_outputs > 1) {
      return "";
    }
    VLOG(1) << "***** Rerouting input around  " << node->name();
    std::vector<NodeDef*> input_nodes;
    for (int i = 0; i < num_inputs; ++i) {
      NodeDef* tmp = node_map_->GetNode(node->input(i));
      if (tmp != nullptr) {
        input_nodes.push_back(tmp);
      }
    }
    for (auto consumer : output_nodes) {
      bool updated_consumer = false;
      VLOG(1) << "***** Considering consumer  " << consumer->name() << "\n"
              << consumer->DebugString();
      for (int i = 0; i < num_inputs; ++i) {
        const string& input = node->input(i);
        // Forward dependency from input to consumer if it doesn't already
        // depend on it.
        if (node_map_->GetOutputs(NodeName(input)).count(consumer) == 0) {
          consumer->add_input(input);
          updated_consumer = true;
          node_map_->AddOutput(NodeName(input), consumer->name());
          nodes_to_simplify->PushBack(input_nodes[i]);
        }
      }
      // Remove dependency on node from consumer.
      updated_consumer |= RemoveInput(
          consumer, AsControlDependency(node->name()), node_map_.get());
      if (updated_consumer) {
        VLOG(1) << "***** Updated consumer  " << consumer->name() << " ("
                << consumer->op() << ")";
        nodes_to_simplify->PushBack(consumer);
      }
    }

    // Clear all (control) inputs to this NoOp node.
    if (fetch_nodes_known_) {
      node_map_->RemoveInputs(node->name());
      node->clear_input();
    }
  }

  return "";
}

Status DependencyOptimizer::OptimizeDependencies() {
  // TODO(rmlarsen,bsteiner): The following code is similar to the control loop
  // in the ArithmeticOptimizer. Dedup this.
  SetVector<NodeDef*> nodes_to_simplify;
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    NodeDef* node = optimized_graph_->mutable_node(i);
    if (node->op() == "NoOp" || SafeToConvertToNoOp(*node)) {
      PruneControlInputs(node);
      nodes_to_simplify.PushBack(node);
    }
  }
  while (!nodes_to_simplify.Empty()) {
    NodeDef* node = nodes_to_simplify.PopBack();
    const string simplified_tensor =
        TryOptimizeDependencies(node, &nodes_to_simplify);
    if (!simplified_tensor.empty() &&
        NodeName(simplified_tensor) != node->name()) {
      // Always consider simplified_tensor for further optimizations.
      NodeDef* simplified_node = node_map_->GetNode(simplified_tensor);
      if (simplified_node != nullptr) {
        nodes_to_simplify.PushBack(simplified_node);
      }
      // When `node` is simplifed to another node rather than in-place, the
      // consumers of `node` are already redirected to `simplified_tensor`.
      // Re-push the consumers into `nodes_to_simplify` for further
      // optimizations.
      std::set<NodeDef*> consumers = node_map_->GetOutputs(node->name());
      for (NodeDef* consumer : consumers) {
        // Update `consumer`'s use of `node` to `input`'s operand.
        for (int i = 0; i < consumer->input_size(); ++i) {
          int operand_pos;
          string operand_node_name =
              ParseNodeName(consumer->input(i), &operand_pos);
          if (operand_node_name == node->name()) {
            *consumer->mutable_input(i) =
                (operand_pos < 0
                     ? AsControlDependency(NodeName(simplified_tensor))
                     : simplified_tensor);
          }
        }
        node_map_->UpdateInput(consumer->name(), node->name(),
                               simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
  }
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    NodeDef* node = optimized_graph_->mutable_node(i);
    PruneControlInputs(node);
  }
  return Status::OK();
}

Status DependencyOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  optimized_graph_ = optimized_graph;
  *optimized_graph_ = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  node_map_.reset(new NodeMap(optimized_graph));
  fetch_nodes_known_ = !item.fetch.empty();
  VLOG(1) << "Graph before optimization:\n" << optimized_graph_->DebugString();
  TF_RETURN_IF_ERROR(OptimizeDependencies());
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
