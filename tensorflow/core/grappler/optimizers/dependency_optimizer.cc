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
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {
// A vector with a set. The set stores the same elements as the vector, and
// quickly answers whether a value is in the vector. Duplicated elements are not
// allowed for now.
template <class T>
class SetVector {
 public:
  // Returns false if value already existed in the set, true otherwise.
  bool PushBack(const T& value) {
    if (!set_.insert(value).second) {
      return false;
    }
    vector_.push_back(value);
    return true;
  }

  T PopBack() {
    T back = vector_.back();
    set_.erase(back);
    vector_.pop_back();
    return back;
  }

  bool Exists(const T& value) const { return set_.count(value); }

  bool Empty() const { return vector_.empty(); }

  void Reserve(int64 size) { vector_.reserve(size); }

 private:
  std::unordered_set<T> set_;
  std::vector<T> vector_;
};

bool HasRegularOutputs(const NodeDef& node, const NodeMap& node_map) {
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& input : output->input()) {
      if (input == node.name()) {
        return true;
      }
    }
  }
  return false;
}

int FindInputSlot(const NodeDef& node, const string& input) {
  for (int i = 0; i < node.input_size(); ++i) {
    if (node.input(i) == input) {
      return i;
    }
  }
  return -1;
}

}  // namespace

bool DependencyOptimizer::SafeToConvertToNoOp(const NodeDef& node) {
  if (!has_fetch_ || HasRegularOutputs(node, *node_map_)) {
    return false;
  }

  if (IsMerge(node)) {
    return false;
  }
  if (!ArithmeticOptimizer::CanDedup(node, nodes_to_preserve_)) {
    return false;
  }

  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok() || op_def->output_arg_size() == 0) {
    return false;
  }

  // TODO(rmlarsen): We have to skip Const nodes to make
  // core/debug/debug_gateway_test pass. See if we can fix that test.
  // TODO(rmlarsen): We have to skip Identity nodes to make an obsolete test in
  // python/training/session_manager_test.py pass. See if we can fix or get rid
  // of that test.
  const std::unordered_set<string> do_not_rewrite_ops = {
      "Assert", "CheckNumerics",         "Const",      "Identity", "_Retval",
      "_Arg",   "_ParallelConcatUpdate", "_TPUExecute"};
  return do_not_rewrite_ops.find(node.op()) == do_not_rewrite_ops.end();
}

string DependencyOptimizer::TryOptimizeDependencies(
    NodeDef* node, GraphDef* graph, std::vector<NodeDef*>* new_nodes) {
  // Change ops that only have control dependencies as outputs to NoOps.
  if (node->op() != "NoOp" && SafeToConvertToNoOp(*node)) {
    VLOG(2) << "***** Replacing  " << node->name() << " (" << node->op()
            << ") with NoOp.";
    // The outputs of this node are not consumed. Replace its inputs with
    // control dependencies and replace the op itself with the NoOp op.
    for (int i = 0; i < node->input_size(); ++i) {
      const string& old_input = node->input(i);
      if (IsControlInput(old_input)) {
        continue;
      }
      const string ctrl_input = ConstantFolding::AddControlDependency(
          old_input, graph, node_map_.get());
      node->set_input(i, ctrl_input);
      node_map_->UpdateInput(node->name(), old_input, ctrl_input);
      new_nodes->push_back(node_map_->GetNode(old_input));
    }
    node->set_op("NoOp");
    node->clear_attr();
    new_nodes->push_back(node);
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
    auto outputs = node_map_->GetOutputs(node->name());
    const int num_outputs = outputs.size();
    const int num_inputs = node->input_size();
    if (num_inputs > 1 && num_outputs > 1) {
      return "";
    }

    for (auto consumer : outputs) {
      for (int i = 0; i < num_inputs; ++i) {
        const string& input = node->input(i);
        // Forward dependencies from inputs to consumer if it doesn't already
        // depend on it.
        if (node_map_->GetOutputs(input).count(consumer) == 0) {
          consumer->add_input(ConstantFolding::AddControlDependency(
              input, graph, node_map_.get()));
          node_map_->AddOutput(NodeName(input), consumer->name());
        }
        new_nodes->push_back(node_map_->GetNode(input));
      }
      // Remove dependency on node from consumer.
      int pos = FindInputSlot(*consumer, AsControlDependency(node->name()));
      if (pos >= 0) {
        consumer->mutable_input()->SwapElements(pos,
                                                consumer->input_size() - 1);
        consumer->mutable_input()->RemoveLast();
        node_map_->RemoveOutput(node->name(), consumer->name());
        new_nodes->push_back(consumer);
      }
    }

    // Clear all control inputs to node.
    node_map_->RemoveInputs(node->name());
    node->clear_input();
    return "";
  }

  return "";
}

Status DependencyOptimizer::OptimizeDependencies(GraphDef* optimized_graph) {
  // TODO(rmlarsen,bsteiner): The folloing code is similar to the control loop
  // in the ArithmeticOptimizer. Dedup this.
  SetVector<NodeDef*> nodes_to_simplify;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    const NodeDef& node = optimized_graph->node(i);
    if (node.op() == "NoOp" || SafeToConvertToNoOp(node)) {
      nodes_to_simplify.PushBack(optimized_graph->mutable_node()->Mutable(i));
    }
  }
  while (!nodes_to_simplify.Empty()) {
    NodeDef* node = nodes_to_simplify.PopBack();
    std::vector<NodeDef*> new_nodes;
    const string simplified_tensor =
        TryOptimizeDependencies(node, optimized_graph, &new_nodes);
    if (simplified_tensor.empty()) {
      continue;
    }
    if (NodeName(simplified_tensor) != node->name()) {
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
          VLOG(2) << "Update input " << consumer->input(i) << " of "
                  << consumer->name() << " to " << simplified_tensor;
        }
        node_map_->UpdateInput(consumer->name(), node->name(),
                               simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
    for (auto new_node : new_nodes) {
      nodes_to_simplify.PushBack(new_node);
    }
  }
  return Status::OK();
}

Status DependencyOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  node_map_.reset(new NodeMap(optimized_graph));
  has_fetch_ = !item.fetch.empty();
  VLOG(2) << "Graph before optimization:\n" << optimized_graph->DebugString();
  TF_RETURN_IF_ERROR(OptimizeDependencies(optimized_graph));
  VLOG(2) << "Graph after optimization:\n" << optimized_graph->DebugString();

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
