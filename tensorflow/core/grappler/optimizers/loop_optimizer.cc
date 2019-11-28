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

#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

using TensorVector = gtl::InlinedVector<TensorValue, 4>;

class LoopInvariantNodeMotionOptimizer {
 public:
  explicit LoopInvariantNodeMotionOptimizer(GraphDef* optimized_graph)
      : optimized_graph_(optimized_graph) {}
  virtual ~LoopInvariantNodeMotionOptimizer() = default;
  Status Optimize();

 private:
  Status FindInvariantNodes(NodeDef* node);
  Status RevertInvariantNodes();
  Status MoveInvariantNodes(const int frame_id);
  Status HandleInvariantNode(NodeDef* node, const int num_outputs,
                             const int frame_id);
  Status HandleConst(NodeDef* node, const int num_outputs, const int frame_id);
  Status HandleInvariantEnter(NodeDef* node, const int num_outputs);

  GraphDef* optimized_graph_;  // Not owned.
  std::unique_ptr<NodeMap> node_map_;
  std::map<NodeDef*, int> invariant_nodes_;
  std::set<int> empty_set_;
  std::vector<std::set<int>> frame_children_;
  std::vector<int> frame_parent_;
  std::map<int, const NodeDef*> loop_cond_;
  std::map<int, std::vector<NodeDef*>> invariant_enters_;
  int new_enter_id_;
};

Status LoopInvariantNodeMotionOptimizer::HandleInvariantEnter(
    NodeDef* node, const int num_outputs) {
  auto consumers = node_map_->GetOutputs(node->name());
  std::vector<string> enter_control_inputs;
  string enter_input;
  for (auto& input : node->input()) {
    if (IsControlInput(input)) {
      enter_control_inputs.push_back(input);
    } else {
      enter_input = input;
    }
  }
  for (auto* consumer : consumers) {
    if (invariant_nodes_.count(consumer)) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        if (NodeName(consumer->input(i)) == node->name()) {
          consumer->set_input(i, enter_input);
          node_map_->AddOutput(NodeName(enter_input), consumer->name());
          node_map_->RemoveOutput(node->name(), consumer->name());
        }
      }
      for (auto& control_input : enter_control_inputs) {
        consumer->add_input(control_input);
        node_map_->AddOutput(NodeName(control_input), consumer->name());
      }
    }
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::HandleConst(NodeDef* node,
                                                     const int num_outputs,
                                                     const int frame_id) {
  NodeDef* const_node = nullptr;
  if (num_outputs == 0) {
    // all successor nodes are invariant
    // Remove the control inputs from this frame to the const node,
    // when moving it out of the frame (in parent frame)
    const_node = node;
    node_map_->RemoveInputs(node->name());
    node->clear_input();
  } else {
    // some successor nodes are variant
    // Have to keep the const node in the frame,
    // so create a new one outside the frame (in parent frame)
    const string const_node_name =
        AddPrefixToNodeName(node->name(), kLoopOptimizer);
    const_node = node_map_->GetNode(const_node_name);
    if (const_node == nullptr) {
      const_node = optimized_graph_->add_node();
      const_node->set_name(const_node_name);
      const_node->set_op("Const");
      const_node->set_device(node->device());
      *const_node->mutable_attr() = node->attr();
      node_map_->AddNode(const_node->name(), const_node);
    }
    auto consumers = node_map_->GetOutputs(node->name());
    for (auto* consumer : consumers) {
      if (invariant_nodes_.count(consumer)) {
        for (int i = 0; i < consumer->input_size(); ++i) {
          if (NodeName(consumer->input(i)) == node->name()) {
            if (IsControlInput(consumer->input(i))) {
              *consumer->mutable_input(i) = AsControlDependency(*const_node);
            } else {
              *consumer->mutable_input(i) = const_node->name();
            }
            node_map_->AddOutput(const_node->name(), consumer->name());
            node_map_->RemoveOutput(node->name(), consumer->name());
          }
        }
      }
    }
  }
  // add a control input from the parent frame
  if (frame_parent_[frame_id] != -1) {
    int parent_id = frame_parent_[frame_id];
    auto loop_cond_it = loop_cond_.find(parent_id);
    if (loop_cond_it == loop_cond_.end()) {
      return errors::InvalidArgument("Frame ", frame_id,
                                     " doesn't have a LoopCond node");
    }
    auto& loop_cond_name = loop_cond_it->second->name();
    NodeDef* switch_node = nullptr;
    for (auto* node : node_map_->GetOutputs(loop_cond_name)) {
      if (node->op() == "Switch") {
        switch_node = node;
        break;
      }
    }
    if (!switch_node) {
      return errors::InvalidArgument("LoopCond node of Frame ", frame_id,
                                     " doesn't connect to any Switch node");
    }
    string switch_output = StrCat(switch_node->name(), ":1");
    const string ctrl_dep = ConstantFolding::AddControlDependency(
        switch_output, optimized_graph_, node_map_.get());
    const_node->add_input(ctrl_dep);
    node_map_->AddOutput(NodeName(ctrl_dep), const_node->name());
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::HandleInvariantNode(
    NodeDef* node, const int num_outputs, const int frame_id) {
  // have to remove control inputs to the invariant node from the same frame
  // when moving this node out of this frame
  for (int i = 0; i < node->input_size(); ++i) {
    if (IsControlInput(node->input(i))) {
      node->mutable_input()->SwapElements(i, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    }
  }
  if (num_outputs == 0) {
    return Status::OK();
  }

  DataTypeVector input_types;
  DataTypeVector output_types;
  OpRegistryInterface* op_registry = OpRegistry::Global();
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(op_registry->LookUp(node->op(), &op_reg_data));
  TF_RETURN_IF_ERROR(InOutTypesForNode(*node, op_reg_data->op_def, &input_types,
                                       &output_types));

  auto consumers = node_map_->GetOutputs(node->name());
  string fname = invariant_enters_[frame_id][0]->attr().at("frame_name").s();
  int piterations =
      invariant_enters_[frame_id][0]->attr().at("parallel_iterations").i();
  for (auto* consumer : consumers) {
    if (!invariant_nodes_.count(consumer)) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        int port;
        string node_name = ParseNodeName(consumer->input(i), &port);
        if (node_name != node->name()) {
          continue;
        }
        if (port < 0) {
          return errors::InvalidArgument(
              "Invariant node should not have control outputs "
              "to variant node");
        }
        DataType output_type = output_types[port];
        NodeDef* new_enter = optimized_graph_->add_node();
        new_enter->set_op("Enter");
        new_enter->set_device(node->device());
        new_enter->set_name(AddPrefixToNodeName(
            StrCat(fname, "_enter_", new_enter_id_++), kLoopOptimizer));
        AttrValue data_type;
        data_type.set_type(output_type);
        new_enter->mutable_attr()->insert({"T", data_type});
        AttrValue frame_name;
        frame_name.set_s(fname);
        new_enter->mutable_attr()->insert({"frame_name", frame_name});
        AttrValue is_const;
        is_const.set_b(true);
        new_enter->mutable_attr()->insert({"is_constant", is_const});
        AttrValue parallel_iterations;
        parallel_iterations.set_i(piterations);
        new_enter->mutable_attr()->insert(
            {"parallel_iterations", parallel_iterations});
        new_enter->add_input(consumer->input(i));
        *consumer->mutable_input(i) = new_enter->name();
        node_map_->AddNode(new_enter->name(), new_enter);
        node_map_->AddOutput(node->name(), new_enter->name());
        node_map_->AddOutput(new_enter->name(), consumer->name());
      }
    }
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::MoveInvariantNodes(
    const int frame_id) {
  for (auto iter = invariant_nodes_.begin(); iter != invariant_nodes_.end();
       ++iter) {
    auto* invariant_node = iter->first;
    const int num_outputs = iter->second;
    if (IsEnter(*invariant_node)) {
      TF_RETURN_IF_ERROR(HandleInvariantEnter(invariant_node, num_outputs));
    } else if (IsConstant(*invariant_node)) {
      TF_RETURN_IF_ERROR(HandleConst(invariant_node, num_outputs, frame_id));
    } else {
      TF_RETURN_IF_ERROR(
          HandleInvariantNode(invariant_node, num_outputs, frame_id));
    }
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::RevertInvariantNodes() {
  std::deque<const NodeDef*> reverted_nodes;
  for (auto iter = invariant_nodes_.begin(); iter != invariant_nodes_.end();) {
    bool erased = false;
    const auto* node = iter->first;
    if (!IsConstant(*node) && !IsEnter(*node) && iter->second > 0) {
      auto& consumers = node_map_->GetOutputs(node->name());
      for (auto* consumer : consumers) {
        if (!invariant_nodes_.count(consumer)) {
          for (const auto& input : consumer->input()) {
            if (IsControlInput(input) && NodeName(input) == node->name()) {
              reverted_nodes.push_back(node);
              invariant_nodes_.erase(iter++);
              erased = true;
              break;
            }
          }
          if (erased) break;
        }
      }
    }
    if (!erased) ++iter;
  }
  while (!reverted_nodes.empty()) {
    const auto* node = reverted_nodes.front();
    reverted_nodes.pop_front();
    std::set<NodeDef*> producers;
    for (const auto& input : node->input()) {
      auto* producer = node_map_->GetNode(input);
      auto iter = invariant_nodes_.find(producer);
      if (iter != invariant_nodes_.end()) {
        if (IsControlInput(input) && !IsConstant(*producer) &&
            !IsEnter(*producer)) {
          reverted_nodes.push_back(producer);
          invariant_nodes_.erase(iter);
        } else {
          producers.insert(producer);
        }
      }
    }
    for (auto* producer : producers) {
      auto iter = invariant_nodes_.find(producer);
      if (iter != invariant_nodes_.end()) {
        ++iter->second;
      }
    }
    for (auto* consumer : node_map_->GetOutputs(node->name())) {
      auto iter = invariant_nodes_.find(consumer);
      if (iter != invariant_nodes_.end()) {
        reverted_nodes.push_back(consumer);
        invariant_nodes_.erase(iter);
      }
    }
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::FindInvariantNodes(
    NodeDef* start_node) {
  std::vector<NodeDef*> stack;
  stack.reserve(32);
  stack.push_back(start_node);
  while (!stack.empty()) {
    NodeDef* node = stack.back();
    stack.pop_back();
    auto consumers = node_map_->GetOutputs(node->name());
    invariant_nodes_.emplace(node, consumers.size());
    for (auto* consumer : consumers) {
      if (invariant_nodes_.count(consumer) || ModifiesFrameInfo(*consumer)) {
        continue;
      }
      bool is_invariant = true;
      for (const auto& input : consumer->input()) {
        if (!IsControlInput(input)) {
          const string name = NodeName(input);
          auto* producer = node_map_->GetNode(name);
          if (!invariant_nodes_.count(producer)) {
            if (IsConstant(*producer)) {
              invariant_nodes_.insert(
                  std::make_pair(producer, node_map_->GetOutputs(name).size()));
            } else {
              is_invariant = false;
              break;
            }
          }
        }
      }
      if (is_invariant) {
        std::set<NodeDef*> producers;
        for (const auto& input : consumer->input()) {
          auto* producer = node_map_->GetNode(input);
          producers.insert(producer);
        }
        for (auto* producer : producers) {
          auto iter = invariant_nodes_.find(producer);
          if (iter != invariant_nodes_.end()) {
            --iter->second;
          }
        }
        stack.push_back(consumer);
      }
    }
  }
  return Status::OK();
}

Status LoopInvariantNodeMotionOptimizer::Optimize() {
  node_map_.reset(new NodeMap(optimized_graph_));
  FrameView frame_view;
  // TODO(ezhulenev): Use GraphView when migrated from NodeMap.
  TF_RETURN_IF_ERROR(frame_view.InferFromGraph(*optimized_graph_));

  frame_parent_.resize(frame_view.num_frames(), -1);
  frame_children_.resize(frame_view.num_frames());
  std::deque<int> worklist;
  for (const NodeDef& node : optimized_graph_->node()) {
    const std::vector<int>& frame_ids = frame_view.Frames(node);

    if (frame_ids.size() >= 3) {
      for (unsigned int i = 1; i < frame_ids.size() - 1; ++i) {
        frame_parent_[frame_ids[i]] = frame_ids[i - 1];
        frame_children_[frame_ids[i]].insert(frame_ids[i + 1]);
      }
    }
    if (frame_ids.size() >= 2) {
      frame_children_[frame_ids[0]].insert(frame_ids[1]);
      frame_parent_[frame_ids.back()] = frame_ids[frame_ids.size() - 2];
    }
    if (!frame_ids.empty()) {
      frame_children_[frame_ids.back()] = empty_set_;
      if (node.op() == "LoopCond") {
        if (loop_cond_.count(frame_ids.back())) {
          return errors::InvalidArgument(
              "Loop ", frame_ids.back(),
              " has more than one LoopCond node: ", node.name(), " and ",
              loop_cond_[frame_ids.back()]->name());
        }
        loop_cond_[frame_ids.back()] = &node;
      }
      if (IsEnter(node) && node.attr().at("is_constant").b()) {
        invariant_enters_[frame_ids.back()].push_back(
            const_cast<NodeDef*>(&node));
      }
    }
  }

  for (size_t i = 0; i < frame_children_.size(); i++) {
    if (frame_children_[i].empty()) {
      worklist.push_back(i);
    }
  }

  while (!worklist.empty()) {
    int frame_id = worklist.front();
    new_enter_id_ = 0;
    worklist.pop_front();
    if (frame_parent_[frame_id] != -1) {
      int parent_id = frame_parent_[frame_id];
      frame_children_[parent_id].erase(frame_id);
      if (frame_children_[parent_id].empty()) {
        worklist.push_back(parent_id);
      }
    }

    if (invariant_enters_[frame_id].empty()) {
      continue;
    }
    invariant_nodes_.clear();
    for (auto* enter : invariant_enters_[frame_id]) {
      TF_RETURN_IF_ERROR(FindInvariantNodes(enter));
    }

    // revert invariant nodes that have control outputs to variant nodes
    TF_RETURN_IF_ERROR(RevertInvariantNodes());

    TF_RETURN_IF_ERROR(MoveInvariantNodes(frame_id));
  }
  return Status::OK();
}

std::vector<int> GetStackPushNodesToConvert(
    const GraphTopologyView& graph_view,
    const std::unordered_set<string>& nodes_to_preserve, int stack_node_idx) {
  VLOG(1) << "Stack node: " << graph_view.graph()->node(stack_node_idx).name();

  const std::unordered_set<string> op_types_to_traverse(
      {"Stack", "StackV2", "Enter", "RefEnter", "Switch", "RefSwitch",
       "_SwitchN", "Identity", "RefIdentity"});
  const auto is_op_to_traverse = [&](const NodeDef* node) -> bool {
    return op_types_to_traverse.find(node->op()) != op_types_to_traverse.end();
  };

  std::vector<int> nodes_to_convert;
  std::vector<int> fanouts;

  DfsTraversal(graph_view, {graph_view.GetNode(stack_node_idx)},
               TraversalDirection::kFollowOutputs,
               DfsPredicates::Advance(is_op_to_traverse),
               DfsCallbacks::PreOrder([&](const NodeDef* node) {
                 const absl::optional<int> idx = graph_view.GetNodeIndex(*node);
                 fanouts.push_back(idx.value());
               }));

  for (int fanout_idx : fanouts) {
    const NodeDef& fanout_node = graph_view.graph()->node(fanout_idx);
    VLOG(1) << "Fanout " << fanout_idx << " : " << fanout_node.name();
    if (IsStackPushOp(fanout_node)) {
      // Check that the stack itself is not a node we want to preserve. This can
      // happen when the graph we have contains only the forward pass for a loop
      // (as when the forward and backward passes are split across different
      // functions).
      if (graph_view.HasNode(fanout_node.input(0))) {
        const NodeDef* stack_node = graph_view.GetNode(fanout_node.input(0));
        while (stack_node->op() != "Stack" && stack_node->op() != "StackV2" &&
               stack_node->input_size() > 0 &&
               graph_view.HasNode(stack_node->input(0))) {
          stack_node = graph_view.GetNode(stack_node->input(0));
        }
        if (nodes_to_preserve.find(stack_node->name()) ==
            nodes_to_preserve.end()) {
          nodes_to_convert.push_back(fanout_idx);
        }
      } else {
        nodes_to_convert.push_back(fanout_idx);
      }
    } else if (IsStackOp(fanout_node) || IsStackCloseOp(fanout_node) ||
               op_types_to_traverse.find(fanout_node.op()) !=
                   op_types_to_traverse.end()) {
      continue;
    } else if (!IsStackPopOp(fanout_node) ||
               (!graph_view.GetFanout(fanout_idx).empty() ||
                nodes_to_preserve.find(fanout_node.name()) !=
                    nodes_to_preserve.end())) {
      // The node is either a stack pop with consumers or something unexpected
      // so we leave the graph alone.
      nodes_to_convert.clear();
      break;
    }
  }

  return nodes_to_convert;
}

Status RemoveStackOps(const std::unordered_set<string>& nodes_to_preserve,
                      GraphDef* optimized_graph) {
  NodeMap node_map(optimized_graph);
  GraphTopologyView graph_view;
  TF_RETURN_IF_ERROR(graph_view.InitializeFromGraph(*optimized_graph));

  for (int node_idx = 0; node_idx < optimized_graph->node_size(); ++node_idx) {
    if (IsStackOp(optimized_graph->node(node_idx))) {
      for (int push_node_idx : GetStackPushNodesToConvert(
               graph_view, nodes_to_preserve, node_idx)) {
        // We found push nodes without corresponding pops. Convert them to
        // Identity passing the data through and add a control dependency from
        // the op supplying the stack handle.
        NodeDef* push_node = optimized_graph->mutable_node(push_node_idx);
        VLOG(1) << "Converting " << push_node_idx << " : "
                << push_node->DebugString();
        if (push_node->attr().count("swap_memory") != 0) {
          push_node->mutable_attr()->erase("swap_memory");
        }
        push_node->set_op("Identity");
        push_node->mutable_input()->SwapElements(0, 1);
        const string ctrl_dep = ConstantFolding::AddControlDependency(
            push_node->input(1), optimized_graph, &node_map);
        push_node->set_input(1, ctrl_dep);
        VLOG(1) << "After converting: " << push_node->DebugString();
      }
    }
  }
  return Status::OK();
}

bool IsSimpleBinaryOperator(const NodeDef& node) {
  return (IsLess(node) || IsLessEqual(node) || IsGreater(node) ||
          IsGreaterEqual(node) || IsEqual(node));
}

Status EvaluateBoolOpForConstantOperands(const NodeDef& op_node,
                                         const NodeDef& constant_operand_0,
                                         const NodeDef& constant_operand_1,
                                         DeviceBase* cpu_device,
                                         ResourceMgr* resource_mgr,
                                         bool* value) {
  VLOG(4) << "Evaluate bool op: op_node=" << op_node.name()
          << " input0=" << constant_operand_0.name()
          << " input1=" << constant_operand_1.name();
  TensorVector inputs;

  const TensorProto& raw_val_0 = constant_operand_0.attr().at("value").tensor();
  Tensor value_0(raw_val_0.dtype(), raw_val_0.tensor_shape());
  CHECK(value_0.FromProto(raw_val_0));
  inputs.emplace_back(&value_0);
  const TensorProto& raw_val_1 = constant_operand_1.attr().at("value").tensor();
  Tensor value_1(raw_val_1.dtype(), raw_val_1.tensor_shape());
  CHECK(value_1.FromProto(raw_val_1));
  inputs.emplace_back(&value_1);

  TensorVector outputs;
  TF_RETURN_IF_ERROR(
      EvaluateNode(op_node, inputs, cpu_device, resource_mgr, &outputs));

  if (outputs.size() != 1 || outputs[0].tensor == nullptr) {
    return Status(error::INVALID_ARGUMENT, "Expected one output.");
  }
  *value = outputs[0].tensor->scalar<bool>()();
  delete outputs[0].tensor;

  return Status::OK();
}

// TODO(lyandy): Consolidate with ConstantFolding implementation.
bool IsReallyConstant(const NodeDef& node,
                      const absl::flat_hash_set<string>& feed_nodes) {
  if (!IsConstant(node)) {
    return false;
  }
  // If the node is fed it's not constant anymore.
  return feed_nodes.find(node.name()) == feed_nodes.end();
}

Status CheckForDeadFanout(const MutableGraphView& view,
                          const NodeDef& switch_node, const NodeMap& node_map,
                          const absl::flat_hash_set<string>& feed_nodes,
                          DeviceBase* cpu_device, ResourceMgr* resource_mgr,
                          bool* has_dead_fanout, int* dead_fanout) {
  *has_dead_fanout = false;
  GraphView::InputPort switch_loopcond_port(&switch_node, 1);
  const NodeDef* switch_predicate =
      view.GetRegularFanin(switch_loopcond_port).node;

  // CASE 1: Control is a constant.
  if (IsReallyConstant(*switch_predicate, feed_nodes)) {
    VLOG(3) << "Found switch node with constant predicate:"
            << " switch_node=" << switch_node.name()
            << " switch_predicate=" << switch_predicate->name();
    Tensor selector;
    CHECK(selector.FromProto(switch_predicate->attr().at("value").tensor()));
    *has_dead_fanout = true;
    *dead_fanout = selector.scalar<bool>()() ? 0 : 1;
    return Status::OK();
  }

  GraphView::InputPort switch_input_port(&switch_node, 0);
  const NodeDef* switch_input = view.GetRegularFanin(switch_input_port).node;

  // CASE 2: Zero-iteration while loop.
  // We check if its a while loop such that the condition is a simple binary
  // operator which returns false for the initialization value.
  // TODO(srjoglekar): Improve to work with arbitrary predicate subgraphs.
  if (!IsMerge(*switch_input) || !IsLoopCond(*switch_predicate)) {
    return Status::OK();
  }

  VLOG(3) << "Try to find a zero iteration while loop:"
          << " switch_node=" << switch_node.name();

  // Find the boolean predicate from a LoopCond node (e.g. Greater).
  NodeDef* switch_ctrl_node = view.GetRegularFanin({switch_predicate, 0}).node;
  if (!switch_ctrl_node || !IsSimpleBinaryOperator(*switch_ctrl_node)) {
    return Status::OK();
  }

  // Find the Merge node & the Constant Operand to the condition node, if
  // available.
  NodeDef* merge_node = nullptr;
  NodeDef* constant_ctrl_input = nullptr;
  int constant_index = 0;
  for (int i = 0; i < switch_ctrl_node->input().size(); ++i) {
    const string& input = switch_ctrl_node->input(i);
    if (IsControlInput(input)) continue;

    NodeDef* node = view.GetNode(switch_ctrl_node->input(i));
    if (IsMerge(*node)) {
      merge_node = node;
    }
    if (IsReallyConstant(*node, feed_nodes)) {
      constant_ctrl_input = node;
      constant_index = i;
    }
  }
  if (merge_node == nullptr || constant_ctrl_input == nullptr) {
    return Status::OK();
  }

  // Find the initialization constant (via Enter, if one exists).
  NodeDef* enter_node = nullptr;
  NodeDef* constant_init_node = nullptr;
  for (const auto& input : merge_node->input()) {
    NodeDef* node = node_map.GetNode(input);
    if (IsEnter(*node)) {
      enter_node = node;
    }
    if (IsReallyConstant(*node, feed_nodes)) {
      constant_init_node = node;
    }
  }
  if (enter_node != nullptr) {
    if (constant_init_node != nullptr) return Status::OK();
    for (const auto& input : enter_node->input()) {
      NodeDef* node = node_map.GetNode(input);
      if (IsReallyConstant(*node, feed_nodes)) {
        constant_init_node = node;
      }
    }
  }
  if (constant_init_node == nullptr) {
    return Status::OK();
  }

  VLOG(4) << "Check if loop will be 0 iterations:"
          << "\n|  switch_node        : " << switch_node.name()
          << "\n|  switch_ctrl_node   : " << switch_ctrl_node->name()
          << "\n|  merge_node         : " << merge_node->name()
          << "\n|  constant_ctrl_input: " << constant_ctrl_input->name()
          << "\n|  enter_node         : "
          << (enter_node ? enter_node->name() : "<n/a>")
          << "\n|  constant_init_node : " << constant_init_node->name();

  // Check if there will be 0 iterations. This will only happen if the condition
  // evaluates to false with respect to the initialization value.
  NodeDef* operand_0 =
      constant_index ? constant_init_node : constant_ctrl_input;
  NodeDef* operand_1 =
      constant_index ? constant_ctrl_input : constant_init_node;
  bool constant_switch_value;
  TF_RETURN_IF_ERROR(EvaluateBoolOpForConstantOperands(
      *switch_ctrl_node, *operand_0, *operand_1, cpu_device, resource_mgr,
      &constant_switch_value));

  if (constant_switch_value == false) {
    VLOG(4) << "Remove 0 iteration while loop:"
            << " switch_node=" << switch_node.name();
    *has_dead_fanout = true;
    *dead_fanout = 1;
  } else {
    VLOG(4) << "Was not able to prove that loop has 0 iterations.";
  }
  return Status::OK();
}

}  // namespace

LoopOptimizer::LoopOptimizer()
    : opt_level_(RewriterConfig::ON),
      cpu_device_(nullptr),
      options_(LoopOptimizerOptions::Default(RewriterConfig::ON)) {}

LoopOptimizer::LoopOptimizer(RewriterConfig::Toggle opt_level,
                             DeviceBase* cpu_device)
    : opt_level_(opt_level),
      cpu_device_(cpu_device),
      options_(LoopOptimizerOptions::Default(RewriterConfig::ON)) {
  resource_mgr_.reset(new ResourceMgr());
}

Status LoopOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  if (!options_.enable_loop_invariant_node_motion &&
      !options_.enable_stack_push_removal &&
      !options_.enable_dead_branch_removal) {
    return errors::Aborted("Nothing to do.");
  }
  *optimized_graph = item.graph;
  // Set up helper data structures.
  if (options_.enable_loop_invariant_node_motion) {
    LoopInvariantNodeMotionOptimizer linm_optimizer(optimized_graph);
    TF_RETURN_IF_ERROR(linm_optimizer.Optimize());
  }
  if (options_.enable_stack_push_removal) {
    TF_RETURN_IF_ERROR(RemoveStackOps(item.NodesToPreserve(), optimized_graph));
  }
  if (options_.enable_dead_branch_removal) {
    // TODO(srjoglekar): Figure out if we can optimize NodeMap creations across
    // optimizer passes.
    NodeMap node_map(optimized_graph);
    absl::flat_hash_set<string> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(NodeName(feed.first));
    }
    TF_RETURN_IF_ERROR(RemoveDeadBranches(item.NodesToPreserve(), node_map,
                                          feed_nodes, optimized_graph));
  }

  return Status::OK();
}

Status LoopOptimizer::RemoveDeadBranches(
    const std::unordered_set<string>& nodes_to_preserve,
    const NodeMap& node_map, const absl::flat_hash_set<string>& feed_nodes,
    GraphDef* optimized_graph) {
  std::unordered_set<const NodeDef*> dead_nodes;
  std::unordered_map<NodeDef*, std::set<int>> dead_merge_inputs;
  // TODO(bsteiner): also rewrite switches as identity. For now we just record
  // them
  absl::flat_hash_set<GraphView::OutputPort> identity_switches;

  MutableGraphView view(optimized_graph);
  for (const NodeDef& node : optimized_graph->node()) {
    if (!IsSwitch(node)) {
      continue;
    }
    if (node.op() == "_SwitchN") {  // _SwitchN not used in loop control flow.
      continue;
    }
    if (nodes_to_preserve.find(node.name()) != nodes_to_preserve.end()) {
      continue;
    }

    int dead_fanout;
    bool has_dead_fanout;
    TF_RETURN_IF_ERROR(CheckForDeadFanout(view, node, node_map, feed_nodes,
                                          cpu_device_, resource_mgr_.get(),
                                          &has_dead_fanout, &dead_fanout));
    if (!has_dead_fanout) {
      continue;
    }
    GraphView::OutputPort dead(&node, dead_fanout);
    identity_switches.insert(dead);

    SetVector<MutableGraphView::InputPort, absl::Hash<MutableGraphView::Port>>
        zombie_inputs;
    for (const MutableGraphView::InputPort& port : view.GetFanout(dead)) {
      if (dead_nodes.find(port.node) == dead_nodes.end()) {
        zombie_inputs.PushBack(port);
      }
    }
    // If we encounter a single node that must be preserved in the fanout of the
    // switch node we need to preserve the entire switch fanout: we therefore
    // work on a local copy that only gets committed to the master copy once the
    // whole fanout has been explored.
    std::unordered_set<const NodeDef*> local_dead_nodes = dead_nodes;
    std::unordered_map<NodeDef*, std::set<int>> local_dead_merge_inputs =
        dead_merge_inputs;
    bool found_node_to_preserve = false;
    while (!found_node_to_preserve && !zombie_inputs.Empty()) {
      MutableGraphView::InputPort dead = zombie_inputs.PopBack();
      if (nodes_to_preserve.find(dead.node->name()) !=
          nodes_to_preserve.end()) {
        found_node_to_preserve = true;
        break;
      }

      if (local_dead_nodes.find(dead.node) != local_dead_nodes.end()) {
        continue;
      }

      if (IsMerge(*dead.node)) {
        const int num_data_inputs = dead.node->attr().at("N").i();
        if (num_data_inputs > 2) {
          // This can happen with _SwitchN/Merge (Case lowering). We skip these
          // to simplify the code for now.
          found_node_to_preserve = true;
          break;
        }
        MutableGraphView::OutputPort value_index(dead.node, 1);
        const absl::flat_hash_set<MutableGraphView::InputPort>& index_fanout =
            view.GetFanout(value_index);
        if (!index_fanout.empty()) {
          // The 2nd output (that indicates which input is propagated) is
          // connected. This never happens in practice, so we'll just skip this
          // case to simplify the code for now.
          found_node_to_preserve = true;
          break;
        }

        bool fully_dead = false;
        // Merge node can become real dead only if all data inputs are dead.
        // Merge always waits for all control edges, but they do not
        // change the node deadness.
        if (dead.port_id >= 0) {
          local_dead_merge_inputs[dead.node].insert(dead.port_id);
          if (local_dead_merge_inputs[dead.node].size() == num_data_inputs) {
            fully_dead = true;
          }
        } else {
          // Keep track of all Merge nodes, even if they do not have dead data
          // inputs. We'll need to cleanup dead control edges for them later.
          local_dead_merge_inputs.insert({dead.node, {}});
        }
        if (fully_dead) {
          local_dead_merge_inputs.erase(dead.node);
          local_dead_nodes.insert(dead.node);
          for (const MutableGraphView::InputPort& port :
               view.GetFanouts(*dead.node, true)) {
            zombie_inputs.PushBack(port);
          }
        }
      } else if (dead.node->op() == "ControlTrigger") {
        // Control trigger have different semantic, so don't touch them
        found_node_to_preserve = true;
        break;
      } else {
        if (local_dead_nodes.insert(dead.node).second) {
          for (const MutableGraphView::InputPort& dead_fanout :
               view.GetFanouts(*dead.node, true)) {
            zombie_inputs.PushBack(dead_fanout);
          }
        }
      }
    }
    if (!found_node_to_preserve) {
      std::swap(dead_nodes, local_dead_nodes);
      std::swap(dead_merge_inputs, local_dead_merge_inputs);
    }
  }

  std::vector<int> nodes_idx_to_delete;
  nodes_idx_to_delete.reserve(dead_nodes.size());
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    if (dead_nodes.count(&optimized_graph->node(i)))
      nodes_idx_to_delete.push_back(i);
  }

  // Names of the nodes that were removed from the graph.
  absl::flat_hash_set<absl::string_view> dead_node_names;
  dead_node_names.reserve(dead_nodes.size());
  for (const NodeDef* dead_node : dead_nodes)
    dead_node_names.insert(dead_node->name());

  // Remove dead inputs from Merge nodes that were not pruned from the graph.
  for (const auto& itr : dead_merge_inputs) {
    NodeDef* dead_node = itr.first;
    if (dead_nodes.find(dead_node) != dead_nodes.end()) {
      // The node has been pruned since all its inputs are dead.
      continue;
    }
    // Remove dead data input.
    const std::set<int>& dead_inputs = itr.second;
    CHECK_LE(dead_inputs.size(), 1);
    // (This loop would delete >1 items possibly in the wrong order.)
    for (int index : dead_inputs) {
      dead_node->mutable_input()->DeleteSubrange(index, 1);
    }
    // Turn Merge into Identity only if we deleted the other data input.
    if (!dead_inputs.empty()) {
      const int num_data_inputs = dead_node->attr().at("N").i();
      CHECK_EQ(num_data_inputs, dead_inputs.size() + 1);
      dead_node->set_op("Identity");
      dead_node->mutable_attr()->erase("N");
    }
    // Remove control inputs from dead nodes.
    int pos = 0;
    while (pos < dead_node->input_size()) {
      TensorId tensor = ParseTensorName(dead_node->input(pos));
      if (tensor.index() == Graph::kControlSlot &&
          dead_node_names.contains(tensor.node())) {
        auto* inputs = dead_node->mutable_input();
        inputs->SwapElements(pos, dead_node->input_size() - 1);
        inputs->RemoveLast();
      } else {
        ++pos;
      }
    }
  }

  EraseNodesFromGraph(std::move(nodes_idx_to_delete), optimized_graph);

  return Status::OK();
}

void LoopOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& optimize_output, double result) {
  // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
