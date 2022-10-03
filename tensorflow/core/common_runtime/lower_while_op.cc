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

#include "tensorflow/core/common_runtime/lower_while_op.h"

#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

namespace {

using NodeOut = NodeBuilder::NodeOut;

constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

// Helper to convert a functional While op to its lowered form.
//
// Example:
//
// Input graph:
//
// loop_var -> WhileOp<cond_func, body_func> -> consumer
//
// Output graph(top to down flow):
//
//                   loop_var
//                      |
//                    Enter
//                      |
//  cond_func ---<--- Merge  ---<--- NextIteration
//      |               |                |
//      V               V                ^
//      |               |                |
//  LoopCond  --->--- Switch --->--- body_func
//                      |
//                     Exit
//                      |
//                   consumer
//
// DT_RESOURCE tensors are handled specially:
//
// resource_loop_var -> Enter[is_constant=True] -> cond_func and body_func
//      |
//      V
//   consumer
class LowerWhileHelper {
 public:
  static Status Run(Node* while_op, const NameAttrList& cond_fn,
                    const NameAttrList& body_fn, int parallel_iterations,
                    Graph* graph, const FunctionLibraryDefinition* flib_def,
                    bool keep_node_fetchable) {
    LowerWhileHelper helper(while_op, cond_fn, body_fn, parallel_iterations,
                            graph, flib_def, keep_node_fetchable);
    return helper.RunInternal();
  }

 private:
  // Create a LowerWhileHelper to create the lowering of While op that has cond
  // and body functions named `cond_fn_name` and `body_fn_name` respectively in
  // the given graph.
  LowerWhileHelper(Node* while_op, const NameAttrList& cond_fn,
                   const NameAttrList& body_fn, int parallel_iterations,
                   Graph* graph, const FunctionLibraryDefinition* flib_def,
                   bool keep_node_fetchable);

  Status RunInternal();

  void InitializeInputOutputToLoweredNodeMap();

  // Creates an Enter node for each `while_op_` input and adds them to
  // `enter_nodes_`. If the `while_op_` has an incoming control edge from a
  // `src` node we add a control edge from `src` to each Enter node.
  Status CreateEnterNodes();

  // Creates a Merge node for each Enter node and adds to `merge_nodes_`.
  // Initially now both inputs of a Merge node are the Enter node. Input at
  // index 1 is later updated to the output of NextIteration node in
  // `UpdateMergeNodes`.
  Status CreateMergeNodes();

  // Creates the call node for cond func and stores in `cond_call_node_`.
  Status CreateCondFuncCallNode();

  // Creates a Switch node for each loop var and adds to `switch_nodes_`.
  // Output at index 1(true) of a Switch node is fed into the loop body.
  // Output at index 0(false) of a Switch node is fed into the Exit nodes.
  Status CreateSwitchNodes();

  // Creates the call node for body func and stores in `body_call_node_`.
  Status CreateBodyFuncCallNode();

  // Creates an Exit node for each loop var and adds to `exit_nodes_`. These
  // are fed into the consumers of the `while_op_`.
  Status CreateExitNodes();

  // Creates an NextIteration node for each loop var and adds to
  // `next_iteration_nodes_`.
  Status CreateNextIterationNodes();

  // Updates input at index 1 of each merge node created in `CreateMergeNodes`
  // to use the output of NextIteration node created in
  // `CreateNextIterationNodes` instead.
  Status UpdateMergeNodes();

  // Updates consumers of the original `while_op_` to instead use the outputs
  // from the exit nodes in `exit_nodes_`. Also updates any outgoing control
  // edges to depend on `lowered_while_executed_` instead.
  Status UpdateConsumers();

  // Returns unique name containing the name of the While op being rewritten
  // (name_), infix and a suffix to ensure it is unique within the graph.
  string NewName(const string& infix);

  // Returns whether the While op's input/output at `index` is a `DT_RESOURCE`.
  bool IsResource(int index);

  // The original While op.
  Node* while_op_;
  // The call node for the cond branch.
  Node* cond_call_node_;
  // The LoopCond node specifying the loop termination condition.
  Node* loop_cond_node_;
  // The call node for the body branch.
  Node* body_call_node_;
  // The node with the same name as the original While op:
  //   (a) IdentityN node with same outputs if 'keep_node_fetchable_ == true'.
  //   (b) NoOp node with control edge from 'lowered_while_executed_' otherwise.
  Node* lowered_while_output_;
  // The NoOp node with control edges from all Exit nodes. This node will be
  // used as a source of outgoing control edges from lowered While node.
  Node* lowered_while_executed_;
  Graph* graph_;
  const FunctionLibraryDefinition* flib_def_;
  // Name of the `while_op_`.
  string name_;
  // Max number of parallel_iterations for the while loop.
  const int parallel_iterations_;
  bool keep_node_fetchable_;

  NodeDebugInfo debug_info_;
  NodeBuilder cond_call_builder_;
  NodeBuilder body_call_builder_;

  // `Enter` nodes, one per loop input/output.
  // Note: `Enter` nodes with type `DT_RESOURCE` have attr `is_constant=True`.
  std::vector<Node*> enter_nodes_;

  // Merge/Switch/NextIteration/Exit nodes, one per non-resource loop
  // input/output.
  std::vector<Node*> merge_nodes_;
  std::vector<Node*> switch_nodes_;
  std::vector<Node*> exit_nodes_;
  std::vector<Node*> next_iterations_nodes_;
  // Maps from the loop input/output indices to their corresponding
  // Merge/Switch/NextIteration/Exit node indices. For inputs/outputs of
  // `DT_RESOURCE` type there are no Merge/Switch/NextIteration/Exit nodes
  // in which case the mapping contains -1.
  std::vector<int> op_input_output_to_lowered_node_;

  size_t num_loop_inputs_;
};

LowerWhileHelper::LowerWhileHelper(Node* while_op, const NameAttrList& cond_fn,
                                   const NameAttrList& body_fn,
                                   int parallel_iterations, Graph* graph,
                                   const FunctionLibraryDefinition* flib_def,
                                   bool keep_node_fetchable)
    : while_op_(while_op),
      graph_(graph),
      flib_def_(flib_def),
      name_(while_op->name()),
      parallel_iterations_(parallel_iterations),
      keep_node_fetchable_(keep_node_fetchable),
      debug_info_(*while_op_),
      cond_call_builder_(NewName("cond"), cond_fn.name(), flib_def,
                         &debug_info_),
      body_call_builder_(NewName("body"), body_fn.name(), flib_def,
                         &debug_info_),
      num_loop_inputs_(while_op_->num_inputs()) {
  cond_call_builder_.Attr(kLowerAsMultiDeviceFunctionAttr, true);
  for (const auto& i : cond_fn.attr()) {
    cond_call_builder_.Attr(i.first, i.second);
  }
  body_call_builder_.Attr(kLowerAsMultiDeviceFunctionAttr, true);
  for (const auto& i : body_fn.attr()) {
    body_call_builder_.Attr(i.first, i.second);
  }
  // We intentionally `resize` instead of `reserve` space in `enter_nodes_`
  // because we need to set it's elements out of order in `CreateEnterNodes`.
  enter_nodes_.resize(num_loop_inputs_);
  merge_nodes_.reserve(num_loop_inputs_);
  switch_nodes_.reserve(num_loop_inputs_);
  exit_nodes_.reserve(num_loop_inputs_);
  next_iterations_nodes_.reserve(num_loop_inputs_);
  op_input_output_to_lowered_node_.resize(num_loop_inputs_, -1);
}

Status LowerWhileHelper::RunInternal() {
  InitializeInputOutputToLoweredNodeMap();
  TF_RETURN_IF_ERROR(CreateEnterNodes());
  TF_RETURN_IF_ERROR(CreateMergeNodes());
  TF_RETURN_IF_ERROR(CreateCondFuncCallNode());
  TF_RETURN_IF_ERROR(CreateSwitchNodes());
  TF_RETURN_IF_ERROR(CreateBodyFuncCallNode());
  TF_RETURN_IF_ERROR(CreateExitNodes());
  TF_RETURN_IF_ERROR(CreateNextIterationNodes());
  TF_RETURN_IF_ERROR(UpdateMergeNodes());
  TF_RETURN_IF_ERROR(UpdateConsumers());
  return OkStatus();
}

void LowerWhileHelper::InitializeInputOutputToLoweredNodeMap() {
  int counter = 0;
  for (int i = 0; i < num_loop_inputs_; i++) {
    if (!IsResource(i)) {
      op_input_output_to_lowered_node_[i] = counter++;
    }
  }
}

Status LowerWhileHelper::CreateEnterNodes() {
  // Note: `Node::input_edge` runs in  O(num_inputs) so we use
  // `Node::input_edges` instead so that below loop runs in O(num_inputs) time
  // and not O(num_inputs^2).
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(while_op_->input_edges(&edges));
  for (const Edge* edge : edges) {
    Node* enter_node;
    NodeBuilder builder =
        NodeBuilder(NewName("enter"), "Enter", flib_def_, &debug_info_)
            .Input(NodeOut(edge->src(), edge->src_output()))
            .Attr("frame_name", name_)
            .Attr("parallel_iterations", parallel_iterations_)
            .Device(edge->src()->requested_device())
            .AssignedDevice(edge->src()->assigned_device_name());
    if (IsResource(edge->dst_input())) {
      builder.Attr("is_constant", true);
    }
    TF_RETURN_IF_ERROR(builder.Finalize(graph_, &enter_node));
    enter_nodes_[edge->dst_input()] = enter_node;
  }
  // Create a NoOp node that takes incoming control inputs of the original While
  // op as control inputs and use it as a control input for all Enter nodes.
  std::vector<Node*> control_inputs;
  for (const Edge* e : while_op_->in_edges()) {
    if (e->IsControlEdge()) {
      control_inputs.push_back(e->src());
    }
  }
  if (!control_inputs.empty()) {
    Node* incoming_control_node;
    TF_RETURN_IF_ERROR(NodeBuilder(NewName("LoopControlInputs"), "NoOp",
                                   flib_def_, &debug_info_)
                           .ControlInputs(control_inputs)
                           .Device(while_op_->requested_device())
                           .Finalize(graph_, &incoming_control_node));
    for (Node* n : enter_nodes_) {
      graph_->AddControlEdge(incoming_control_node, n);
    }
  }
  return OkStatus();
}

Status LowerWhileHelper::CreateMergeNodes() {
  for (Node* enter_node : enter_nodes_) {
    if (enter_node->output_type(0) == DT_RESOURCE) {
      continue;
    }
    Node* merge_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName("merge"), "Merge", flib_def_, &debug_info_)
            .Input({NodeOut(enter_node, 0), NodeOut(enter_node, 0)})
            .Device(enter_node->requested_device())
            .AssignedDevice(enter_node->assigned_device_name())
            .Finalize(graph_, &merge_node));
    merge_nodes_.emplace_back(merge_node);
  }
  return OkStatus();
}

Status LowerWhileHelper::CreateCondFuncCallNode() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    if (IsResource(i)) {
      cond_call_builder_.Input(NodeOut(enter_nodes_[i], 0));
    } else {
      cond_call_builder_.Input(
          NodeOut(merge_nodes_[op_input_output_to_lowered_node_[i]], 0));
    }
  }
  cond_call_builder_.Device(while_op_->requested_device());
  TF_RETURN_IF_ERROR(cond_call_builder_.Finalize(graph_, &cond_call_node_));
  // Add a control edge to make sure the Const nodes in the cond function
  // are in the same frame as the rest of the function, otherwise
  // `BuildControlFlowInfo` throws an error.
  graph_->AddControlEdge(merge_nodes_[0], cond_call_node_);
  TF_RETURN_IF_ERROR(
      NodeBuilder(NewName("LoopCond"), "LoopCond", flib_def_, &debug_info_)
          .Input(NodeOut(cond_call_node_, 0))
          .Device(while_op_->requested_device())
          .Finalize(graph_, &loop_cond_node_));
  return OkStatus();
}

Status LowerWhileHelper::CreateSwitchNodes() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    if (IsResource(i)) {
      continue;
    }
    string op_name;
    {
      const Node* input_node;
      TF_RETURN_IF_ERROR(while_op_->input_node(i, &input_node));
      op_name = strings::StrCat(input_node->name(), "_switch");
    }
    Node* merge_node = merge_nodes_[op_input_output_to_lowered_node_[i]];
    Node* switch_node;
    string op_type = "Switch";
    if (IsRefType(merge_node->output_type(0))) {
      op_type = "RefSwitch";
    }
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName(op_name), op_type, flib_def_, &debug_info_)
            .Input(NodeOut(merge_node, 0))
            .Input(NodeOut(loop_cond_node_, 0))
            .Device(merge_node->requested_device())
            .AssignedDevice(merge_node->assigned_device_name())
            .Finalize(graph_, &switch_node));
    switch_nodes_.emplace_back(switch_node);
  }
  return OkStatus();
}

Status LowerWhileHelper::CreateBodyFuncCallNode() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    if (IsResource(i)) {
      body_call_builder_.Input(NodeOut(enter_nodes_[i], 0));
    } else {
      body_call_builder_.Input(
          NodeOut(switch_nodes_[op_input_output_to_lowered_node_[i]], 1));
    }
  }
  body_call_builder_.Device(while_op_->requested_device());
  TF_RETURN_IF_ERROR(body_call_builder_.Finalize(graph_, &body_call_node_));
  // Add a control edge to make sure the Const nodes in the body function
  // are in the same frame as the rest of the function, otherwise
  // `BuildControlFlowInfo` throws an error.
  // TODO(srbs): The choice of input at index 0 seems arbitrary(is it?) however
  // this is how tf.while_loop does it. Can this affect performance if the 0th
  // node is not the first one to be ready? Can we speed that case up using some
  // sort of multi-input Merge?
  Node* body_control_node_;
  string op_type = "Identity";
  if (IsRefType(switch_nodes_[0]->output_type(1))) {
    op_type = "RefIdentity";
  }
  TF_RETURN_IF_ERROR(NodeBuilder(NewName("loop_body_control"), op_type,
                                 flib_def_, &debug_info_)
                         .Input(NodeOut(switch_nodes_[0], 1))
                         .Device(while_op_->requested_device())
                         .Finalize(graph_, &body_control_node_));
  graph_->AddControlEdge(body_control_node_, body_call_node_);
  return OkStatus();
}

Status LowerWhileHelper::CreateExitNodes() {
  std::vector<NodeOut> outputs;
  outputs.reserve(num_loop_inputs_);
  for (int i = 0; i < num_loop_inputs_; i++) {
    if (IsResource(i)) {
      // Note(srbs): A resource output of this While should never be used but we
      // need this for the IdentityN node below.
      OutputTensor resource_tensor;
      TF_RETURN_IF_ERROR(enter_nodes_[i]->input_tensor(0, &resource_tensor));
      outputs.emplace_back(resource_tensor);
    } else {
      Node* exit_node;
      TF_RETURN_IF_ERROR(
          NodeBuilder(NewName("exit"), "Exit", flib_def_, &debug_info_)
              .Input(NodeOut(switch_nodes_[op_input_output_to_lowered_node_[i]],
                             0))
              .Device(switch_nodes_[op_input_output_to_lowered_node_[i]]
                          ->requested_device())
              .AssignedDevice(switch_nodes_[op_input_output_to_lowered_node_[i]]
                                  ->assigned_device_name())
              .Finalize(graph_, &exit_node));
      exit_nodes_.emplace_back(exit_node);
      outputs.emplace_back(NodeOut(exit_node, 0));
    }
  }

  // We split data and control outputs of lowered while op, because otherwise
  // after lowering of multi-device loop body we might end up with DT_RESOURCE
  // inputs from multiple devices coming into IdentityN.

  // Add a NoOp node that has control edges from all Exit nodes. This node is
  // used for rewriting control edges with the original while op as src.
  TF_RETURN_IF_ERROR(NodeBuilder(NewName("LoopExecuted"), "NoOp",
                                 OpRegistry::Global(), &debug_info_)
                         .ControlInputs(exit_nodes_)
                         .Device(while_op_->requested_device())
                         .Finalize(graph_, &lowered_while_executed_));

  if (keep_node_fetchable_) {
    // Add an IdentityN node that has the same outputs and same name as the
    // original functional While op. This is used for fetching the output of the
    // While node by name in calls to sess.run.
    TF_RETURN_IF_ERROR(
        NodeBuilder(name_, "IdentityN", OpRegistry::Global(), &debug_info_)
            .Input(outputs)
            .Device(while_op_->requested_device())
            .Finalize(graph_, &lowered_while_output_));
  } else {
    // Even if we don't plan to fetch tensors from the lowered While op, we must
    // keep it a valid source of control edges, because it might be a part of
    // function control output set.
    TF_RETURN_IF_ERROR(
        NodeBuilder(name_, "NoOp", OpRegistry::Global(), &debug_info_)
            .ControlInput(lowered_while_executed_)
            .Device(while_op_->requested_device())
            .Finalize(graph_, &lowered_while_output_));
  }

  return OkStatus();
}

Status LowerWhileHelper::CreateNextIterationNodes() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    Node* next_iteration;
    if (IsResource(i)) {
      continue;
    }
    Node* merge_node = merge_nodes_[op_input_output_to_lowered_node_[i]];
    TF_RETURN_IF_ERROR(NodeBuilder(NewName("next_iteration"), "NextIteration",
                                   flib_def_, &debug_info_)
                           .Input(NodeOut(body_call_node_, i))
                           .ControlInput(body_call_node_)
                           .Device(merge_node->requested_device())
                           .AssignedDevice(merge_node->assigned_device_name())
                           .Finalize(graph_, &next_iteration));
    next_iterations_nodes_.emplace_back(next_iteration);
  }
  return OkStatus();
}

Status LowerWhileHelper::UpdateMergeNodes() {
  for (int i = 0; i < merge_nodes_.size(); i++) {
    TF_RETURN_IF_ERROR(
        graph_->UpdateEdge(next_iterations_nodes_[i], 0, merge_nodes_[i], 1));
  }
  return OkStatus();
}

Status LowerWhileHelper::UpdateConsumers() {
  for (const Edge* e : while_op_->out_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(lowered_while_executed_, e->dst());
    } else {
      if (IsResource(e->src_output())) {
        OutputTensor resource;
        TF_RETURN_IF_ERROR(
            enter_nodes_[e->src_output()]->input_tensor(0, &resource));
        graph_->AddEdge(resource.node, resource.index, e->dst(),
                        e->dst_input());
      } else {
        // Feed the outputs directly from the exit nodes so that downstream ops
        // can start before all the outputs have been computed.
        int exit_node_index = op_input_output_to_lowered_node_[e->src_output()];
        if (exit_node_index < 0) {
          return errors::Internal(
              "Expecting an Exit node for a Resource tensor.");
        }
        graph_->AddEdge(exit_nodes_[exit_node_index], 0, e->dst(),
                        e->dst_input());
      }
    }
  }
  return OkStatus();
}

string LowerWhileHelper::NewName(const string& infix) {
  return graph_->NewName(strings::StrCat(name_, "/", infix));
}

bool LowerWhileHelper::IsResource(int index) {
  return while_op_->input_type(index) == DT_RESOURCE;
}

}  // namespace

Status RewriteWhileNode(Node* n, Graph* g,
                        const FunctionLibraryDefinition* flib_def,
                        bool keep_node_fetchable) {
  VLOG(2) << "Lower While node (keep_node_fetchable=" << keep_node_fetchable
          << "): " << SummarizeNode(*n);

  const AttrValue* cond_attr = n->attrs().Find("cond");
  if (cond_attr == nullptr) {
    return errors::InvalidArgument("While cond function missing");
  }
  const AttrValue* body_attr = n->attrs().Find("body");
  if (body_attr == nullptr) {
    return errors::InvalidArgument("While body function missing");
  }
  const AttrValue* parallel_iterations_attr =
      n->attrs().Find("parallel_iterations");
  if (parallel_iterations_attr == nullptr) {
    return errors::InvalidArgument("parallel_iterations attr missing");
  }
  if (parallel_iterations_attr->i() < 1) {
    return errors::InvalidArgument("parallel_iterations must be > 0");
  }

  TF_RETURN_IF_ERROR(LowerWhileHelper::Run(
      n, cond_attr->func(), body_attr->func(), parallel_iterations_attr->i(), g,
      flib_def, keep_node_fetchable));
  g->RemoveNode(n);

  return OkStatus();
}

}  // namespace tensorflow
