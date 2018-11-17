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
#include "tensorflow/core/common_runtime/lower_if_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

namespace {

using NodeOut = NodeBuilder::NodeOut;

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
//                          loop_var
//                             |
//                           Enter
//                             |
// inlined_cond_func ---<--- Merge -----<----- NextIteration
//      |                      |                    |
//      V                      V                    ^
//      |                      |                    |
//  LoopCond ------>-------- Switch ---->---- inlined_body_func
//                             |
//                           Exit
//                             |
//                          consumer
class LowerWhileHelper {
 public:
  static Status Run(Node* while_op, const string& cond_fn_name,
                    const string& body_fn_name, Graph* graph,
                    const FunctionLibraryDefinition& flib) {
    LowerWhileHelper helper(while_op, cond_fn_name, body_fn_name, graph, flib);
    return helper.RunInternal();
  }

 private:
  // Create a LowerWhileHelper to create the lowering of While op that has cond
  // and body functions named `cond_fn_name` and `body_fn_name` respectively in
  // the given graph.
  LowerWhileHelper(Node* while_op, const string& cond_fn_name,
                   const string& body_fn_name, Graph* graph,
                   const FunctionLibraryDefinition& flib);

  Status RunInternal();

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
  // This gets inlined later in `InlineCallNodes`.
  Status CreateCondFuncCallNode();

  // Creates a Switch node for each loop var and adds to `switch_nodes_`.
  // Output at index 1(true) of a Switch node is fed into the loop body.
  // Output at index 0(false) of a Switch node is fed into the Exit nodes.
  Status CreateSwitchNodes();

  // Creates the call node for body func and stores in `body_call_node_`.
  // This gets inlined later in `InlineCallNodes`.
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
  // edges to depend on `lowered_while_output_` instead.
  Status UpdateConsumers();

  // Inlines the cond and body functions.
  Status InlineCallNodes();

  // Returns unique name containing the name of the While op being rewritten
  // (name_), infix and a suffix to ensure it is unique within the graph.
  string NewName(const string& infix);

  // The original While op.
  Node* while_op_;
  // The call node for the cond branch. This gets inlined.
  Node* cond_call_node_;
  // The LoopCond node specifying the loop termination condition.
  Node* loop_cond_node_;
  // The call node for the body branch. This gets inlined.
  Node* body_call_node_;
  // The IdentityN node with the same outputs as the original While op.
  Node* lowered_while_output_;
  Graph* graph_;
  const FunctionLibraryDefinition& flib_;
  // Name of the `while_op_`.
  string name_;

  NodeBuilder cond_call_builder_;
  NodeBuilder body_call_builder_;

  std::vector<Node*> enter_nodes_;
  std::vector<Node*> merge_nodes_;
  std::vector<Node*> switch_nodes_;
  std::vector<Node*> exit_nodes_;
  std::vector<Node*> next_iterations_nodes_;

  size_t num_loop_inputs_;
};

LowerWhileHelper::LowerWhileHelper(Node* while_op, const string& cond_fn_name,
                                   const string& body_fn_name, Graph* graph,
                                   const FunctionLibraryDefinition& flib)
    : while_op_(while_op),
      graph_(graph),
      flib_(flib),
      name_(while_op->name()),
      cond_call_builder_(NewName("cond"), cond_fn_name, graph->op_registry()),
      body_call_builder_(NewName("body"), body_fn_name, graph->op_registry()),
      num_loop_inputs_(while_op_->num_inputs()) {
  // We intentionally `resize` instead of `reserve` space in `enter_nodes_`
  // because we need to set it's elements out of order in `CreateEnterNodes`.
  enter_nodes_.resize(num_loop_inputs_);
  merge_nodes_.reserve(num_loop_inputs_);
  switch_nodes_.reserve(num_loop_inputs_);
  exit_nodes_.reserve(num_loop_inputs_);
  next_iterations_nodes_.reserve(num_loop_inputs_);
}

Status LowerWhileHelper::RunInternal() {
  TF_RETURN_IF_ERROR(CreateEnterNodes());
  TF_RETURN_IF_ERROR(CreateMergeNodes());
  TF_RETURN_IF_ERROR(CreateCondFuncCallNode());
  TF_RETURN_IF_ERROR(CreateSwitchNodes());
  TF_RETURN_IF_ERROR(CreateBodyFuncCallNode());
  TF_RETURN_IF_ERROR(CreateExitNodes());
  TF_RETURN_IF_ERROR(CreateNextIterationNodes());
  TF_RETURN_IF_ERROR(UpdateMergeNodes());
  TF_RETURN_IF_ERROR(UpdateConsumers());
  TF_RETURN_IF_ERROR(InlineCallNodes());
  return Status::OK();
}

Status LowerWhileHelper::CreateEnterNodes() {
  // Note: `Node::input_edge` runs in  O(num_inputs) so we use
  // `Node::input_edges` instead so that below loop runs in O(num_inputs) time
  // and not O(num_inputs^2).
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(while_op_->input_edges(&edges));
  for (const Edge* edge : edges) {
    Node* enter_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName("enter"), "Enter", graph_->op_registry())
            .Input(NodeOut(edge->src(), edge->src_output()))
            .Attr("frame_name", name_)
            .Finalize(graph_, &enter_node));
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
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName("LoopControlInputs"), "NoOp", graph_->op_registry())
            .ControlInputs(control_inputs)
            .Finalize(graph_, &incoming_control_node));
    for (Node* n : enter_nodes_) {
      graph_->AddControlEdge(incoming_control_node, n);
    }
  }
  return Status::OK();
}

Status LowerWhileHelper::CreateMergeNodes() {
  for (Node* enter_node : enter_nodes_) {
    Node* merge_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName("merge"), "Merge", graph_->op_registry())
            .Input({NodeOut(enter_node, 0), NodeOut(enter_node, 0)})
            .Finalize(graph_, &merge_node));
    merge_nodes_.emplace_back(merge_node);
  }
  return Status::OK();
}

Status LowerWhileHelper::CreateCondFuncCallNode() {
  for (Node* merge_node : merge_nodes_) {
    cond_call_builder_.Input(NodeOut(merge_node, 0));
  }
  TF_RETURN_IF_ERROR(cond_call_builder_.Finalize(graph_, &cond_call_node_));
  // Add a control edge to make sure the Const nodes in the cond function
  // are in the same frame as the rest of the function, otherwise
  // `BuildControlFlowInfo` throws an error.
  graph_->AddControlEdge(merge_nodes_[0], cond_call_node_);
  TF_RETURN_IF_ERROR(
      NodeBuilder(NewName("LoopCond"), "LoopCond", graph_->op_registry())
          .Input(NodeOut(cond_call_node_, 0))
          .Finalize(graph_, &loop_cond_node_));
  return Status::OK();
}

Status LowerWhileHelper::CreateSwitchNodes() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    string op_name;
    {
      const Node* input_node;
      TF_RETURN_IF_ERROR(while_op_->input_node(i, &input_node));
      op_name = strings::StrCat(input_node->name(), "_switch");
    }
    Node* switch_node;
    string op_type = "Switch";
    if (IsRefType(merge_nodes_[i]->output_type(0))) {
      op_type = "RefSwitch";
    }
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName(op_name), op_type, graph_->op_registry())
            .Input(NodeOut(merge_nodes_[i], 0))
            .Input(NodeOut(loop_cond_node_, 0))
            .Finalize(graph_, &switch_node));
    switch_nodes_.emplace_back(switch_node);
  }
  return Status::OK();
}

Status LowerWhileHelper::CreateBodyFuncCallNode() {
  for (Node* switch_node : switch_nodes_) {
    body_call_builder_.Input(NodeOut(switch_node, 1));
  }
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
  TF_RETURN_IF_ERROR(
      NodeBuilder(NewName("loop_body_control"), op_type, graph_->op_registry())
          .Input(NodeOut(switch_nodes_[0], 1))
          .Finalize(graph_, &body_control_node_));
  graph_->AddControlEdge(body_control_node_, body_call_node_);
  return Status::OK();
}

Status LowerWhileHelper::CreateExitNodes() {
  std::vector<NodeOut> outputs;
  outputs.reserve(num_loop_inputs_);
  for (Node* switch_node : switch_nodes_) {
    Node* exit_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(NewName("exit"), "Exit", graph_->op_registry())
            .Input(NodeOut(switch_node, 0))
            .Finalize(graph_, &exit_node));
    exit_nodes_.emplace_back(exit_node);
    outputs.emplace_back(NodeOut(exit_node, 0));
  }

  // Add an IdentityN node that has the same outputs and same name as the
  // original functional While op. This is used for
  // 1. Rewiring the control edges with the original while op as src.
  // 2. Fetching the output of the While node by name in calls to sess.run.
  NodeBuilder ib(name_, "IdentityN");
  ib.Input(outputs);
  TF_RETURN_IF_ERROR(ib.Finalize(graph_, &lowered_while_output_));
  return Status::OK();
}

Status LowerWhileHelper::CreateNextIterationNodes() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    Node* next_iteration;
    TF_RETURN_IF_ERROR(NodeBuilder(NewName("next_iteration"), "NextIteration",
                                   graph_->op_registry())
                           .Input(NodeOut(body_call_node_, i))
                           .Finalize(graph_, &next_iteration));
    next_iterations_nodes_.emplace_back(next_iteration);
  }
  return Status::OK();
}

Status LowerWhileHelper::UpdateMergeNodes() {
  for (int i = 0; i < num_loop_inputs_; i++) {
    TF_RETURN_IF_ERROR(
        graph_->UpdateEdge(next_iterations_nodes_[i], 0, merge_nodes_[i], 1));
  }
  return Status::OK();
}

Status LowerWhileHelper::UpdateConsumers() {
  for (const Edge* e : while_op_->out_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(lowered_while_output_, e->dst());
    } else {
      // Feed the outputs directly from the exit nodes so that downstream ops
      // can start before all the outputs have been computed.
      graph_->AddEdge(exit_nodes_[e->src_output()], 0, e->dst(),
                      e->dst_input());
    }
  }
  return Status::OK();
}

string LowerWhileHelper::NewName(const string& infix) {
  return graph_->NewName(strings::StrCat(name_, "/", infix));
}

Status InlineCallInGraph(Node* n, Graph* g,
                         const FunctionLibraryDefinition& lib) {
  const FunctionDef* fdef = lib.Find(n->type_string());
  CHECK(fdef != nullptr);
  FunctionBody* fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*fdef, n->attrs(), &lib,
                              [&lib](const string& op, const OpDef** sig) {
                                return lib.LookUpOpDef(op, sig);
                              },
                              &fbody));
  // TODO(jpienaar): Improve this interface to make the need to delete it
  // explicit.
  InlineFunctionBody(g->flib_def(), g, n, fbody, false);
  delete fbody;
  return Status::OK();
}

Status LowerWhileHelper::InlineCallNodes() {
  TF_RETURN_IF_ERROR(InlineCallInGraph(cond_call_node_, graph_, flib_));
  TF_RETURN_IF_ERROR(InlineCallInGraph(body_call_node_, graph_, flib_));
  return Status::OK();
}

}  // namespace

Status RewriteWhileNode(Node* n, Graph* g,
                        const FunctionLibraryDefinition& flib) {
  const AttrValue* cond_attr = n->attrs().Find("cond");
  if (cond_attr == nullptr) {
    return errors::InvalidArgument("While cond function missing");
  }
  const AttrValue* body_attr = n->attrs().Find("body");
  if (body_attr == nullptr) {
    return errors::InvalidArgument("While body function missing");
  }

  TF_RETURN_IF_ERROR(LowerWhileHelper::Run(n, cond_attr->func().name(),
                                           body_attr->func().name(), g, flib));
  g->RemoveNode(n);

  return Status::OK();
}

}  // namespace tensorflow
