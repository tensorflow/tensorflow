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

#include "tensorflow/cc/ops/while_loop.h"

#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

namespace {

// Utility function for converting to internal C++ datatypes.
OutputTensor ToOutputTensor(const Output& output) {
  return OutputTensor(output.node(), output.index());
}

// Utility function for converting to internal C++ datatypes.
std::vector<OutputTensor> ToOutputTensors(const std::vector<Output>& outputs) {
  std::vector<OutputTensor> result(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    result[i] = ToOutputTensor(outputs[i]);
  }
  return result;
}

// Utility function for converting to internal C++ datatypes.
std::vector<Node*> ToNodes(const std::vector<Output>& outputs) {
  std::vector<Node*> result(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    result[i] = outputs[i].node();
  }
  return result;
}

// Manually generates the name of the `loop_var_idx`-th NextIteration node of a
// loop being constructed with `scope`. This is used to define the backedge
// before the NextIteration node is created.
string NextIterationName(const Scope& scope, int loop_var_idx) {
  string result;
  const string& prefix = scope.impl()->name();
  if (!prefix.empty()) strings::StrAppend(&result, prefix, "/");
  strings::StrAppend(&result, "NextIteration");
  if (loop_var_idx > 0) strings::StrAppend(&result, "_", loop_var_idx);
  return result;
}

// Creates the `loop_var_idx`-th Merge node of a loop being constructed with
// `scope`. `enter_output` is the `loop_var_idx`-th Enter node's output.
Status CreateMerge(const Scope& scope, int loop_var_idx,
                   const Output& enter_output, Output* merge_output) {
  // The merge nodes accept the while loop's back edges as an input (i.e. the
  // not-yet-created next iteration nodes). Use the underlying NodeBuilder API
  // directly to create the back edge.
  NodeBuilder::NodeOut enter_input(enter_output.node(), enter_output.index());

  const int next_output_index = 0;
  DataType dtype = enter_output.node()->output_type(0);
  NodeBuilder::NodeOut next_input(NextIterationName(scope, loop_var_idx),
                                  next_output_index, dtype);

  std::vector<NodeBuilder::NodeOut> input_list({enter_input, next_input});
  const string unique_name = scope.GetUniqueNameForOp("Merge");
  NodeBuilder builder = NodeBuilder(unique_name, "Merge").Input(input_list);
  scope.UpdateBuilder(&builder);

  Node* merge_node;
  TF_RETURN_IF_ERROR(builder.Finalize(scope.graph(), &merge_node));
  TF_RETURN_IF_ERROR(scope.DoShapeInference(merge_node));
  *merge_output = Output(merge_node, 0);
  return OkStatus();
}

// Creates the condition subgraph defined by `cond`.
Status CreateCond(const Scope& scope, const CondGraphBuilderFn& cond,
                  const std::vector<Output>& inputs, Output* output) {
  // The control dependency is for constants in the cond graph, and other ops
  // that do not depend on the loop variables. This ensures that these ops are
  // in the while loop frame (since they will indirectly depend on an Enter node
  // defining the frame) and that they are executed once per loop iteration.
  //
  // TODO(skyewm): the control dep will be added to all nodes in the cond graph.
  // This is at best unnecessary, and at worst may prevent different parts of
  // different loop iterations from executing in parallel.
  Scope cond_scope =
      scope.NewSubScope("cond").WithControlDependencies(inputs[0]);
  Output raw_cond_out;
  TF_RETURN_IF_ERROR(cond(cond_scope, inputs, &raw_cond_out));

  TF_RETURN_IF_ERROR(scope.graph()->IsValidOutputTensor(raw_cond_out.node(),
                                                        raw_cond_out.index()));
  if (raw_cond_out.type() != DT_BOOL) {
    return errors::InvalidArgument(
        "BuildWhileLoop: 'cond' argument must return a boolean output, got ",
        DataTypeString(raw_cond_out.type()));
  }
  // TODO(skyewm): check that raw_cond_out is scalar

  *output = LoopCond(scope, raw_cond_out).output;
  return OkStatus();
}

// Create the body subgraph defined by `body`. `outputs` must be non-null and
// empty.
Status CreateBody(const Scope& scope, const BodyGraphBuilderFn& body,
                  const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
  DCHECK(outputs != nullptr);
  DCHECK(outputs->empty());

  // The control dependency is analogous to that in CreateCond().
  Scope body_scope =
      scope.NewSubScope("body").WithControlDependencies(inputs[0]);
  TF_RETURN_IF_ERROR(body(body_scope, inputs, outputs));

  const size_t num_loop_vars = inputs.size();
  if (outputs->size() != num_loop_vars) {
    return errors::InvalidArgument(
        "BuildWhileLoop: 'body' argument expected to return ", num_loop_vars,
        " output(s), got ", outputs->size());
  }
  for (const Output& output : *outputs) {
    TF_RETURN_IF_ERROR(
        scope.graph()->IsValidOutputTensor(output.node(), output.index()));
    // TODO(skyewm): check output types/shapes
  }
  return OkStatus();
}

}  // namespace

// A while loop with a single loop variable looks like this:
//
// (output)
//     ^    +---------------+
//     |    | body subgraph +-------------+
//    Exit  +---------------+             |
//      ^    ^                            |
//      |    |                            |
//      Switch<--------+                  v
//        ^            |             NextIteration
//        |     +------+--------+         |
//        +---->| cond subgraph |         |
//        |     +---------------+         |
//       Merge<---------------------------+
//       ^
//       |
//    Enter
//      ^
//      |
//   (input)
//
// If there are multiple loop variables, each of the control flow ops is
// duplicated for each loop variable.
// TODO(skyewm): link to public version of design doc
Status BuildWhileLoop(const Scope& scope, const std::vector<Output>& inputs,
                      const CondGraphBuilderFn& cond,
                      const BodyGraphBuilderFn& body, const string& frame_name,
                      OutputList* outputs, bool create_while_ctx,
                      Output* cond_output) {
  DCHECK(!inputs.empty());
  DCHECK(outputs != nullptr);
  DCHECK(outputs->empty());

  TF_RETURN_IF_ERROR(scope.status());
  const size_t num_loop_vars = inputs.size();

  std::vector<Output> enter_outputs(num_loop_vars);
  for (size_t i = 0; i < num_loop_vars; ++i) {
    enter_outputs[i] = internal::Enter(scope, inputs[i], frame_name);
  }
  TF_RETURN_IF_ERROR(scope.status());

  std::vector<Output> merge_outputs(num_loop_vars);
  for (size_t i = 0; i < num_loop_vars; ++i) {
    TF_RETURN_IF_ERROR(
        CreateMerge(scope, i, enter_outputs[i], &merge_outputs[i]));
  }

  Output cond_out;
  TF_RETURN_IF_ERROR(CreateCond(scope, cond, merge_outputs, &cond_out));
  if (cond_output != nullptr) *cond_output = cond_out;

  std::vector<Output> switch_trues(num_loop_vars);
  std::vector<Output> switch_falses(num_loop_vars);
  for (size_t i = 0; i < num_loop_vars; ++i) {
    auto switch_i = Switch(scope, merge_outputs[i], cond_out);
    switch_trues[i] = switch_i.output_true;
    switch_falses[i] = switch_i.output_false;
  }
  TF_RETURN_IF_ERROR(scope.status());

  std::vector<Output> body_outputs;
  TF_RETURN_IF_ERROR(CreateBody(scope, body, switch_trues, &body_outputs));

  std::vector<Output> next_outputs(num_loop_vars);
  for (size_t i = 0; i < num_loop_vars; ++i) {
    next_outputs[i] = NextIteration(scope, body_outputs[i]);
    DCHECK_EQ(next_outputs[i].node()->name(), NextIterationName(scope, i));
  }
  TF_RETURN_IF_ERROR(scope.status());

  // Create the backedges from the NextIteration nodes to the Merge nodes.
  for (size_t i = 0; i < num_loop_vars; ++i) {
    const int merge_backedge_output_index = 1;
    scope.graph()->AddEdge(next_outputs[i].node(), next_outputs[i].index(),
                           merge_outputs[i].node(),
                           merge_backedge_output_index);
  }

  outputs->resize(num_loop_vars);
  for (size_t i = 0; i < num_loop_vars; ++i) {
    (*outputs)[i] = internal::Exit(scope, switch_falses[i]);
  }
  TF_RETURN_IF_ERROR(scope.status());

  if (create_while_ctx) {
    WhileContext* while_ctx;
    TF_RETURN_IF_ERROR(scope.graph()->AddWhileContext(
        frame_name, ToNodes(enter_outputs), ToNodes(*outputs),
        ToOutputTensor(cond_out), ToOutputTensors(switch_trues),
        ToOutputTensors(body_outputs), &while_ctx));

    // Set while_ctx for all exit nodes. We currently don't require knowing the
    // while_ctx for any other nodes.
    for (size_t i = 0; i < num_loop_vars; ++i) {
      (*outputs)[i].node()->set_while_ctx(while_ctx);
    }
  }
  return OkStatus();
}

}  // namespace ops
}  // namespace tensorflow
