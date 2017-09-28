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

#include "tensorflow/cc/framework/while_gradients.h"

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"

namespace tensorflow {
namespace {

using ops::BodyGraphBuilderFn;
using ops::BuildWhileLoop;
using ops::CondGraphBuilderFn;

Output ToOutput(OutputTensor output_tensor) {
  return Output(const_cast<Node*>(output_tensor.node), output_tensor.index);
}

std::vector<Output> ToOutputVector(
    const std::vector<OutputTensor>& output_tensors) {
  size_t n = output_tensors.size();
  std::vector<Output> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) result.push_back(ToOutput(output_tensors[i]));
  return result;
}

// The backprop loop counter and main backprop loop run in their own execution
// frame (conceptually, the main forward loop and forward loop counter run
// together in a frame, then the backprop loop counter and backprop loop run
// together in a different frame). This returns the frame name to use for the
// backprop while loops.
// TODO(skyewm): make sure this is unique among existing frame names
string BackPropFrameName(const string& forward_frame_name) {
  return strings::StrCat(forward_frame_name, "_backprop");
}

// Creates a loop that counts the number of iterations performed by the
// while loop associated with `while_ctx`. The returned output yields the
// iteration count.
Status AddForwardLoopCounter(WhileContext* while_ctx, const Scope& scope,
                             Output* count) {
  // Create while loop:
  //   i = 0
  //   while forward loop predicate is true:
  //     ++i

  Output zero = ops::Const(scope, 0, {});

  // Condition function that returns condition output from original while loop.
  CondGraphBuilderFn cond_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           Output* output) {
    *output = ToOutput(while_ctx->cond_output());
    return Status::OK();
  };

  // Body function that adds one to input.
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Add(scope, inputs[0], 1));
    return scope.status();
  };

  // Note that this loop runs in the same execution frame as the forward loop.
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, {zero}, cond_fn, body_fn,
                                    while_ctx->frame_name(), &outputs,
                                    /* create_while_ctx */ false));
  *count = outputs[0];
  return Status::OK();
}

// Creates a loop that executes `loop_count` times. The returned output is the
// boolean predicate indicating if the loop is still executing. This is used to
// drive the gradient computation for the while loop associated with
// `while_ctx`.
Status AddBackPropLoopCounter(WhileContext* while_ctx, const Output& loop_count,
                              const Scope& scope,
                              Output* backprop_execution_pred) {
  // Create while loop:
  //   n = loop_count
  //   while n > 0:
  //     --n

  // Condition function that returns input > 0.
  CondGraphBuilderFn cond_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  Output* output) {
    DCHECK_EQ(inputs.size(), 1);
    *output = ops::Greater(scope, inputs[0], 0);
    return scope.status();
  };

  // Body function that subtracts one from input.
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Subtract(scope, inputs[0], 1));
    return scope.status();
  };

  string frame_name = BackPropFrameName(while_ctx->frame_name());
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(
      scope, {loop_count}, cond_fn, body_fn, frame_name, &outputs,
      /* create_while_ctx */ false, backprop_execution_pred));
  return Status::OK();
}

// Creates the main backprop loop that computes the gradient of the loop
// associated with `while_ctx`. `grad_inputs` are the partial derivatives
// w.r.t. the loop outputs, i.e. the exit nodes. `backprop_execution_pred` is
// the predicate to use for the backprop loop (see AddBackPropLoopCounter()).
// The partial derivatives w.r.t. the loop inputs, i.e. the input loop vars, are
// returned in `grad_outputs`.
Status AddWhileGradientLoop(WhileContext* while_ctx,
                            const std::vector<Output>& grad_inputs,
                            const Output& backprop_execution_pred,
                            const Scope& parent_scope,
                            std::vector<Output>* grad_outputs) {
  DCHECK_EQ(grad_inputs.size(), while_ctx->body_outputs().size());
  DCHECK_EQ(while_ctx->body_inputs().size(), while_ctx->body_outputs().size());

  Scope scope = parent_scope.NewSubScope("while");

  // Create while loop:
  //   while backprop_execution_pred:
  //     forward loop body gradient

  // Condition function that returns 'backprop_execution_pred'.
  CondGraphBuilderFn cond_fn = [backprop_execution_pred](
                                   const Scope& scope,
                                   const std::vector<Output>& inputs,
                                   Output* output) {
    *output = backprop_execution_pred;
    return Status::OK();
  };

  // Body function that builds while body gradient subgraph.
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
    std::vector<Output> body_outputs =
        ToOutputVector(while_ctx->body_outputs());
    std::vector<Output> body_inputs = ToOutputVector(while_ctx->body_inputs());
    return AddSymbolicGradients(scope, body_outputs, body_inputs, inputs,
                                outputs);
  };

  string frame_name = BackPropFrameName(while_ctx->frame_name());
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, grad_inputs, cond_fn, body_fn,
                                    frame_name, grad_outputs,
                                    /* create_while_ctx */ false));
  return Status::OK();
}

}  // namespace

Status AddWhileLoopGradient(WhileContext* while_ctx, const Scope& scope,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  Output forward_loop_count;
  TF_RETURN_IF_ERROR(AddForwardLoopCounter(
      while_ctx, scope.NewSubScope("ForwardLoopCounter"), &forward_loop_count));

  // TODO(skyewm): can we combine the backprop loop counter and main gradient
  // loop into a single loop? The original Python code doesn't combine the
  // loops, but I'm not sure why.
  Output backprop_counter_cond;
  TF_RETURN_IF_ERROR(AddBackPropLoopCounter(
      while_ctx, forward_loop_count, scope.NewSubScope("BackPropLoopCounter"),
      &backprop_counter_cond));

  return AddWhileGradientLoop(while_ctx, grad_inputs, backprop_counter_cond,
                              scope, grad_outputs);
}

}  // namespace tensorflow
