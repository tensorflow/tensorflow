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

#ifndef TENSORFLOW_CC_OPS_WHILE_LOOP_H_
#define TENSORFLOW_CC_OPS_WHILE_LOOP_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {
namespace ops {

// Function that takes cond graph inputs and returns cond graph boolean output.
// 'output' need not be set if an error is returned.
typedef std::function<Status(const Scope&, const std::vector<Output>& inputs,
                             Output* output)>
    CondGraphBuilderFn;

// Function that takes body graph inputs and returns body graph outputs.
// 'outputs' need not be populated if an error is returned.
typedef std::function<Status(const Scope&, const std::vector<Output>& inputs,
                             std::vector<Output>* outputs)>
    BodyGraphBuilderFn;

// Constructs a while loop.
//
// Arguments:
// * scope: used to construct the while loop.
// * inputs: the initial values of the loop variables. Must be non-empty.
// * cond: a function that builds the condition graph of the loop. Takes the
//     current loop variables as inputs and returns a scalar boolean Output
//     indicating whether the loop should continue.
// * body: a function that builds the body graph of the loop. Takes the current
//     loop variables as inputs and returns the updated loop variables.
// * frame_name: the frame name to use for this while loop. This should be a
//     unique name. This will be used as a prefix for created operations.
// * outputs: output param that returns final loop variable outputs in non-error
//     case. Must be non-null and empty.
// * create_while_ctx: if true, a WhileContext is created and populated for this
//     loop. See core/graph/while_context.h for more details on
//     WhileContexts. This is set to false for loops used as part of gradient
//     computations, since they're part of the gradient for a loop in the
//     forward-pass.
//     TODO(skyewm): revisit this. Should we create WhileContexts for all loops,
//     even if we don't need them?
// * cond_output: if non-null, the output of the predicate is returned. This
//     will always be a LoopCond node.
//
// Returns an error if the while loop could not be fully constructed.
//
// TODO(skyewm): clean up partially-constructed loop in error case
// TODO(skyewm): create public interface to this method
Status BuildWhileLoop(const Scope& scope, const std::vector<Output>& inputs,
                      const CondGraphBuilderFn& cond,
                      const BodyGraphBuilderFn& body, const string& frame_name,
                      OutputList* outputs, bool create_while_ctx = true,
                      Output* cond_output = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_WHILE_LOOP_H_
