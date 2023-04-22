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

#ifndef TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_
#define TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/while_context.h"

// Utility functions for constructing while loop gradients

namespace tensorflow {

// Adds the gradient computation for the while loop associated with
// `while_ctx`. `grad_inputs` are the partial derivatives w.r.t. the loop
// outputs, i.e. the exit nodes.  The partial derivatives w.r.t. the loop
// inputs, i.e. the input loop vars, are returned in `grad_outputs`.
// `grad_inputs` and `grad_outputs` are both in loop-variable order, as defined
// by the original inputs to BuildWhileLoop().
// TODO(skyewm): maybe comment on NoGradient once it's supported
Status AddWhileLoopGradient(WhileContext* while_ctx, const Scope& scope,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_
