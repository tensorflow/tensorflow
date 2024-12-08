/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_FRAMEWORK_GRADIENTS_H_
#define TENSORFLOW_CC_FRAMEWORK_GRADIENTS_H_

#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {

/// NOTE: This API is a work in progress and will likely be changing frequently.
///
/// Given initial gradients 'grad_inputs' (which represent the symbolic partial
/// derivatives of some loss function 'L' w.r.t 'outputs'), adds gradient nodes
/// to the graph associated with 'scope', which compute (and return in
/// 'grad_outputs') the symbolic partial derivatives of 'L' w.r.t 'inputs'.
absl::Status AddSymbolicGradients(const Scope& scope,
                                  const std::vector<Output>& outputs,
                                  const std::vector<Output>& inputs,
                                  const std::vector<Output>& grad_inputs,
                                  std::vector<Output>* grad_outputs);

// Same as above, but uses 'OnesLike' for all shapes in
// 'outputs' as grad_inputs.
absl::Status AddSymbolicGradients(const Scope& scope,
                                  const std::vector<Output>& outputs,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* grad_outputs);

/// Returns a sentinel Output that represents 'no gradient' (i.e. no gradient
/// flows along some graph edge during backpropagation).
/// Can be returned in 'grad_outputs' by an invocation of 'AddSymbolicGradients'
/// (note that gradient flow through an Output can be stopped through the use of
/// the StopGradient node).
Output NoGradient();

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_GRADIENTS_H_
