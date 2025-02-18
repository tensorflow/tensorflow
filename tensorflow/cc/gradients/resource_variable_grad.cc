/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/status/status.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/array_ops.h"

namespace tensorflow {
namespace ops {
namespace {

absl::Status ReadVariableOpGrad(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("ReadVariableOp", ReadVariableOpGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
