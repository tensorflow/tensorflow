/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"

namespace tensorflow {
namespace ops {
namespace {

Status SoftmaxGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  // Softmax gradient function.
  // p = softmax(x) maps from [batch, n] to [batch, m]
  // dp/dx = [dp0/dx0   ... dp0/dxn-1  ]
  //         [  ...           ...      ]
  //         [dpm-1/dx0 ... dpm-1/dxn-1]
  // dL/dx = dp/dx * dL/dy
  //
  // Using alternative formula:
  // dL/dx = dL/dy * y - sum(dL/dy * y) * y
  //    = (dL/dy - sum(dL/dy * y)) * y
  auto y = op.output(0);
  auto dyy = Mul(scope, grad_inputs[0], y);
  auto sum = Reshape(scope, Sum(scope, dyy, {1}), {-1, 1});
  auto sub = Sub(scope, grad_inputs[0], sum);
  auto dx = Mul(scope, sub, y);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softmax", SoftmaxGrad);

Status ReluGradHelper(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto dx = ReluGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu", ReluGradHelper);

Status Relu6GradHelper(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  auto dx = Relu6Grad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu6", Relu6GradHelper);

Status EluGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto dx = EluGrad(scope, grad_inputs[0], op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Elu", EluGradHelper);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
