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

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

namespace tensorflow {
namespace ops {
namespace {

// We duplicate this function for now since AccumulateNV2 is currently 
// defined under contrib and AddNGrad is private to the "real" math_grad.cc
Status AddNGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  // AddN doesn't support broadcasting, so all the inputs must be the
  // same shape.
  // Note:
  // dy/dx_k = d(x_1 + x_2 + ... + x_n)/dx_k = 1 for all x_k
  // hence dx_k = dy for all x_k
  // So the gradient for AddN just transfers the incoming gradient to
  // all outgoing gradients.
  auto incoming = Identity(scope, grad_inputs[0]);
  for (int32 i = 0; i < op.num_inputs(); ++i) {
    grad_outputs->push_back(incoming);
  }
  return scope.status();
}
//REGISTER_GRADIENT_OP("AddN", AddNGrad);

// AccumulateNV2 is equivalent to AddN for gradient computation purposes
REGISTER_GRADIENT_OP("AccumulateNV2", AddNGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
