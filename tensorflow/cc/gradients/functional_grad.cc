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

#include <iostream>
#include <vector>

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/functional_ops.h"

namespace tensorflow {
namespace ops {
namespace {

Status PartitionedCallGrad(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  NameAttrList f;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "f", &f));
  for (const auto& attr : op.node()->attrs()) {
    (*f.mutable_attr())[attr.first] = attr.second;
  }

  std::vector<Output> func_inputs;
  std::vector<DataType> input_dtypes;
  const int num_inputs = op.num_inputs();
  func_inputs.reserve(num_inputs + grad_inputs.size());
  input_dtypes.reserve(num_inputs);

  for (int i = 0; i < num_inputs; i++) {
    func_inputs.push_back(op.input(i));
    input_dtypes.push_back(op.input_type(i));
  }

  func_inputs.insert(std::end(func_inputs), std::begin(grad_inputs),
                     std::end(grad_inputs));

  auto grad = SymbolicGradient(scope, func_inputs, input_dtypes, f);
  for (int i = 0; i < num_inputs; i++) {
    grad_outputs->push_back(grad[i]);
  }

  return scope.status();
}

REGISTER_GRADIENT_OP("PartitionedCall", PartitionedCallGrad);
REGISTER_GRADIENT_OP("StatefulPartitionedCall", PartitionedCallGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
