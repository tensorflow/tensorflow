/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/identify_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"

// This transformation rule tries to identify the HardSwish structure generated
// by tensorflow.
// The formula of hardswish is:
// f(x) = x * relu6((x+3))/6
//
// We look for the following tensorflow subgraph:
// x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)
namespace toco {

using util::IsBinaryOp;

::tensorflow::Status IdentifyHardSwish::Run(Model* model, std::size_t op_index,
                                            bool* modified) {
  *modified = false;
  const auto add_with_relu6_op_it = (model->operators.begin() + op_index);
  const auto add_with_relu6_op = add_with_relu6_op_it->get();
  if (!util::IsBinaryOp(add_with_relu6_op, OperatorType::kAdd,
                        FusedActivationFunctionType::kRelu6)) {
    return ::tensorflow::Status::OK();
  }
  std::vector<const Operator*> ops;
  ops.push_back(add_with_relu6_op);
  const auto* mul_op = GetOpWithInput(*model, add_with_relu6_op->outputs[0]);
  ops.push_back(mul_op);

  if (mul_op->type == OperatorType::kFakeQuant) {
    mul_op = GetOpWithInput(*model, mul_op->outputs[0]);
    ops.push_back(mul_op);
  }
  if (!IsBinaryOp(mul_op, OperatorType::kMul)) {
    return ::tensorflow::Status::OK();
  }

  const auto* output_op = GetOpWithInput(*model, mul_op->outputs[0]);
  ops.push_back(output_op);
  if (output_op->type == OperatorType::kFakeQuant) {
    output_op = GetOpWithInput(*model, output_op->outputs[0]);
    ops.push_back(output_op);
  }
  if (!IsBinaryOp(output_op, OperatorType::kMul)) {
    return ::tensorflow::Status::OK();
  }
  const auto add_3_tensor =
      util::GetSingleScalarInputIndexOfBinaryOp(model, add_with_relu6_op, 3.0f);
  if (add_3_tensor < 0) {
    // Expected 3.0f got something else.;
    return ::tensorflow::Status::OK();
  }
  const auto input_tensor_name = add_with_relu6_op->inputs[1 - add_3_tensor];

  // Now we verify that the 3 mul arguments are respectively:
  // 1. non-constant input of add_with_relu6_op
  // 2. 1/6
  // 3. (and add_with_relu6_op[0].outputs[0] - which we already know!)
  std::vector<std::string> mul_inputs = mul_op->inputs;
  mul_inputs.insert(mul_inputs.end(), output_op->inputs.begin(),
                    output_op->inputs.end());

  // 1. Check that we have the input tensor as one of the multiplicants
  if (std::find(mul_inputs.begin(), mul_inputs.end(), input_tensor_name) ==
      mul_inputs.end()) {
    // Input tensor not found! << input_tensor_name << std::endl;
    return ::tensorflow::Status::OK();
  }
  // 2. Find 1/6
  bool found = false;
  for (const auto& input : mul_inputs) {
    found |= util::CheckArrayIsScalarFloat(model, input, 1.f / 6.f);
  }
  if (!found) {
    // Input tensor is not divided by 6!.";
    return ::tensorflow::Status::OK();
  }
  //  Success! Now delete the subgraph and instert new one
  const auto output_tensor_name = output_op->outputs[0];
  auto* hardswish_op = new HardSwishOperator;
  hardswish_op->inputs = {input_tensor_name};
  hardswish_op->outputs = {output_tensor_name};
  model->operators.emplace(add_with_relu6_op_it, hardswish_op);
  AddMessageF("Creating hardswish op (%s) replacing equivalent subgraph",
              LogName(*hardswish_op));
  while (!ops.empty()) {
    DeleteOpAndArrays(model, ops.back());
    ops.pop_back();
  }
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
