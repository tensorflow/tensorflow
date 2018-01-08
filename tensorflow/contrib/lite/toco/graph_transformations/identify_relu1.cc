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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator* op) {
  auto it = model->operators.begin();
  for (; it != model->operators.end(); ++it) {
    if (it->get() == op) {
      break;
    }
  }
  return it;
}

bool CheckArrayIsScalarFloat(Model* model, const std::string& name, float val) {
  const auto& op_array = model->GetArray(name);
  if (!op_array.buffer || op_array.buffer->type != ArrayDataType::kFloat ||
      RequiredBufferSizeForShape(op_array.shape()) != 1) {
    return false;
  }
  const auto& op_data = op_array.GetBuffer<ArrayDataType::kFloat>().data;
  return op_data[0] == val;
}

// Returns index of scalar input when there is exactly one scalar, -1 otherwise
int GetSingleScalarInputIndexOfBinaryOp(Model* model, const Operator* op,
                                        float val) {
  bool input0_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[0], val);
  bool input1_is_scalar = CheckArrayIsScalarFloat(model, op->inputs[1], val);
  return input0_is_scalar == input1_is_scalar ? -1 : input0_is_scalar ? 0 : 1;
}
}  // namespace

bool IdentifyRelu1::Run(Model* model, std::size_t op_index) {
  const auto maximum_it = model->operators.begin() + op_index;
  const auto* maximum_op = maximum_it->get();
  if (maximum_op->type != OperatorType::kTensorFlowMaximum) {
    return false;
  }
  CHECK_EQ(maximum_op->inputs.size(), 2);
  if (maximum_op->outputs.size() != 1) {
    return false;
  }
  int scalar_input_index =
      GetSingleScalarInputIndexOfBinaryOp(model, maximum_op, -1.0f);
  if (scalar_input_index == -1) {
    return false;
  }
  const auto* minimum_op = GetOpWithInput(*model, maximum_op->outputs[0]);
  if (!minimum_op || minimum_op->type != OperatorType::kTensorFlowMinimum) {
    return false;
  }
  if (GetSingleScalarInputIndexOfBinaryOp(model, minimum_op, 1.0f) == -1) {
    return false;
  }
  CHECK_EQ(minimum_op->inputs.size(), 2);

  // Create and emplace Relu1 node
  auto* relu1_op = new Relu1Operator;
  relu1_op->inputs = {maximum_op->inputs[!scalar_input_index]};
  relu1_op->outputs = minimum_op->outputs;
  model->operators.emplace(maximum_it, relu1_op);

  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*relu1_op));

  // Erase Maximum scalar input & operator
  model->arrays.erase(maximum_op->inputs[scalar_input_index]);
  model->operators.erase(FindOperator(model, maximum_op));

  // Erase Minimum inputs & operator
  model->arrays.erase(minimum_op->inputs[0]);
  model->arrays.erase(minimum_op->inputs[1]);
  model->operators.erase(FindOperator(model, minimum_op));

  return true;
}

}  // namespace toco
