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
  // Follow sequences of min+max and max+min. First get the leading op.
  const auto op_it = model->operators.begin() + op_index;
  const auto* op_0 = op_it->get();
  if (op_0->type != OperatorType::kMinimum &&
      op_0->type != OperatorType::kMaximum) {
    return false;
  }

  // Get the paired op and ensure it's the counter to the first.
  const auto* op_1 = GetOpWithInput(*model, op_0->outputs[0]);
  if (!op_1 ||
      (op_1->type != OperatorType::kMinimum &&
       op_1->type != OperatorType::kMaximum) ||
      op_0->type == op_1->type) {
    return false;
  }

  const auto* min_op = op_0->type == OperatorType::kMinimum ? op_0 : op_1;
  const auto* max_op = op_0->type == OperatorType::kMaximum ? op_0 : op_1;

  if (min_op->inputs.size() != 2 || max_op->inputs.size() != 2) {
    return false;
  }
  if (min_op->outputs.size() != 1 || max_op->outputs.size() != 1) {
    return false;
  }

  // Get the original input to the min+max pair.
  int min_scalar_input_index =
      GetSingleScalarInputIndexOfBinaryOp(model, min_op, 1.0f);
  int max_scalar_input_index =
      GetSingleScalarInputIndexOfBinaryOp(model, max_op, -1.0f);
  if (min_scalar_input_index == -1 || max_scalar_input_index == -1) {
    return false;
  }
  int op_0_scalar_input_index =
      op_0 == min_op ? min_scalar_input_index : max_scalar_input_index;

  // Create and emplace Relu1 node.
  auto* relu1_op = new Relu1Operator;
  relu1_op->inputs = {op_0->inputs[!op_0_scalar_input_index]};
  relu1_op->outputs = op_1->outputs;
  model->operators.emplace(op_it, relu1_op);

  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*relu1_op));

  // Erase op scalar inputs & operators. Note that we preserve the non-scalar
  // input to the first op as that's been redirected to the relu1_op.
  DeleteArrayIfUsedOnce(op_0->inputs[op_0_scalar_input_index], model);
  DeleteArrayIfUsedOnce(op_1->inputs[0], model);
  DeleteArrayIfUsedOnce(op_1->inputs[1], model);
  model->operators.erase(FindOperator(model, op_0));
  model->operators.erase(FindOperator(model, op_1));

  return true;
}

}  // namespace toco
