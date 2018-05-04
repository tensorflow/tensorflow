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
#include <cmath>
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
}  // namespace

bool IdentifyL2Normalization::Run(Model* model, std::size_t op_index) {
  const auto div_it = model->operators.begin() + op_index;
  const auto* div_or_mul_op = div_it->get();
  OperatorType expected_op_type_producing_div_or_mul_input;
  if (div_or_mul_op->type == OperatorType::kDiv) {
    expected_op_type_producing_div_or_mul_input = OperatorType::kTensorFlowSqrt;
  } else if (div_or_mul_op->type == OperatorType::kMul) {
    expected_op_type_producing_div_or_mul_input =
        OperatorType::kTensorFlowRsqrt;
  } else {
    return false;
  }
  CHECK_EQ(div_or_mul_op->inputs.size(), 2);
  Operator* op_producing_div_or_mul_input[2] = {
      GetOpWithOutput(*model, div_or_mul_op->inputs[0]),
      GetOpWithOutput(*model, div_or_mul_op->inputs[1]),
  };
  if (!op_producing_div_or_mul_input[1] ||
      op_producing_div_or_mul_input[1]->type !=
          expected_op_type_producing_div_or_mul_input) {
    return false;
  }
  Operator* sqrt_or_rsqrt_op = op_producing_div_or_mul_input[1];
  CHECK_EQ(sqrt_or_rsqrt_op->inputs.size(), 1);
  Operator* op_producing_sqrt_or_rsqrt_input =
      GetOpWithOutput(*model, sqrt_or_rsqrt_op->inputs[0]);
  if (!op_producing_sqrt_or_rsqrt_input) {
    return false;
  }

  // There may be an Add or a Maximum here, adding or clamping to a "small"
  // constant scalar.
  // Reported bug: b/29395854
  Operator* add_op = nullptr;
  Operator* op_producing_add_input = nullptr;
  if (op_producing_sqrt_or_rsqrt_input->type == OperatorType::kAdd ||
      op_producing_sqrt_or_rsqrt_input->type ==
          OperatorType::kTensorFlowMaximum) {
    add_op = op_producing_sqrt_or_rsqrt_input;
    bool add_can_be_removed = false;
    CHECK_EQ(op_producing_sqrt_or_rsqrt_input->inputs.size(), 2);
    for (int i = 0; i < 2; i++) {
      const auto& input_array =
          model->GetArray(op_producing_sqrt_or_rsqrt_input->inputs[i]);
      if (!input_array.buffer) {
        continue;
      }
      if (input_array.buffer->type != ArrayDataType::kFloat) {
        continue;
      }
      if (RequiredBufferSizeForShape(input_array.shape()) != 1) {
        continue;
      }
      const auto& input_float_data =
          input_array.GetBuffer<ArrayDataType::kFloat>().data;
      if (std::abs(input_float_data[0]) > 1e-3f) {
        continue;
      }
      add_can_be_removed = true;
      op_producing_add_input = GetOpWithOutput(*model, add_op->inputs[1 - i]);
      break;
    }
    if (!add_can_be_removed) {
      AddMessageF(
          "Giving up trying to identify L2Normalization subgraph "
          " because the operator producing the input to the square root, %s,"
          ", does not match the expected pattern",
          LogName(*op_producing_sqrt_or_rsqrt_input));
      return false;
    }
  }

  Operator* sum_op =
      add_op ? op_producing_add_input : op_producing_sqrt_or_rsqrt_input;
  if (sum_op->type != OperatorType::kTensorFlowSum) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: "
        "expected Sum op, got %s",
        LogName(*sum_op));
    return false;
  }

  Operator* square_op = GetOpWithOutput(*model, sum_op->inputs[0]);
  if (square_op->type != OperatorType::kTensorFlowSquare) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: "
        "expected Square op, got %s",
        LogName(*square_op));
    return false;
  }

  CHECK_EQ(square_op->inputs.size(), 1);

  if (square_op->inputs[0] != div_or_mul_op->inputs[0]) {
    AddMessageF(
        "Giving up trying to identify L2Normalization subgraph: %s does not "
        "take the same input as the Mul/Div node",
        LogName(*square_op));
    return false;
  }

  // Create and emplace the new L2Normalization
  auto* l2norm_op = new L2NormalizationOperator;
  l2norm_op->inputs = {div_or_mul_op->inputs[0]};
  l2norm_op->outputs = div_or_mul_op->outputs;
  model->operators.emplace(div_it, l2norm_op);

  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*l2norm_op));

  // Erase the subgraph that is now replaced by L2Normalization
  model->operators.erase(FindOperator(model, square_op));
  model->EraseArray(sum_op->inputs[0]);
  if (sum_op->inputs.size() > 1) {
    model->EraseArray(sum_op->inputs[1]);
  }
  model->operators.erase(FindOperator(model, sum_op));
  if (add_op) {
    model->EraseArray(add_op->inputs[0]);
    model->EraseArray(add_op->inputs[1]);
    model->operators.erase(FindOperator(model, add_op));
  }
  model->EraseArray(sqrt_or_rsqrt_op->inputs[0]);
  model->operators.erase(FindOperator(model, sqrt_or_rsqrt_op));
  model->EraseArray(div_or_mul_op->inputs[1]);
  model->operators.erase(FindOperator(model, div_or_mul_op));
  return true;
}

}  // namespace toco
