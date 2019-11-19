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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/identify_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

using util::GetSingleScalarInputIndexOfBinaryOp;

::tensorflow::Status IdentifyRelu1::Run(Model* model, std::size_t op_index,
                                        bool* modified) {
  *modified = false;
  // Follow sequences of min+max and max+min. First get the leading op.
  const auto op_it = model->operators.begin() + op_index;
  const auto* op_0 = op_it->get();
  if (op_0->type != OperatorType::kMinimum &&
      op_0->type != OperatorType::kMaximum) {
    return ::tensorflow::Status::OK();
  }

  // Get the paired op and ensure it's the counter to the first.
  const auto* op_1 = GetOpWithInput(*model, op_0->outputs[0]);
  if (!op_1 ||
      (op_1->type != OperatorType::kMinimum &&
       op_1->type != OperatorType::kMaximum) ||
      op_0->type == op_1->type) {
    return ::tensorflow::Status::OK();
  }

  const auto* min_op = op_0->type == OperatorType::kMinimum ? op_0 : op_1;
  const auto* max_op = op_0->type == OperatorType::kMaximum ? op_0 : op_1;

  if (min_op->inputs.size() != 2 || max_op->inputs.size() != 2) {
    return ::tensorflow::Status::OK();
  }
  if (min_op->outputs.size() != 1 || max_op->outputs.size() != 1) {
    return ::tensorflow::Status::OK();
  }

  // Get the original input to the min+max pair.
  int min_scalar_input_index =
      GetSingleScalarInputIndexOfBinaryOp(model, min_op, 1.0f);
  int max_scalar_input_index =
      GetSingleScalarInputIndexOfBinaryOp(model, max_op, -1.0f);
  if (min_scalar_input_index == -1 || max_scalar_input_index == -1) {
    return ::tensorflow::Status::OK();
  }
  int op_0_scalar_input_index =
      op_0 == min_op ? min_scalar_input_index : max_scalar_input_index;

  // Create and emplace Relu1 node.
  auto* relu1_op = new Relu1Operator;
  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*relu1_op));
  relu1_op->inputs = {op_0->inputs[!op_0_scalar_input_index]};
  relu1_op->outputs = op_1->outputs;
  model->operators.emplace(op_it, relu1_op);

  DeleteOpAndArrays(model, op_0);
  DeleteOpAndArrays(model, op_1);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
