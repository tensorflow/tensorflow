/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool IsElementwiseOperator(OperatorType optype) {
  switch (optype) {
    case OperatorType::kCast:
    case OperatorType::kExp:
    case OperatorType::kFloor:
    case OperatorType::kNeg:
    case OperatorType::kRelu:
    case OperatorType::kRelu1:
    case OperatorType::kRelu6:
    case OperatorType::kTanh:
    case OperatorType::kSqrt:
    case OperatorType::kSquare:
      return true;
    default:
      return false;
  }
}

bool IsMoveOperator(OperatorType optype) {
  switch (optype) {
    case OperatorType::kDepthToSpace:
    case OperatorType::kExpandDims:
    case OperatorType::kSpaceToDepth:
    case OperatorType::kSqueeze:
    case OperatorType::kReshape:
    case OperatorType::kTranspose:
      return true;
    default:
      return false;
  }
}

}  // namespace

// Swap elementwise operators such that all value operators occur before all
// element move operators, e.g. negation then transpose.
::tensorflow::Status ReorderElementwiseUnary::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  const auto element_op_it = model->operators.begin() + op_index;
  std::unique_ptr<Operator>& element_op = *element_op_it;
  if (!IsElementwiseOperator(element_op->type)) {
    return ::tensorflow::Status::OK();
  }

  const string intermediate_name = element_op->inputs[0];
  auto it = FindOpWithOutput(*model, intermediate_name);
  if (it == model->operators.end()) {
    AddMessageF("No preceding operator");
    return ::tensorflow::Status::OK();
  }

  std::unique_ptr<Operator>& move_op = *it;
  if (!IsMoveOperator(move_op->type)) {
    AddMessageF("Preceding operator is not a move operator");
    return ::tensorflow::Status::OK();
  }

  if (CountOpsWithInput(*model, intermediate_name) != 1) {
    AddMessageF("Input %s used elsewhere", intermediate_name);
    return ::tensorflow::Status::OK();
  }

  // Check that the intermediate is discardable.
  if (!IsDiscardableArray(*model, intermediate_name)) {
    AddMessageF(
        "Cannot swap elementwise as it would invalidate %s which is "
        "an output array.",
        intermediate_name);
    return ::tensorflow::Status::OK();
  }

  // op->inputs may change so we need to keep a value by copy.
  const string input_name = move_op->inputs[0];
  const string output_name = element_op->outputs[0];

  AddMessageF("Swapping around operators with %s and %s", LogName(*element_op),
              LogName(*move_op));

  // If the output array is an exit node for the graph then we need to retain
  // the name as an output node. This makes the naming scheme a little confusing
  // but is required in this rare case.
  if (!IsDiscardableArray(*model, output_name)) {
    // The output name of the sequence needs to stay static, so create a new
    // array new use for the intermediate.
    const auto new_intermediate_name =
        AvailableArrayName(*model, element_op->outputs[0] + "_reorder");
    AddMessageF("Adding new array %s to preserve output array name %s",
                new_intermediate_name, output_name);

    element_op->inputs[0] = input_name;
    element_op->outputs[0] = new_intermediate_name;
    model->EraseArray(intermediate_name);
    move_op->inputs[0] = new_intermediate_name;
    move_op->outputs[0] = output_name;
  } else {
    // The intermediate array is now the output array.
    for (int i = 0; i < model->operators.size(); i++) {
      Operator* consumer = model->operators[i].get();
      for (int j = 0; j < consumer->inputs.size(); j++) {
        if (consumer->inputs[j] == output_name) {
          consumer->inputs[j] = intermediate_name;
        }
      }
    }

    element_op->inputs[0] = input_name;
    move_op->inputs[0] = output_name;
  }

  // Reset both arrays as shape, type, min/max, etc can all change because of
  // the position swap.
  model->EraseArray(element_op->outputs[0]);
  model->EraseArray(move_op->outputs[0]);

  // Reconstruct.
  model->GetOrCreateArray(element_op->outputs[0]);
  model->GetOrCreateArray(move_op->outputs[0]);

  // Swap the order of the operators.
  element_op.swap(move_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
