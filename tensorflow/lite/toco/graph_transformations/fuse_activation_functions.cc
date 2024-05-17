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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status FuseActivationFunctions::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  const auto ac_it = model->operators.begin() + op_index;
  const auto* ac_op = ac_it->get();

  if (ac_op->type != OperatorType::kRelu6 &&
      ac_op->type != OperatorType::kRelu1 &&
      ac_op->type != OperatorType::kRelu) {
    return absl::OkStatus();
  }

  // Find the op producing the array passed to this activation function
  Operator* op = GetOpWithOutput(*model, ac_op->inputs[0]);

  if (!op) return absl::OkStatus();

  if (CountTrueOutputs(*model, *op) > 1) {
    AddMessageF(
        "Not fusing activation function %s into %s because it has more than "
        "one  consumed output",
        LogName(*ac_op), LogName(*op));
    return absl::OkStatus();
  }

  CHECK_EQ(op->outputs[0], ac_op->inputs[0]);

  int count_ops_consuming_output = CountOpsWithInput(*model, ac_op->inputs[0]);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not fusing activation function into %s because it is consumed by more "
        "than 1 other operator",
        LogName(*ac_op), LogName(*op));
    return absl::OkStatus();
  }

  if (!IsDiscardableArray(*model, op->outputs[0])) {
    AddMessageF(
        "Not fusing activation function %s into %s because output %s it is not "
        "discardable",
        LogName(*ac_op), LogName(*op), op->outputs[0]);
    return absl::OkStatus();
  }

  if (op->fused_activation_function != FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not fusing activation function %s into %s because it already has a "
        "fused activation function",
        LogName(*ac_op), LogName(*op));
    return absl::OkStatus();
  }

  if (!OperatorSupportsFusedActivation(op->type)) {
    AddMessageF(
        "Not fusing activation function %s because the %s op doesn't support "
        "it",
        LogName(*ac_op), LogName(*op));
    return absl::OkStatus();
  }

  AddMessageF("Fusing activation function %s into the preceding %s",
              LogName(*ac_op), LogName(*op));
  if (ac_op->type == OperatorType::kRelu6) {
    op->fused_activation_function = FusedActivationFunctionType::kRelu6;
  } else if (ac_op->type == OperatorType::kRelu1) {
    op->fused_activation_function = FusedActivationFunctionType::kRelu1;
  } else if (ac_op->type == OperatorType::kRelu) {
    op->fused_activation_function = FusedActivationFunctionType::kRelu;
  } else {
    LOG(FATAL) << "Unhandled activation function type";
  }
  op->outputs[0] = ac_op->outputs[0];
  DeleteOpAndArrays(model, ac_op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
