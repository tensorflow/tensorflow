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
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool UnfuseActivationFunctions::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  auto* op = it->get();

  // If a conv operation has an im2col array, yield: it should be dropped first.
  if ((op->type == OperatorType::kConv) && (op->outputs.size() == 2)) {
    return false;
  }

  Operator* ac_op = nullptr;
  switch (op->fused_activation_function) {
    case FusedActivationFunctionType::kRelu:
      ac_op = new ReluOperator;
      break;
    case FusedActivationFunctionType::kRelu6:
      ac_op = new Relu6Operator;
      break;
    case FusedActivationFunctionType::kRelu1:
      ac_op = new Relu1Operator;
      break;
    default:
      return false;
  }

  // At this point we know that the op has a fused activation function. At the
  // moment that only happens with ops having a single output, may be
  // relaxed in the future.
  CHECK_EQ(op->outputs.size(), 1);

  // Emplace unfused activation function, drop the fused one.
  model->operators.emplace(it + 1, ac_op);
  op->fused_activation_function = FusedActivationFunctionType::kNone;

  // Wire up arrays, constructing a new intermediate array to connect the
  // op to its new unfused activation function.
  ac_op->outputs = op->outputs;
  const string& tmp_array_name =
      AvailableArrayName(*model, op->outputs[0] + "_unfused");
  CHECK(!model->HasArray(tmp_array_name));
  model->GetOrCreateArray(tmp_array_name);
  ac_op->inputs = {tmp_array_name};
  op->outputs = {tmp_array_name};
  return true;
}

}  // namespace toco
