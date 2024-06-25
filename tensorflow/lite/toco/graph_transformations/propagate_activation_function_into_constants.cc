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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status PropagateActivationFunctionIntoConstants::Run(
    Model* model, std::size_t op_index, bool* modified) {
  *modified = false;
  const auto ac_it = model->operators.begin() + op_index;
  const auto* ac_op = ac_it->get();
  if (ac_op->type != OperatorType::kRelu6 &&
      ac_op->type != OperatorType::kRelu1 &&
      ac_op->type != OperatorType::kRelu) {
    return absl::OkStatus();
  }

  // Find the op producing the array passed to this activation function.
  auto* src_op = GetOpWithOutput(*model, ac_op->inputs[0]);
  if (!src_op) {
    return absl::OkStatus();
  }

  // Ensure the src_op is not used without the activation function applied.
  if (CountTrueOutputs(*model, *src_op) > 1) {
    AddMessageF(
        "Not propagating activation function %s into %s because it has more "
        "than one consumed output",
        LogName(*ac_op), LogName(*src_op));
  }

  // Filter to the list of supported ops.
  std::string src_op_input;
  switch (src_op->type) {
    case OperatorType::kGather:
      src_op_input = src_op->inputs[0];
      break;
    default:
      return absl::OkStatus();
  }
  CHECK_EQ(src_op->outputs[0], ac_op->inputs[0]);

  // Ensure the input is constant as otherwise this needs to happen at runtime.
  // If we bail here, it's still possible that FuseActivationFunctions will fuse
  // the activation if it's supported by the op.
  if (!IsConstantParameterArray(*model, src_op_input)) {
    AddMessageF(
        "Not propagating activation function %s into %s:%s because it is not "
        "constant",
        LogName(*ac_op), LogName(*src_op), src_op_input);
    return absl::OkStatus();
  }

  // Get the array we'll be working with and ensure it's a compatible type.
  auto& const_array = model->GetArray(src_op_input);
  if (const_array.data_type != ArrayDataType::kFloat) {
    AddMessageF(
        "Not propagating activation function %s into %s:%s because it is "
        "non-float data",
        LogName(*ac_op), LogName(*src_op), src_op_input);
    return absl::OkStatus();
  }
  auto& const_array_data =
      const_array.GetMutableBuffer<ArrayDataType::kFloat>().data;

  // Perform the activation function directly into the constant data array.
  for (size_t i = 0; i < const_array_data.size(); ++i) {
    const float value = const_array_data[i];
    float new_value = value;
    switch (ac_op->type) {
      case OperatorType::kRelu: {
        static constexpr float kLower = 0;
        new_value = value < kLower ? kLower : value;
        break;
      }
      case OperatorType::kRelu1: {
        static constexpr float kUpper = 1;
        static constexpr float kLower = -1;
        new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
        break;
      }
      case OperatorType::kRelu6: {
        static constexpr float kUpper = 6;
        static constexpr float kLower = 0;
        new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
        break;
      }
      default:
        LOG(FATAL) << "Unsupported activation function " << LogName(*ac_op);
        return absl::OkStatus();
    }
    const_array_data[i] = new_value;
  }

  AddMessageF("Propagated activation function %s into %s:%s", LogName(*ac_op),
              LogName(*src_op), src_op_input);
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
