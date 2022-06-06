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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/toco_types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool IsTrivialMinMax(GraphTransformation* transformation, const Model& model,
                     OperatorType op_type, const std::string& input_array_name,
                     const std::string& clamp_value_array_name) {
  const auto& clamp_value_array = model.GetArray(clamp_value_array_name);
  if (!IsConstantParameterArray(model, clamp_value_array_name)) {
    transformation->AddMessageF("Clip value array %s is non-constant",
                                clamp_value_array_name);
    return false;
  }
  const auto& clamp_value_buffer =
      clamp_value_array.GetBuffer<ArrayDataType::kFloat>();
  CHECK_EQ(clamp_value_buffer.Length(), 1);
  float clamp_value = clamp_value_buffer.data[0];

  double clamp_min;
  double clamp_max;
  switch (op_type) {
    case OperatorType::kMinimum:  //  Element-wise Minimum
      clamp_min = -std::numeric_limits<double>::infinity();
      clamp_max = clamp_value;
      break;
    case OperatorType::kMaximum:  //  Element-wise Maximum
      clamp_min = clamp_value;
      clamp_max = std::numeric_limits<double>::infinity();
      break;
    default:
      CHECK(false);
      return false;
  }

  const auto& input_array = model.GetArray(input_array_name);
  return IsArrayQuantizedRangeSubset(transformation, input_array, clamp_min,
                                     clamp_max);
}

}  // namespace

// Attempts to remove min/max functions if the quantization params indicate that
// the representable values fall inside the clip range.
::tensorflow::Status RemoveTrivialQuantizedMinMax::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if ((op->type != OperatorType::kMinimum &&
       op->type != OperatorType::kMaximum) ||
      op->inputs.size() != 2) {
    return ::tensorflow::OkStatus();
  }
  if (IsTrivialMinMax(this, *model, op->type, op->inputs[0], op->inputs[1])) {
    AddMessageF(
        "Removing trivial min/max %s because the quantization parameters imply "
        "at least as tight a clamp anyway.",
        LogName(*op));
    *modified = RemoveTrivialPassthroughOp(this, model, op_index);
    return ::tensorflow::OkStatus();
  }
  return ::tensorflow::OkStatus();
}

}  // namespace toco
