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
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

absl::Status ResolveTensorFlowConcat::Run(Model* model, std::size_t op_index,
                                          bool* modified) {
  *modified = false;
  auto concat_it = model->operators.begin() + op_index;
  const auto* tf_concat_op = concat_it->get();
  if (tf_concat_op->type != OperatorType::kConcat &&
      tf_concat_op->type != OperatorType::kConcatV2) {
    return absl::OkStatus();
  }

  CHECK_GE(tf_concat_op->inputs.size(), 2);
  // TensorFlow Concat and ConcatV2 nodes only differ by the ordering
  // of inputs: in Concat,the axis is the first input, while in
  // ConcatV2, it is the last input.
  std::size_t axis_pos = 0;
  if (tf_concat_op->type == OperatorType::kConcatV2) {
    axis_pos = tf_concat_op->inputs.size() - 1;
  }
  const std::string axis_name = tf_concat_op->inputs[axis_pos];
  std::vector<std::string> concat_input_names;
  for (std::size_t i = 0; i < tf_concat_op->inputs.size(); i++) {
    if (i != axis_pos) {
      concat_input_names.push_back(tf_concat_op->inputs[i]);
    }
  }
  // If the axis array hasn't been resolved to a constant yet,
  // we need to yield.
  const auto& axis_array = model->GetArray(axis_name);
  if (!axis_array.buffer) {
    AddMessageF("Waiting for the axis of %s to be resolved to a constant",
                LogName(*tf_concat_op));
    return absl::OkStatus();
  }

  CHECK(axis_array.data_type == ArrayDataType::kInt32);
  const auto& axis_data = axis_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(axis_data.size(), 1);
  const int axis = axis_data[0];

  // Create the Concatenation op replacing the TensorFlowConcat op.
  auto* concatenation_op = new ConcatenationOperator;
  concatenation_op->axis = axis;
  concatenation_op->inputs = concat_input_names;
  concatenation_op->outputs = {tf_concat_op->outputs[0]};
  auto depth_concat_it = model->operators.emplace(concat_it, concatenation_op);
  CHECK_EQ(depth_concat_it->get(), concatenation_op);

  DeleteOpAndArrays(model, tf_concat_op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
