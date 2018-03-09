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
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveConstantShapeOrRank::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();
  if (!(op->type == OperatorType::kTensorFlowShape ||
        op->type == OperatorType::kRank)) {
    return false;
  }

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been resolved
    return false;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until the input array's shape has been resolved.
    return false;
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been resolved.
    return false;
  }

  // Compute the output
  CHECK(!output_array.buffer);
  auto& output_buffer = output_array.GetMutableBuffer<ArrayDataType::kInt32>();
  if (op->type == OperatorType::kTensorFlowShape) {
    // Copy the input shape into the output buffer.
    output_buffer.data = input_array.shape().dims();
  } else if (op->type == OperatorType::kRank) {
    // Copy the dimension count into the output buffer.
    output_buffer.data.resize(1);
    output_buffer.data[0] = input_array.shape().dimensions_count();
  }
  output_array.mutable_shape()->ReplaceDims(
      {static_cast<int>(output_buffer.data.size())});

  // Delete the input array if no longer used
  if (IsDiscardableArray(*model, op->inputs[0]) &&
      CountOpsWithInput(*model, op->inputs[0]) == 1) {
    model->EraseArray(op->inputs[0]);
  }

  model->operators.erase(it);
  return true;
}

}  // namespace toco
