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

bool ResolveConstantRange::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  auto* base_op = it->get();
  if (base_op->type != OperatorType::kRange) {
    return false;
  }
  auto* op = static_cast<RangeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 3);
  const auto& start_array = *model->arrays[op->inputs[0]];
  if (!start_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return false;
  }
  const auto& limit_array = *model->arrays[op->inputs[1]];
  if (!limit_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return false;
  }
  const auto& delta_array = *model->arrays[op->inputs[2]];
  if (!delta_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return false;
  }

  for (const auto& input : op->inputs) {
    if (!IsConstantParameterArray(*model, input)) {
      // yield if any input is mutable
      return false;
    }
  }

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = *model->arrays[op->outputs[0]];
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return false;
  }

  CHECK_EQ(RequiredBufferSizeForShape(start_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(limit_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(delta_array.shape()), 1)
      << "Range op inputs must be scalar.";

  CHECK(start_array.data_type == ArrayDataType::kInt32)
      << "Range op inputs must be int32.";
  CHECK(limit_array.data_type == ArrayDataType::kInt32)
      << "Range op inputs must be int32.";
  CHECK(delta_array.data_type == ArrayDataType::kInt32)
      << "Range op inputs must be int32.";

  // Compute buffer contents
  int start = start_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  int limit = limit_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  int delta = delta_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  auto& buffer = output_array.GetMutableBuffer<ArrayDataType::kInt32>();
  buffer.data.clear();
  for (int32 val = start; val < limit; val += delta) {
    buffer.data.push_back(val);
  }
  CHECK_EQ(floor((limit - start) / delta), buffer.data.size());
  CHECK_EQ(buffer.data.size(), output_array.shape().dims()[0]);

  // Delete the input array if no longer used
  if (IsDiscardableArray(*model, op->inputs[0]) &&
      CountOpsWithInput(*model, op->inputs[0]) == 1) {
    model->arrays.erase(op->inputs[0]);
  }
  if (IsDiscardableArray(*model, op->inputs[1]) &&
      CountOpsWithInput(*model, op->inputs[1]) == 1) {
    model->arrays.erase(op->inputs[1]);
  }
  if (IsDiscardableArray(*model, op->inputs[2]) &&
      CountOpsWithInput(*model, op->inputs[2]) == 1) {
    model->arrays.erase(op->inputs[2]);
  }

  // Delete the operator
  model->operators.erase(it);

  return true;
}

}  // namespace toco
