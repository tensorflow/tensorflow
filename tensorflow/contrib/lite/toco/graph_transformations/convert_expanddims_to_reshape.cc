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

#include "absl/strings/str_cat.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ConvertExpandDimsToReshape::Run(Model* model, std::size_t op_index) {
  auto expand_it = model->operators.begin() + op_index;
  if (expand_it->get()->type != OperatorType::kExpandDims) {
    return false;
  }
  ExpandDimsOperator* expand_op =
      static_cast<ExpandDimsOperator*>(expand_it->get());
  CHECK_EQ(expand_op->inputs.size(), 2);
  CHECK_EQ(expand_op->outputs.size(), 1);

  const auto& input_array = model->GetArray(expand_op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return false;
  }

  const auto& axis_array = model->GetArray(expand_op->inputs[1]);
  if (!axis_array.has_shape()) {
    // Yield until input axis array shape has been resolved.
    return false;
  }
  CHECK_EQ(RequiredBufferSizeForShape(axis_array.shape()), 1);
  if (!axis_array.buffer) {
    // Yield until the input axis array is constant
    return false;
  }
  int axis = axis_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  std::vector<int> reshape_dims(input_array.shape().dims());
  if (axis < 0) {
    axis = reshape_dims.size();
  }
  reshape_dims.insert(reshape_dims.begin() + axis, 1);

  // The input tensor has shape, and the axis input is constant. We can now
  // replace ExpandDims with a Reshape.
  auto* reshape_op = new TensorFlowReshapeOperator;

  // Copy inputs
  reshape_op->inputs.push_back(expand_op->inputs[0]);
  reshape_op->outputs = expand_op->outputs;

  // Create a new input array
  string axis_array_name = expand_op->inputs[1];
  string shape_array_name = toco::AvailableArrayName(*model, axis_array_name);
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  *(shape_array.mutable_shape()->mutable_dims()) = {
      1, static_cast<int>(reshape_dims.size())};
  reshape_op->inputs.push_back(shape_array_name);
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  shape_buffer.data = reshape_dims;

  // Delete axis array if unused
  if (IsDiscardableArray(*model, axis_array_name) &&
      CountOpsWithInput(*model, axis_array_name) == 1 &&
      !GetOpWithOutput(*model, axis_array_name)) {
    model->EraseArray(axis_array_name);
  }

  // Replace the operator in the graph.
  const auto reshape_it = model->operators.emplace(expand_it, reshape_op);
  expand_it = reshape_it + 1;
  CHECK_EQ(expand_it->get(), expand_op);
  model->operators.erase(expand_it);

  return true;
}

}  // namespace toco
