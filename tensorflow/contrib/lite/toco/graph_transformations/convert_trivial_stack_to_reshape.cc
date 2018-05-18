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

#include "absl/strings/str_cat.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ConvertTrivialStackToReshape::Run(Model* model, std::size_t op_index) {
  auto stack_it = model->operators.begin() + op_index;
  if (stack_it->get()->type != OperatorType::kStack) {
    return false;
  }
  auto* stack_op = static_cast<StackOperator*>(stack_it->get());
  if (stack_op->inputs.size() > 1) {
    // Not trivial.
    return false;
  }
  CHECK_EQ(stack_op->outputs.size(), 1);

  const auto& input_array = model->GetArray(stack_op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return false;
  }
  if (input_array.shape().dimensions_count() == 0) {
    // Input array cannot be 0-D.
    // (Unsure if this is TF behavior, but was required to get a test to pass.)
    return false;
  }

  AddMessageF("Converting trivial %s to a reshape", LogName(*stack_op));

  // Note that we could convert to ExpandDims but toco prefers reshapes.
  auto* reshape_op = new TensorFlowReshapeOperator;
  reshape_op->inputs = {stack_op->inputs[0]};
  reshape_op->outputs = stack_op->outputs;

  // Create shape param.
  string shape_array_name =
      AvailableArrayName(*model, stack_op->outputs[0] + "_shape");
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  *(shape_array.mutable_shape()->mutable_dims()) = {
      1 + input_array.shape().dimensions_count()};
  reshape_op->inputs.push_back(shape_array_name);
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  shape_buffer.data.push_back(1);
  for (int dim : input_array.shape().dims()) {
    shape_buffer.data.push_back(dim);
  }

  // Replace the operator in the graph.
  const auto reshape_it = model->operators.emplace(stack_it, reshape_op);
  stack_it = reshape_it + 1;
  CHECK_EQ(stack_it->get(), stack_op);
  model->operators.erase(stack_it);

  return true;
}

}  // namespace toco
