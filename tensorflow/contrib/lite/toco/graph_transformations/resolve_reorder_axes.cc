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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveReorderAxes::Run(Model* model, std::size_t op_index) {
  auto reorder_it = model->operators.begin() + op_index;
  auto* reorder_op = static_cast<ReorderAxesOperator*>(reorder_it->get());
  if (reorder_op->type != OperatorType::kReorderAxes) {
    return false;
  }
  const auto& input_array_name = reorder_op->inputs[0];
  const auto& output_array_name = reorder_op->outputs[0];
  auto& input_array = model->GetArray(input_array_name);
  auto& output_array = model->GetArray(output_array_name);
  string constant_input_array_name = input_array_name;
  if (!input_array.buffer) {
    const auto* op_producing_input = GetOpWithOutput(*model, input_array_name);
    if (op_producing_input &&
        op_producing_input->type == OperatorType::kFakeQuant) {
      constant_input_array_name = op_producing_input->inputs[0];
    }
  }
  auto& constant_input_array = model->GetArray(constant_input_array_name);
  if (!constant_input_array.buffer) {
    return false;
  }
  // Yield until output dims have been resolved.
  if (!output_array.has_shape()) {
    return false;
  }
  // Reorder the input array dims and buffer data
  CHECK(constant_input_array.buffer->type == ArrayDataType::kFloat);
  CHECK(!output_array.buffer);
  auto& input_data =
      constant_input_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  std::vector<float> reordered_data;
  reordered_data.resize(RequiredBufferSizeForShape(output_array.shape()));
  const auto input_axes_order = reorder_op->input_axes_order;
  const auto output_axes_order = reorder_op->output_axes_order;
  // TODO(b/62904716) Shapes should be used directly.
  Shape input_shape = constant_input_array.shape();
  Shape output_shape = output_array.shape();
  if (AxesCount(input_axes_order) == 2) {
    UnextendShape(&input_shape, 2);
    UnextendShape(&output_shape, 2);
  }
  ShuffleArray(input_shape, input_axes_order, output_axes_order, output_shape,
               input_data.data(), reordered_data.data());
  input_data = reordered_data;
  input_array.copy_shape(output_array.shape());
  constant_input_array.copy_shape(output_array.shape());

  // Update the edges of the graph to point to the input array
  for (const auto& other_op : model->operators) {
    for (auto& input : other_op->inputs) {
      if (input == output_array_name) {
        input = input_array_name;
      }
    }
  }

  AddMessageF("Reordered axes for array %s", input_array_name);

  // Remove the op and output array.
  model->arrays.erase(output_array_name);
  model->operators.erase(reorder_it);
  return true;
}

}  // namespace toco
