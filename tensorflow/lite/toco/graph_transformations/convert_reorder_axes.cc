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
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// Creates a Reshape operator from ReorderAxes operator.
TensorFlowReshapeOperator* CreateReshapeFromReorderAxes(
    Model* model, ReorderAxesOperator* reorder_op, const Shape& input_shape) {
  auto* reshape_op = new TensorFlowReshapeOperator;

  // Copy inputs and outputs to Reshape.
  reshape_op->inputs.push_back(reorder_op->inputs[0]);
  reshape_op->outputs = reorder_op->outputs;

  // Create reshape dimensions based on input shape. Conversion from
  // ReorderAxes to Reshape requires a 4D input shape.
  CHECK_EQ(input_shape.dimensions_count(), 4);
  std::vector<int> reshape_dims = {1, input_shape.dims(0), input_shape.dims(1),
                                   input_shape.dims(3) * input_shape.dims(2)};

  // Create a new input array for Reshape.
  string reshape_array_name =
      AvailableArrayName(*model, reshape_op->outputs[0]);
  reshape_op->inputs.push_back(reshape_array_name);

  Array& reshape_array = model->GetOrCreateArray(reshape_array_name);
  *(reshape_array.mutable_shape()->mutable_dims()) = {
      1, static_cast<int>(reshape_dims.size())};
  reshape_array.data_type = ArrayDataType::kInt32;
  auto& reshape_buffer =
      reshape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  reshape_buffer.data = reshape_dims;

  return reshape_op;
}

// Creates a Transpose operator from ReorderAxes operator.
TransposeOperator* CreateTransposeFromReorderAxes(
    Model* model, ReorderAxesOperator* reorder_op, const Shape& input_shape,
    const AxesOrder& input_axes_order, const AxesOrder& output_axes_order) {
  auto* transpose_op = new TransposeOperator;

  // Copy inputs and outputs to Transpose.
  transpose_op->inputs.push_back(reorder_op->inputs[0]);
  transpose_op->outputs = reorder_op->outputs;

  // Create permutations data based on input and output axes order.
  std::vector<int> permutations_data;
  GetShuffleShape(input_axes_order, output_axes_order, &permutations_data);

  // Create a new input permutations array for Transpose.
  string perm_array_name = AvailableArrayName(*model, transpose_op->outputs[0]);
  transpose_op->inputs.push_back(perm_array_name);

  Array& perm_array = model->GetOrCreateArray(perm_array_name);
  *(perm_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(permutations_data.size())};
  perm_array.data_type = ArrayDataType::kInt32;
  auto& perm_buffer = perm_array.GetMutableBuffer<ArrayDataType::kInt32>();
  perm_buffer.data = permutations_data;

  return transpose_op;
}

// Converts ReorderAxes into Transpose and Reshape which are compatible with the
// TFLite interpreter.
::tensorflow::Status ConvertReorderAxes::Run(Model* model, std::size_t op_index,
                                             bool* modified) {
  *modified = false;
  auto reorder_it = model->operators.begin() + op_index;
  if (reorder_it->get()->type != OperatorType::kReorderAxes)
    return ::tensorflow::Status::OK();

  auto* reorder_op = static_cast<ReorderAxesOperator*>(reorder_it->get());
  CHECK_EQ(reorder_op->inputs.size(), 1);
  CHECK_EQ(reorder_op->outputs.size(), 1);

  const auto& input_array_name = reorder_op->inputs[0];
  const auto& output_array_name = reorder_op->outputs[0];
  auto& input_array = model->GetArray(input_array_name);
  auto& output_array = model->GetArray(output_array_name);

  // Get input array. If kFakeQuant is the input into ReorderAxes, get the input
  // array passed into kFakeQuant. kFakeQuant op is dropped when possible.
  string constant_input_array_name = input_array_name;
  if (!input_array.buffer) {
    const auto* op_producing_input = GetOpWithOutput(*model, input_array_name);
    if (op_producing_input &&
        op_producing_input->type == OperatorType::kFakeQuant) {
      constant_input_array_name = op_producing_input->inputs[0];
    }
  }

  // Yield if input array contains constants or if output array size has not
  // been adjusted to reflect the permutations in ReorderAxes. ReorderAxes will
  // be merged into a constant array when possible.
  if (IsConstantParameterArray(*model, constant_input_array_name))
    return ::tensorflow::Status::OK();
  if (!output_array.has_shape()) return ::tensorflow::Status::OK();

  const auto input_axes_order = reorder_op->input_axes_order;
  const auto output_axes_order = reorder_op->output_axes_order;
  const Shape input_shape = input_array.shape();

  // Creates a Reshape or Transpose operator depending on the conversion.
  if (input_axes_order == AxesOrder::kHWIM &&
      output_axes_order == AxesOrder::k1HWO) {
    // Add Reshape operator into the graph. This special case is not just a
    // permutation. The input dimensions get merged into 3 dimensions while the
    // order of the elements does not change.
    auto* reshape_op =
        CreateReshapeFromReorderAxes(model, reorder_op, input_shape);
    const auto reshape_it = model->operators.emplace(reorder_it, reshape_op);
    reorder_it = reshape_it + 1;
  } else {
    // Add Transpose operator into the graph.
    auto* transpose_op = CreateTransposeFromReorderAxes(
        model, reorder_op, input_shape, input_axes_order, output_axes_order);
    const auto transpose_it =
        model->operators.emplace(reorder_it, transpose_op);
    reorder_it = transpose_it + 1;
  }

  // Remove ReorderAxes operator from the graph.
  CHECK_EQ(reorder_it->get(), reorder_op);
  model->operators.erase(reorder_it);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
