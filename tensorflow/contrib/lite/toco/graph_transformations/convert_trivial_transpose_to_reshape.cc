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
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ConvertTrivialTransposeToReshape::Run(Model* model, std::size_t op_index) {
  auto transpose_it = model->operators.begin() + op_index;
  if (transpose_it->get()->type != OperatorType::kTranspose) {
    return false;
  }
  TransposeOperator* transpose_op =
      static_cast<TransposeOperator*>(transpose_it->get());

  const auto& output_array = *model->arrays[transpose_op->outputs[0]];
  if (!output_array.has_shape()) {
    // Yield until PropagateFixedSizes has been run on this op.
    return false;
  }
  // Note: We can assume we have error checked inputs in PropagateFixedSizes.

  // This transpose is trivial if we only have one non-unitary dimension.
  std::vector<int> const& dims = output_array.shape().dims();
  unsigned non_unitary_axis_count = 0;
  for (int i = 0; i < dims.size(); i++) {
    if (dims[i] != 1) {
      non_unitary_axis_count++;
    }
  }
  if (non_unitary_axis_count > 1) {
    // Transpose is not trivial
    return false;
  }

  // This transpose is trivial. Replace it with a Reshape op.
  auto* reshape_op = new TensorFlowReshapeOperator;

  // Copy input and output
  reshape_op->inputs.push_back(transpose_op->inputs[0]);
  reshape_op->outputs = transpose_op->outputs;

  // Create a new input array for the shape input
  string perm_array_name = transpose_op->inputs[1];
  string shape_array_name = toco::AvailableArrayName(*model, perm_array_name);
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  *(shape_array.mutable_shape()->mutable_dims()) = {
      1, static_cast<int>(dims.size())};
  reshape_op->inputs.push_back(shape_array_name);
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  shape_buffer.data = dims;

  // Delete perm array if unused
  if (IsDiscardableArray(*model, perm_array_name) &&
      CountOpsWithInput(*model, perm_array_name) == 1) {
    model->arrays.erase(perm_array_name);
  }

  // Replace the operator in the graph.
  const auto reshape_it = model->operators.emplace(transpose_it, reshape_op);
  transpose_it = reshape_it + 1;
  CHECK_EQ(transpose_it->get(), transpose_op);
  model->operators.erase(transpose_it);

  return true;
}

}  // namespace toco
