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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool TransposeAffectsMemoryOrder(std::vector<int> perm,
                                 std::vector<int> in_shape) {
  CHECK_EQ(perm.size(), in_shape.size());
  // See what the ordering of the non-unary columns are before and after
  // transpose permutation. If the major indices stay in the same order (not
  // just the shape) then the flat buffer representation shouldn't change.
  std::vector<int> old_major_index_ordering;
  std::vector<int> new_major_index_ordering;
  for (int i = 0, end = in_shape.size(); i < end; i++) {
    if (in_shape[i] != 1) {
      old_major_index_ordering.push_back(i);
    }

    if (in_shape[perm[i]] != 1) {
      new_major_index_ordering.push_back(perm[i]);
    }
  }

  CHECK_EQ(new_major_index_ordering.size(), old_major_index_ordering.size());

  return old_major_index_ordering != new_major_index_ordering;
}

}  // namespace

::tensorflow::Status ConvertTrivialTransposeToReshape::Run(Model* model,
                                                           std::size_t op_index,
                                                           bool* modified) {
  *modified = false;
  auto transpose_it = model->operators.begin() + op_index;
  if (transpose_it->get()->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }
  TransposeOperator* transpose_op =
      static_cast<TransposeOperator*>(transpose_it->get());

  const auto& input_array = model->GetArray(transpose_op->inputs[0]);
  const auto& output_array = model->GetArray(transpose_op->outputs[0]);
  if (!input_array.has_shape() || !output_array.has_shape()) {
    // Yield until PropagateFixedSizes has been run on this op.
    return ::tensorflow::Status::OK();
  }
  // Note: We can assume we have error checked inputs in PropagateFixedSizes.

  // Check that the permutation has propagated.
  std::vector<int> const& perm = transpose_op->perm;
  if (perm.empty()) {
    return ::tensorflow::Status::OK();
  }

  // This transpose is trivial if non-unitary dimensions remain in the same
  // order.
  std::vector<int> const& input_dims = input_array.shape().dims();
  std::vector<int> const& output_dims = output_array.shape().dims();

  if (TransposeAffectsMemoryOrder(perm, input_dims)) {
    return ::tensorflow::Status::OK();
  }

  // This transpose is trivial. Replace it with a Reshape op.
  auto* reshape_op = new TensorFlowReshapeOperator;

  // Copy input and output
  reshape_op->inputs.push_back(transpose_op->inputs[0]);
  reshape_op->outputs = transpose_op->outputs;

  // Create a new input array for the shape input
  std::string perm_array_name = transpose_op->inputs[1];
  std::string shape_array_name =
      toco::AvailableArrayName(*model, perm_array_name);
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  *(shape_array.mutable_shape()->mutable_dims()) = {
      1, static_cast<int>(output_dims.size())};
  reshape_op->inputs.push_back(shape_array_name);
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  shape_buffer.data = output_dims;

  // Delete perm array if unused
  if (IsDiscardableArray(*model, perm_array_name) &&
      CountOpsWithInput(*model, perm_array_name) == 1) {
    model->EraseArray(perm_array_name);
  }

  // Replace the operator in the graph.
  model->operators.emplace(transpose_it, reshape_op);
  DeleteOpAndArrays(model, transpose_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
