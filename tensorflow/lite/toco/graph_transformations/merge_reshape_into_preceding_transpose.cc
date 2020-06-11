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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool OperatorReady(const Model& model, const Operator* op) {
  if (!model.HasArray(op->inputs[0]) || !model.HasArray(op->inputs[1]) ||
      !model.HasArray(op->outputs[0])) {
    // Arrays are missing.
    return false;
  }

  if (!model.GetArray(op->inputs[0]).has_shape() ||
      !model.GetArray(op->outputs[0]).has_shape()) {
    // Input and output needs the shape.
    return false;
  }

  if (!model.GetArray(op->inputs[1]).buffer) {
    // Buffer needs to be a constant.
    return false;
  }

  return true;
}

// Returns whether the reshape could be a transpose.
std::vector<int32> ReshapeToTranspose(const Model& model,
                                      const TensorFlowReshapeOperator* op) {
  CHECK(!op->shape.empty());
  CHECK(model.HasArray(op->inputs[0]));
  CHECK(model.HasArray(op->outputs[0]));

  const auto& input_array = model.GetArray(op->inputs[0]);
  const auto& output_array = model.GetArray(op->outputs[0]);

  CHECK(input_array.has_shape());
  CHECK(output_array.has_shape());

  std::vector<int> in_shape = input_array.shape().dims();
  std::vector<int> out_shape = output_array.shape().dims();

  std::vector<int> one_indices;
  std::vector<int> not_one_indices;

  // Separate into one indices and not one indices.
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (in_shape[i] == 1) {
      one_indices.push_back(i);
    } else {
      not_one_indices.push_back(i);
    }
  }

  // Reorder the vertices.
  std::vector<int> perm;
  perm.reserve(in_shape.size());
  int one_index = 0;
  int not_one_index = 0;
  for (const auto val : out_shape) {
    if (val == 1) {
      perm.push_back(one_indices[one_index]);
      one_index++;
    } else {
      perm.push_back(not_one_indices[not_one_index]);
      not_one_index++;
    }
  }

  return perm;
}

}  // namespace

// When a transpose is fed into a reshape, it is possible for the two operators
// to be merged if the reshape does not affect memory ordering and does not
// affects the number of dimensions. This only occurs when only unary dimensions
// are shifting position.
::tensorflow::Status MergeReshapeIntoPrecedingTranspose::Run(
    Model* model, std::size_t op_index, bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* reshape_op = ConvertOperator<TensorFlowReshapeOperator*>(
      it->get(), OperatorType::kReshape);

  if (reshape_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (!OperatorReady(*model, reshape_op) || reshape_op->shape.empty()) {
    return ::tensorflow::Status::OK();
  }

  const string intermediate_name = reshape_op->inputs[0];
  const string output_name = reshape_op->outputs[0];

  // Guarantee the input is only consume by the reshape.
  if (CountOpsWithInput(*model, intermediate_name) != 1) {
    return ::tensorflow::Status::OK();
  }

  // Check for the parent operator.
  const auto& transpose_it = FindOpWithOutput(*model, intermediate_name);
  if (transpose_it == model->operators.end()) {
    return ::tensorflow::Status::OK();
  }

  // Find the parent operator and guarantee it is a transpose.
  TransposeOperator* transpose_op = ConvertOperator<TransposeOperator*>(
      transpose_it->get(), OperatorType::kTranspose);

  if (transpose_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (!OperatorReady(*model, transpose_op) || transpose_op->perm.empty()) {
    return ::tensorflow::Status::OK();
  }

  if (!ReshapeIsEquivalentToTranspose(*model, reshape_op,
                                      false /*allow_extra_unary_dimensions*/)) {
    return ::tensorflow::Status::OK();
  }

  // Check that the intermediate is not an output array.
  if (!IsDiscardableArray(*model, intermediate_name)) {
    AddMessageF(
        "Cannot fuse %s and %s as it would invalidate the transpose "
        "output array.",
        LogName(*transpose_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }

  AddMessageF("Merging operations %s and %s", LogName(*transpose_op),
              LogName(*reshape_op));

  // const auto& intermediate_array = model->GetArray(intermediate_name);
  // const auto& output_array = model->GetArray(output_name);

  auto merged_perm = ReshapeToTranspose(*model, reshape_op);

  // Combine the permutations.
  const auto& transpose_perm = transpose_op->perm;
  for (size_t i = 0; i < merged_perm.size(); i++) {
    merged_perm[i] = transpose_perm[merged_perm[i]];
  }

  // Remove the reshape as passthrough operation.
  if (!RemoveTrivialPassthroughOp(this, model, op_index)) {
    return ::tensorflow::Status::OK();
  }

  // Update transpose_op's constant buffer to contain the new permutation.
  model->GetArray(transpose_op->inputs[1])
      .GetMutableBuffer<ArrayDataType::kInt32>()
      .data = merged_perm;
  transpose_op->perm = merged_perm;

  // transpose_ops's shape will likely has changed.
  model->GetArray(transpose_op->outputs[0]).clear_shape();

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
