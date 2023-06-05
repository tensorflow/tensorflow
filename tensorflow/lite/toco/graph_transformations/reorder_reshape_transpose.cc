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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool OperatorReady(const Model& model, const Operator* op) {
  if (!model.HasArray(op->inputs[0]) || !model.HasArray(op->inputs[1]) ||
      !model.HasArray(op->outputs[0])) {
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

// Utility function to filter out a value.
void Filter(std::vector<int>* vec, int value) {
  vec->erase(std::remove(vec->begin(), vec->end(), value), vec->end());
}

// Computes a new permutation used to swap a reshape-transpose to a
// transpose-reshape. In this case the permutation operates on the intermediate
// shape.
std::vector<int> ComputeNewPerm(std::vector<int> input_dims,
                                std::vector<int> intermediate_dims,
                                std::vector<int> perm) {
  // These are the major axis of the input.
  std::vector<int> input_indices;
  for (size_t i = 0; i < input_dims.size(); i++) {
    if (input_dims[i] != 1) {
      input_indices.push_back(i);
    }
  }

  // This maps which indices of the input produced the intermediate indices for
  // non-unary dimensions.
  std::unordered_map<int, int> intermediate_to_input_indices_map;
  for (size_t i = 0; i < intermediate_dims.size(); i++) {
    if (intermediate_dims[i] != 1) {
      intermediate_to_input_indices_map[i] =
          input_indices[intermediate_to_input_indices_map.size()];
    }
  }

  // Translate the transpose permutation to a new permutation starting with the
  // major indices.
  std::vector<int> new_perm;
  new_perm.reserve(input_dims.size());
  for (size_t i = 0; i < perm.size(); i++) {
    if (intermediate_dims[perm[i]] == 1) continue;

    new_perm.push_back(intermediate_to_input_indices_map[perm[i]]);
  }

  // Fill the rest of the transpose in with the ones.
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] == 1) {
      new_perm.push_back(index);
    }
  }

  CHECK_EQ(new_perm.size(), input_dims.size());
  return new_perm;
}

}  // namespace

// Swaps reshape-transpose to transpose-reshape whenever possible. This is
// possible when the reshape does not affect memory ordering.
::tensorflow::Status ReorderReshapeTranspose::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  auto transpose_it = model->operators.begin() + op_index;

  TransposeOperator* transpose_op = ConvertOperator<TransposeOperator*>(
      transpose_it->get(), OperatorType::kTranspose);

  if (transpose_op == nullptr) {
    return ::tensorflow::OkStatus();
  }

  if (!OperatorReady(*model, transpose_op) || transpose_op->perm.empty()) {
    // Wait for values to propagate.
    return ::tensorflow::OkStatus();
  }

  // Find the operator that produces the transpose op.
  auto reshape_it = FindOpWithOutput(*model, transpose_op->inputs[0]);
  if (reshape_it == model->operators.end()) {
    return ::tensorflow::OkStatus();
  }

  TensorFlowReshapeOperator* reshape_op =
      ConvertOperator<TensorFlowReshapeOperator*>(reshape_it->get(),
                                                  OperatorType::kReshape);
  if (reshape_op == nullptr) {
    return ::tensorflow::OkStatus();
  }

  // Ignore if the reshape is uninitialized.
  if (!OperatorReady(*model, reshape_op) || reshape_op->shape.empty()) {
    return ::tensorflow::OkStatus();
  }

  // Need to copy to keep static if permutated.
  const std::string input_name = reshape_op->inputs[0];
  const std::string intermediate_name = reshape_op->outputs[0];
  const std::string output_name = transpose_op->outputs[0];

  // Intermediate should not be consumed by any other operators.
  if (CountOpsWithInput(*model, intermediate_name) != 1) {
    AddMessageF("Input %s used elsewhere", intermediate_name);
    return ::tensorflow::OkStatus();
  }

  // Check that the intermediate is not an output array.
  if (!IsDiscardableArray(*model, intermediate_name)) {
    AddMessageF(
        "Cannot reorder reshape-transpose as it would invalidate %s which is "
        "an output array.",
        intermediate_name);
    return ::tensorflow::OkStatus();
  }

  // Get the arrays.
  const auto& input_array = model->GetArray(input_name);
  const auto& intermediate_array = model->GetArray(intermediate_name);
  const auto& output_array = model->GetArray(output_name);

  // Get the shapes of each array.
  Shape input_shape = input_array.shape();
  Shape intermediate_shape = intermediate_array.shape();
  Shape output_shape = output_array.shape();

  // Assign ids to non-unary indices.
  std::vector<int> input_dims = input_shape.dims();
  std::vector<int> intermediate_dims = intermediate_shape.dims();
  std::vector<int> output_dims = output_shape.dims();

  // If the reshape is equivalent to a transpose with fewer/more unary
  // dimensions then it can be moved between the transpose.
  if (!ReshapeIsEquivalentToTranspose(*model, reshape_op,
                                      true /*allow_extra_unary_dims*/)) {
    return ::tensorflow::OkStatus();
  }

  if (!IsDiscardableArray(*model, output_name)) {
    // The output name of the sequence needs to stay static, so create a new
    // array new use for the intermediate.
    const auto new_intermediate_name =
        AvailableArrayName(*model, transpose_op->outputs[0] + "_exchange");
    AddMessageF("Adding new array %s to preserve output array name %s",
                new_intermediate_name, transpose_op->outputs[0]);
    transpose_op->inputs[0] = input_name;
    transpose_op->outputs[0] = new_intermediate_name;
    reshape_op->inputs[0] = new_intermediate_name;
    reshape_op->outputs[0] = output_name;
    DeleteArrayIfUnused(intermediate_name, model);
  } else {
    // The intermediate array is now the output array.
    for (size_t i = 0; i < model->operators.size(); i++) {
      Operator* consumer = model->operators[i].get();
      for (size_t j = 0; j < consumer->inputs.size(); j++) {
        if (consumer->inputs[j] == output_name) {
          consumer->inputs[j] = intermediate_name;
        }
      }
    }

    transpose_op->inputs[0] = input_name;
    reshape_op->inputs[0] = output_name;
  }

  // If transposes constant buffer is used elsewhere, make a new copy.
  if (CountOpsWithInput(*model, transpose_op->inputs[1]) != 1) {
    transpose_op->inputs[1] =
        AvailableArrayName(*model, transpose_op->inputs[1] + "_copy");
  }

  // Make the new transpose permutation.
  const std::vector<int> new_perm =
      ComputeNewPerm(input_dims, intermediate_dims, transpose_op->perm);
  CHECK_EQ(input_dims.size(), new_perm.size());

  auto& transpose_array = model->GetOrCreateArray(transpose_op->inputs[1]);
  transpose_array.data_type = ArrayDataType::kInt32;
  transpose_array.GetMutableBuffer<ArrayDataType::kInt32>().data = new_perm;
  *(transpose_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(new_perm.size())};
  transpose_op->perm = new_perm;

  // If the reshape's constant buffer is reused, create a new one.
  if (CountOpsWithInput(*model, reshape_op->inputs[1]) != 1) {
    reshape_op->inputs[1] =
        AvailableArrayName(*model, reshape_op->inputs[1] + "_copy");
  }

  // We need to modify the reshape input array to target the new output size.
  auto& reshape_array = model->GetOrCreateArray(reshape_op->inputs[1]);
  reshape_array.GetMutableBuffer<ArrayDataType::kInt32>().data = output_dims;
  *(reshape_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(output_shape.dimensions_count())};
  reshape_op->shape.clear();

  AddMessageF("Swapping around operators between %s and %s", input_name,
              output_name);

  model->GetOrCreateArray(transpose_op->outputs[0]).clear_shape();
  model->GetOrCreateArray(reshape_op->outputs[0]).clear_shape();

  // Swap the order of the operators.
  transpose_it->swap(*reshape_it);

  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
