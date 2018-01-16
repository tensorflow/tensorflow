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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveSpaceToBatchNDAttributes::Run(Model* model, std::size_t op_index) {
  const auto op_it = model->operators.begin() + op_index;
  if (op_it->get()->type != OperatorType::kSpaceToBatchND) return false;

  auto* op = static_cast<SpaceToBatchNDOperator*>(op_it->get());

  // The attributes are resolved only when the 3 attributes (block_shape,
  // before_paddings, after_paddings) are all constant.
  if (!op->block_shape.empty()) {
    return false;
  }

  const int block_shape_index = 1;
  const int paddings_index = 2;

  CHECK_EQ(op->inputs.size(), 3);
  if (!IsConstantParameterArray(*model, op->inputs[block_shape_index]) ||
      !IsConstantParameterArray(*model, op->inputs[paddings_index]))
    return false;

  // Handling block_shape.
  const auto& block_shape_array = *model->arrays[op->inputs[block_shape_index]];
  if (!block_shape_array.has_shape()) return false;
  const std::vector<int>& block_shape_dims = block_shape_array.shape().dims();
  CHECK_EQ(block_shape_dims.size(), 1);
  std::vector<int> block_shape_buffer =
      block_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (int i = 0; i < block_shape_dims[0]; ++i) {
    op->block_shape.push_back(block_shape_buffer[i]);
  }

  // Handling paddings.
  const auto& paddings_array = *model->arrays[op->inputs[paddings_index]];
  if (!paddings_array.has_shape()) return false;
  const std::vector<int>& paddings_dims = paddings_array.shape().dims();
  CHECK_EQ(paddings_dims.size(), 2);
  std::vector<int> paddings_buffer =
      paddings_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (int i = 0; i < paddings_dims[0]; ++i) {
    op->before_paddings.push_back(paddings_buffer[i * 2]);
    op->after_paddings.push_back(paddings_buffer[i * 2 + 1]);
  }

  return true;
}

}  // namespace toco
