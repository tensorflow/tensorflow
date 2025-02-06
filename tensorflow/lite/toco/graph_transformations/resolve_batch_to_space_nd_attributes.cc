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
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status ResolveBatchToSpaceNDAttributes::Run(Model* model,
                                                          std::size_t op_index,
                                                          bool* modified) {
  *modified = false;
  const auto op_it = model->operators.begin() + op_index;
  if (op_it->get()->type != OperatorType::kBatchToSpaceND)
    return absl::OkStatus();

  auto* op = static_cast<BatchToSpaceNDOperator*>(op_it->get());

  // The attributes are resolved only when the 3 attributes (block_shape,
  // before_crops, after_crops) are all constant.
  if (!op->block_shape.empty()) {
    return absl::OkStatus();
  }

  CHECK_EQ(op->inputs.size(), 3);
  if (!IsConstantParameterArray(*model, op->inputs[1]) ||
      !IsConstantParameterArray(*model, op->inputs[2]))
    return absl::OkStatus();

  // Handle crops
  const auto& crops_array = model->GetArray(op->inputs[2]);
  if (!crops_array.has_shape()) return absl::OkStatus();
  const std::vector<int>& crops_dims = crops_array.shape().dims();
  if (crops_dims.size() != 2) {
    // Code only handles crops of 2 dimensions. Perhaps another transformation
    // will delete this op.
    return absl::OkStatus();
  }
  const std::vector<int>& crops_buffer =
      crops_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (int i = 0; i < crops_dims[0]; ++i) {
    op->before_crops.push_back(crops_buffer[i * 2]);
    op->after_crops.push_back(crops_buffer[i * 2 + 1]);
  }

  // Handle block_shape
  const auto& block_shape_array = model->GetArray(op->inputs[1]);
  if (!block_shape_array.has_shape()) return absl::OkStatus();
  const std::vector<int>& block_shape_dims = block_shape_array.shape().dims();
  CHECK_EQ(block_shape_dims.size(), 1);
  const std::vector<int>& block_shape_buffer =
      block_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (int i = 0; i < block_shape_dims[0]; ++i) {
    op->block_shape.push_back(block_shape_buffer[i]);
  }

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
