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
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveStridedSliceAttributes::Run(Model* model, std::size_t op_index) {
  const auto slice_it = model->operators.begin() + op_index;
  auto* slice_op = slice_it->get();
  if (slice_op->type != OperatorType::kStridedSlice) return false;

  auto* op = static_cast<StridedSliceOperator*>(slice_op);
  if (!op->start_indices.empty()) {
    // We have already resolved these attributes
    return false;
  }

  CHECK_EQ(op->inputs.size(), 4);
  const auto& start_array = *model->arrays[op->inputs[1]];
  if (!start_array.has_shape()) return false;

  const auto& stop_array = *model->arrays[op->inputs[2]];
  if (!stop_array.has_shape()) return false;

  const auto& stride_array = *model->arrays[op->inputs[3]];
  if (!stride_array.has_shape()) return false;

  if (!IsConstantParameterArray(*model, op->inputs[1])) return false;
  if (!IsConstantParameterArray(*model, op->inputs[2])) return false;
  if (!IsConstantParameterArray(*model, op->inputs[3])) return false;

  op->start_indices = start_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->stop_indices = stop_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->strides = stride_array.GetBuffer<ArrayDataType::kInt32>().data;

  CHECK_GE(op->start_indices.size(), 1);
  CHECK_LE(op->start_indices.size(), 4);
  CHECK_EQ(op->stop_indices.size(), op->start_indices.size());
  CHECK_EQ(op->strides.size(), op->stop_indices.size());

  // Ideally, we would remove the input arrays after they have been resolved.
  // However, we must then reconstitute these input arrays for all supported
  // export formats. For now, leave the arrays so we don't have to modify our
  // exporters. Ideally, we wouldn't have op attributes, and would work directly
  // with the input arrays.
  return true;
}
}  // namespace toco
