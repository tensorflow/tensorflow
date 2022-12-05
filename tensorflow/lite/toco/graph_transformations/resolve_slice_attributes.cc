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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveSliceAttributes::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  const auto slice_it = model->operators.begin() + op_index;
  auto* slice_op = slice_it->get();
  if (slice_op->type != OperatorType::kSlice) return ::tensorflow::OkStatus();

  auto* op = static_cast<SliceOperator*>(slice_op);
  if (!op->begin.empty()) return ::tensorflow::OkStatus();

  CHECK_EQ(op->inputs.size(), 3);
  if (!IsConstantParameterArray(*model, op->inputs[1]))
    return ::tensorflow::OkStatus();
  if (!IsConstantParameterArray(*model, op->inputs[2]))
    return ::tensorflow::OkStatus();

  const auto& begin_array = model->GetArray(op->inputs[1]);
  if (!begin_array.has_shape()) return ::tensorflow::OkStatus();

  const auto& size_array = model->GetArray(op->inputs[2]);
  if (!size_array.has_shape()) return ::tensorflow::OkStatus();

  op->begin = begin_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->size = size_array.GetBuffer<ArrayDataType::kInt32>().data;

  // TODO(dkalenichenko): Delete the extra inputs?

  *modified = true;
  return ::tensorflow::OkStatus();
}
}  // namespace toco
