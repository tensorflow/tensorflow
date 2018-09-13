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

bool ResolvePadV2Attributes::Run(Model* model, std::size_t op_index) {
  const auto pad_it = model->operators.begin() + op_index;
  auto* pad_op = pad_it->get();
  if (pad_op->type != OperatorType::kPadV2) return false;

  auto* op = static_cast<PadV2Operator*>(pad_op);
  if (!op->left_padding.empty()) return false;

  CHECK_EQ(op->inputs.size(), 3);
  if (!IsConstantParameterArray(*model, op->inputs[1])) return false;

  const auto& array = model->GetArray(op->inputs[1]);
  if (!array.has_shape()) return false;

  const std::vector<int>& dims = array.shape().dims();
  CHECK_EQ(dims.size(), 2);

  std::vector<int> buffer = array.GetBuffer<ArrayDataType::kInt32>().data;

  for (int i = 0; i < dims[0]; ++i) {
    op->left_padding.push_back(buffer[i * 2]);
    op->right_padding.push_back(buffer[i * 2 + 1]);
  }

  // TODO(dkalenichenko): Delete the extra input?

  return true;
}
}  // namespace toco
