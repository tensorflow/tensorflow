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

bool ResolveTransposeAttributes::Run(Model* model, std::size_t op_index) {
  const auto op_it = model->operators.begin() + op_index;
  if (op_it->get()->type != OperatorType::kTranspose) return false;

  auto* op = static_cast<TransposeOperator*>(op_it->get());
  if (!op->perm.empty()) return false;

  CHECK_EQ(op->inputs.size(), 2);
  if (!IsConstantParameterArray(*model, op->inputs[1])) return false;

  // Handling perm.
  const auto& perm_array = *model->arrays[op->inputs[1]];
  if (!perm_array.has_shape()) return false;

  const std::vector<int>& perm_dims = perm_array.shape().dims();
  CHECK_EQ(perm_dims.size(), 1);

  std::vector<int> perm_buffer =
      perm_array.GetBuffer<ArrayDataType::kInt32>().data;
  for (int i = 0; i < perm_dims[0]; ++i) {
    op->perm.push_back(perm_buffer[i]);
  }

  return true;
}

}  // namespace toco
