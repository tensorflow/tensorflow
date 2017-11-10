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

bool ResolveMeanAttributes::Run(Model* model, std::size_t op_index) {
  auto* mean_op = model->operators[op_index].get();
  if (mean_op->type != OperatorType::kMean) return false;
  auto* op = static_cast<MeanOperator*>(mean_op);

  if (!op->reduction_indices.empty()) return false;
  if (op->inputs.size() != 2) return false;
  if (!IsConstantParameterArray(*model, op->inputs[1])) return false;

  const auto& indices_array = *model->arrays[op->inputs[1]];
  if (!indices_array.has_shape()) return false;

  op->reduction_indices = indices_array.GetBuffer<ArrayDataType::kInt32>().data;

  // At the moment, we only support simultaneous reduction over width and
  // height. This is mainly limited by the fact that currently, the runtime
  // arrays are always 4-dimensional.
  CHECK_EQ(op->reduction_indices.size(), 2);
  CHECK((op->reduction_indices[0] == 1 && op->reduction_indices[1] == 2) ||
        (op->reduction_indices[0] == 2 && op->reduction_indices[1] == 1));

  return true;
}

}  // namespace toco
