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
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveFakeQuantArgsFromVars::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::OkStatus();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(fakequant_base_op);

  if (fakequant_op->minmax) {
    // Already resolved.
    return ::tensorflow::OkStatus();
  }

  CHECK_EQ(fakequant_op->inputs.size(), 3);
  // We need to yield until the min and max parameters have been
  // resolved to constant arrays.
  for (int i = 1; i <= 2; i++) {
    if (!IsConstantParameterArray(*model, fakequant_op->inputs[i])) {
      return ::tensorflow::OkStatus();
    }
  }

  // Obtain the final min/max values
  const auto& min_array = model->GetArray(fakequant_op->inputs[1]);
  const auto& max_array = model->GetArray(fakequant_op->inputs[2]);
  CHECK_EQ(RequiredBufferSizeForShape(min_array.shape()), 1);
  CHECK_EQ(RequiredBufferSizeForShape(max_array.shape()), 1);
  fakequant_op->minmax.reset(new MinMax);
  MinMax& minmax = *fakequant_op->minmax;
  minmax.min = min_array.GetBuffer<ArrayDataType::kFloat>().data[0];
  minmax.max = max_array.GetBuffer<ArrayDataType::kFloat>().data[0];
  // We always want [min, max] to contain 0.
  if (minmax.min > 0 || minmax.max < 0) {
    LOG(WARNING) << "For " << LogName(*fakequant_op) << " the MinMax range "
                 << "[" << minmax.min << ", " << minmax.max
                 << "] does not contain 0. "
                 << "Proceeding by tweaking it to contain 0, which will result "
                    "in poor accuracy.";
  }
  minmax.min = std::min(minmax.min, 0.);
  minmax.max = std::max(minmax.max, 0.);

  // We won't use the input arrays that provided these min and max
  // values, anymore. Delete them unless they are used by something
  // else.
  for (int i = 1; i <= 2; i++) {
    DeleteArrayIfUnusedOutsideOfOp(fakequant_op->inputs[i], fakequant_op,
                                   model);
  }
  fakequant_op->inputs.resize(1);
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
