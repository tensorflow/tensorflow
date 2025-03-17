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

#include "absl/status/status.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

absl::Status DropFakeQuant::Run(Model* model, std::size_t op_index,
                                bool* modified) {
  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return absl::OkStatus();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(fakequant_base_op);

  if (!fakequant_op->minmax) {
    return absl::OkStatus();
  }

  const auto& output_array = model->GetArray(fakequant_op->outputs[0]);
  if (!output_array.minmax) {
    return absl::OkStatus();
  }

  // Drop min/max inputs
  for (int i = 1, end = fakequant_op->inputs.size(); i < end; i++) {
    if (CountOpsWithInput(*model, fakequant_op->inputs[i]) == 1) {
      model->EraseArray(fakequant_op->inputs[i]);
    }
  }
  fakequant_op->inputs.resize(1);

  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
