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
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status RemoveTrivialConcatenation::Run(Model* model,
                                                     std::size_t op_index,
                                                     bool* modified) {
  *modified = false;
  const auto concat_it = model->operators.begin() + op_index;
  auto* concat_op = concat_it->get();
  if (concat_op->type != OperatorType::kConcatenation) {
    return absl::OkStatus();
  }
  if (concat_op->inputs.size() != 1) {
    return absl::OkStatus();
  }
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
