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

::tensorflow::Status RemoveTrivialConcatenationInput::Run(Model* model,
                                                          std::size_t op_index,
                                                          bool* modified) {
  *modified = false;
  // TensorFlow allows Concatenation nodes to have 0-D inputs,
  // and they are then treated as empty i.e. omitted from concatenation,
  // in violation of the notion that 0-D is equivalent to 1x1x1x1.
  // Thus we have to drop these 0-D inputs from Concatenation nodes.
  // Sometimes, there will remain only one non-trivial input, and
  // the other graph transformation RemoveTrivialConcatenation will then drop
  // it.
  const auto concat_it = model->operators.begin() + op_index;
  auto* concat_op = concat_it->get();
  if (concat_op->type != OperatorType::kConcatenation) {
    return ::tensorflow::OkStatus();
  }
  std::vector<std::string> trivial_inputs;
  std::vector<std::string> nontrivial_inputs;
  for (const std::string& input : concat_op->inputs) {
    const auto& input_array = model->GetArray(input);
    const bool is_trivial =
        input_array.has_shape() && input_array.shape().dimensions_count() == 0;
    if (is_trivial) {
      trivial_inputs.push_back(input);
    } else {
      nontrivial_inputs.push_back(input);
    }
  }

  if (trivial_inputs.empty()) {
    return ::tensorflow::OkStatus();
  }

  // Drop trivial inputs.
  concat_op->inputs = nontrivial_inputs;
  for (const std::string& input : trivial_inputs) {
    DeleteArrayIfUnusedOutsideOfOp(input, concat_op, model);
  }
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
