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

bool ResolveTensorFlowMerge::Run(Model* model, std::size_t op_index) {
  const auto merge_it = model->operators.begin() + op_index;
  const auto* merge_op = merge_it->get();
  if (merge_op->type != OperatorType::kTensorFlowMerge) {
    return false;
  }

  // We need to yield until this Merge node has only 1 input, which will mean
  // that that is the selected input. Other graph transformations on other nodes
  // such as ResolveTensorFlowSwitch, will take care of trimming the
  // non-selected inputs, so that at some point there will be only 1 input left.
  if (merge_op->inputs.size() > 1) {
    AddMessageF("Waiting for %s to be resolved", LogName(*merge_op));
    return false;
  }

  // Now that the merge node has 1 input exactly, it is the same as an Identity
  // node and can be resolved trivially.
  CHECK_EQ(merge_op->inputs.size(), 1);

  // Update the edges of the graph ahead of removing the node.
  for (const auto& other_op : model->operators) {
    for (auto& input : other_op->inputs) {
      if (input == merge_op->outputs[0]) {
        input = merge_op->inputs[0];
      }
    }
  }

  // Remove the node and its output array.
  AddMessageF("Removing already-resolved %s", LogName(*merge_op));
  model->arrays.erase(merge_op->outputs[0]);
  model->operators.erase(merge_it);
  return true;
}

}  // namespace toco
