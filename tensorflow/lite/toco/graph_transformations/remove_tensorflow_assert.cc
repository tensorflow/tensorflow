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
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status RemoveTensorFlowAssert::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  const auto assert_it = model->operators.begin() + op_index;
  const auto* assert_op = assert_it->get();
  if (assert_op->type != OperatorType::kAssert) {
    return absl::OkStatus();
  }

  bool changed = false;
  // Remove any other node's dependency on this assert node
  for (const auto& op : model->operators) {
    auto it = op->inputs.begin();
    while (it != op->inputs.end()) {
      if (*it == assert_op->outputs[0]) {
        op->inputs.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
  }
  CHECK(!CountOpsWithInput(*model, assert_op->outputs[0]));

  if (changed) {
    AddMessageF(
        "Prepared for the removal of %s by removing any other op's dependency "
        "on it",
        LogName(*assert_op));
  }

  // That's it. We can stop here, no need to duplicate the work that
  // RemoveUnusedOp will do removing this now-unused node.
  *modified = changed;
  return absl::OkStatus();
}

}  // namespace toco
