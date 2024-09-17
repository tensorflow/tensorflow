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
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// This pass will convert an AddN operator with only 2 inputs into a regular Add
// operator, to which more optimizations may apply.
::tensorflow::Status ConvertTrivialAddNToAdd::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  auto addn_it = model->operators.begin() + op_index;
  if (addn_it->get()->type != OperatorType::kAddN) {
    return absl::OkStatus();
  }
  AddNOperator* addn_op = static_cast<AddNOperator*>(addn_it->get());
  CHECK_GE(addn_op->inputs.size(), 2);
  CHECK_EQ(addn_op->outputs.size(), 1);

  // We only reduce AddN with N=2 to a regular Add.
  if (addn_op->inputs.size() != 2) {
    return absl::OkStatus();
  }

  // Copy inputs & outputs to regular Add.
  auto* add_op = new AddOperator;
  add_op->inputs.push_back(addn_op->inputs[0]);
  add_op->inputs.push_back(addn_op->inputs[1]);
  add_op->outputs = addn_op->outputs;

  // Replace the AddN operator in the graph.
  model->operators.emplace(addn_it, add_op);
  DeleteOpAndArrays(model, addn_op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
