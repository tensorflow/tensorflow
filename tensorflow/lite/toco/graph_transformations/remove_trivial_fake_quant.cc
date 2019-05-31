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
#include <iterator>
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

namespace {

bool IsFakeQuantTrivial(GraphTransformation* transformation, const Model& model,
                        const FakeQuantOperator& fakequant_op) {
  CHECK(fakequant_op.type == OperatorType::kFakeQuant);

  if (!fakequant_op.minmax) {
    // Require ReadFakeQuantMinMax to have run.
    return false;
  }

  // FakeQuants are trivial if they are taking input from another identical
  // FakeQuant op.
  auto* producing_op = GetOpWithOutput(model, fakequant_op.inputs[0]);
  if (!producing_op || producing_op->type != OperatorType::kFakeQuant) {
    return false;
  }
  const auto& producing_fakequant_op =
      *static_cast<FakeQuantOperator*>(producing_op);
  if (!producing_fakequant_op.minmax) {
    // Require ReadFakeQuantMinMax to have run.
    return false;
  }

  if (*fakequant_op.minmax == *producing_fakequant_op.minmax &&
      fakequant_op.num_bits == producing_fakequant_op.num_bits) {
    transformation->AddMessageF(
        "%s is trivial because it is preceded by an identical FakeQuant %s",
        LogName(fakequant_op), LogName(producing_fakequant_op));
    return true;
  }

  return false;
}

}  // namespace

// Removes FakeQuant ops that are trivial (have no effect, are redundant, etc).
::tensorflow::Status RemoveTrivialFakeQuant::Run(Model* model,
                                                 std::size_t op_index,
                                                 bool* modified) {
  *modified = false;
  const auto op_it = model->operators.begin() + op_index;
  auto* op = op_it->get();
  if (op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(op);

  if (!IsFakeQuantTrivial(this, *model, *fakequant_op)) {
    AddMessageF("%s is not trivial", LogName(*fakequant_op));
    return ::tensorflow::Status::OK();
  }

  AddMessageF("Removing trivial %s", LogName(*fakequant_op));

  CHECK_EQ(fakequant_op->inputs.size(), 1);
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return ::tensorflow::Status::OK();
}

}  // namespace toco
