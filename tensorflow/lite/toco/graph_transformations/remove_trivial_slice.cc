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
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool IsSliceTrivial(const Model& model, const Operator& op,
                    RemoveTrivialSlice* transformation) {
  CHECK(op.type == OperatorType::kSlice);

  // Slices are trivial if they are slicing the entire input contents.
  const auto& input_array = model.GetArray(op.inputs[0]);
  const auto& output_array = model.GetArray(op.outputs[0]);
  if (input_array.has_shape() && output_array.has_shape()) {
    if (input_array.shape() == output_array.shape()) {
      transformation->AddMessageF(
          "%s is trivial because its input and output shapes are equal",
          LogName(op));
      return true;
    }
  }

  return false;
}

}  // namespace

absl::Status RemoveTrivialSlice::Run(Model* model, std::size_t op_index,
                                     bool* modified) {
  *modified = false;
  const auto reshape_it = model->operators.begin() + op_index;
  auto* slice_op = reshape_it->get();
  if (slice_op->type != OperatorType::kSlice) {
    return absl::OkStatus();
  }

  if (!IsSliceTrivial(*model, *slice_op, this)) {
    return absl::OkStatus();
  }

  AddMessageF("Removing trivial %s", LogName(*slice_op));

  CHECK_EQ(slice_op->inputs.size(), 3);
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
