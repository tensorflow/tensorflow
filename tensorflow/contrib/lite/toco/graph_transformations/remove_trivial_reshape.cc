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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool IsReshapeTrivial(const Model& model, const Operator& op,
                      RemoveTrivialReshape* transformation) {
  CHECK(op.type == OperatorType::kTensorFlowReshape);

  // One way in which a reshape can be trivial is if its
  // output shape is == its input shape
  const auto& input_array = model.GetArray(op.inputs[0]);
  const auto& output_array = model.GetArray(op.outputs[0]);
  if (input_array.has_shape() && output_array.has_shape()) {
    if (transformation->treat_expand_dims_as_trivial() &&
        ShapesAgreeUpToExtending(input_array.shape(), output_array.shape())) {
      transformation->AddMessageF(
          "%s is trivial because its input and output shapes are equal up to "
          "extending "
          "by 1's, and we are told to aggressively discard such Reshape ops.",
          LogName(op));
      return true;
    }
    if (input_array.shape().dims() == output_array.shape().dims()) {
      transformation->AddMessageF(
          "%s is trivial because its input and output shapes are equal",
          LogName(op));
      return true;
    }
  }

  // Another way in which a reshape can be trivial is if its output
  // is only consumed by another reshape.
  if (CountOpsWithInput(model, op.outputs[0]) == 1) {
    const auto* next_op = GetOpWithInput(model, op.outputs[0]);
    if (next_op->type == OperatorType::kTensorFlowReshape) {
      transformation->AddMessageF(
          "%s is trivial because its output is only consumed by another "
          "Reshape op",
          LogName(op));
      return true;
    }
  }

  return false;
}

}  // namespace

bool RemoveTrivialReshape::Run(Model* model, std::size_t op_index) {
  const auto reshape_it = model->operators.begin() + op_index;
  auto* reshape_op = reshape_it->get();
  if (reshape_op->type != OperatorType::kTensorFlowReshape) {
    return false;
  }

  if (!IsReshapeTrivial(*model, *reshape_op, this)) {
    return false;
  }

  AddMessageF("Removing trivial %s", LogName(*reshape_op));

  CHECK_EQ(reshape_op->inputs.size(), 2);
  return RemoveTrivialPassthroughOp(this, model, op_index);
}

}  // namespace toco
