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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool ShapesAllowSwapping(const string& input_array_name,
                         const string& const_array_name, Model* model) {
  const Array& input_array = model->GetOrCreateArray(input_array_name);
  const Array& const_array = model->GetOrCreateArray(const_array_name);
  // Wait until these shapes have been resolved.
  if (!input_array.has_shape() || !const_array.has_shape()) {
    return false;
  }

  // Currently swapping is not handled for scalar const_array, though that could
  // be done once there is a test model.
  if (RequiredBufferSizeForShape(input_array.shape()) !=
      RequiredBufferSizeForShape(const_array.shape())) {
    return false;
  }

  return true;
}

}  // namespace

// Swaps:
//                   Input
//                      \
//                   (Reshape Op)       Const
//                         \             /
//                       (Add/Sub/Mul/Div op)
//                               |
//                             Output
//
// To:
//
//                     Input            Const
//                        \              /
//                      (Add/Sub/Mul/Div op)
//                               |
//                         (Reshape Op)
//                               |
//                             Output
//
// This can allow Add/Mul ops from batch normalization to be folded into an
// Input op from a FullyConnected layer.
bool SwapElementwiseBinary::Run(Model* model, std::size_t op_index) {
  const auto element_wise_op_it = model->operators.begin() + op_index;
  std::unique_ptr<Operator>& element_wise_op = *element_wise_op_it;
  DCHECK(element_wise_op);

  switch (element_wise_op->type) {
    case OperatorType::kAdd:
    case OperatorType::kSub:
    case OperatorType::kMul:
    case OperatorType::kDiv:
      break;
    default:
      return false;
  }

  int reshape_input = -1;
  Operator* op = GetOpWithOutput(*model, element_wise_op->inputs[0]);
  if (!op) {
    return false;
  }

  if (op->type == OperatorType::kTensorFlowReshape) {
    reshape_input = 0;
  } else {
    op = GetOpWithOutput(*model, element_wise_op->inputs[1]);
    if (!op || op->type != OperatorType::kTensorFlowReshape) {
      return false;
    }
    reshape_input = 1;
  }

  int const_input = (reshape_input == 0) ? 1 : 0;
  const string& const_input_array = element_wise_op->inputs[const_input];
  if (!IsConstantParameterArray(*model, const_input_array)) {
    return false;
  }

  // Do not fold division if denominator is not constant.
  if (element_wise_op->type != OperatorType::kDiv && const_input != 1) {
    return false;
  }

  const auto reshape_it =
      FindOpWithOutput(*model, element_wise_op->inputs[reshape_input]);
  // Note: we take copies of the tensor names here, instead of const-refs as we
  // may overwrite the original names.
  const string reshape_input_name = (*reshape_it)->inputs[0];
  const string intermediate_name = (*reshape_it)->outputs[0];
  const string element_wise_output_name = element_wise_op->outputs[0];

  // Check the reshape op input and const op have their shapes resolved.
  if (!ShapesAllowSwapping(reshape_input_name, const_input_array, model)) {
    return false;
  }

  int count_ops_consuming_output = CountOpsWithInput(*model, intermediate_name);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not exchanging element-wise function with %s because it is "
        "consumed by more than 1 other operator",
        LogName(**reshape_it));
    return false;
  }

  // If the element_wise_op was originally producing an output_array we can't
  // swap as otherwise the output array would change. It'd be nice to still be
  // able to swap but if code is relying on the fetch names instead of array
  // indices this won't work.
  for (int i = 0; i < model->flags.output_arrays_size(); ++i) {
    if (model->flags.output_arrays(i) == element_wise_op->outputs[0]) {
      AddMessageF(
          "Not exchanging activation function with %s to preserve output array "
          "name %s",
          LogName(**reshape_it), element_wise_op->outputs[0]);
      return false;
    }
  }

  // Rewire by changing inputs, including all consumers.
  // TODO(b/76086261): Replace with new utility function.
  Operator* consumer = GetFirstOpWithInput(*model, element_wise_output_name);
  while (consumer) {
    for (int i = 0; i < consumer->inputs.size(); ++i) {
      if (consumer->inputs[i] == element_wise_output_name) {
        consumer->inputs[i] = intermediate_name;
      }
    }
    consumer = GetFirstOpWithInput(*model, element_wise_output_name);
  }
  element_wise_op->inputs[reshape_input] = reshape_input_name;
  (*reshape_it)->inputs[0] = element_wise_output_name;

  // Clear shapes; this will allow shape propagation to fix the sizes for us.
  model->GetOrCreateArray(element_wise_output_name).clear_shape();

  // Finally, swap operators.  Note that this only works when there are no other
  // direct descendents of the reshape operator.
  element_wise_op.swap(*reshape_it);

  return true;
}

}  // namespace toco
