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
#include <algorithm>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool IsTailOfShape(const Shape& tail, const Shape& shape) {
  // Return true if 'tail' dimensions are the same as the ending dimensions of
  // 'shape'.

  int shape_end = shape.dimensions_count() - 1;
  int tail_end = tail.dimensions_count() - 1;

  if (tail_end > shape_end) {
    // tail cannot be longer than shape.
    return false;
  }

  // Walk dimensions back to front and compare
  for (int i = 0; i <= tail_end; i++) {
    if (shape.dims(shape_end - i) != tail.dims(tail_end - i)) {
      return false;
    }
  }
  return true;
}

}  // namespace

// If a binary operator is doing a broadcast operation from a constant array,
// and the constant array shape is the tail of both the other input shape, and a
// subsequent reshape op's output shape, we can swap their order. Since we
// prefer to have reshape ops after mathematic ops, this can allow for the
// collapsing of some reshapes. The WaveNet model in particular benefits from
// this transformation.
//
// Note we are testing for one particular case of a broader set of possible
// binary-reshape op transformations. This transformation could be generalized.
::tensorflow::Status MoveBinaryOperatorBeforeReshape::Run(Model* model,
                                                          std::size_t op_index,
                                                          bool* modified) {
  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  Operator* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv &&
      binary_op->type != OperatorType::kFloorDiv &&
      binary_op->type != OperatorType::kFloorMod &&
      binary_op->type != OperatorType::kMinimum &&
      binary_op->type != OperatorType::kMaximum &&
      binary_op->type != OperatorType::kLess &&
      binary_op->type != OperatorType::kLessEqual &&
      binary_op->type != OperatorType::kGreater &&
      binary_op->type != OperatorType::kGreaterEqual) {
    return ::tensorflow::Status::OK();
  }

  // BINARY OP INPUT CHECKS
  CHECK_EQ(binary_op->inputs.size(), 2);
  const bool input_is_const[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!input_is_const[0] && !input_is_const[1]) {
    // To limit our scope, we require one constant input. Though there's no
    // reason this transformation wouldn't work with all variable inputs.
    return ::tensorflow::Status::OK();
  }
  if (input_is_const[0] && input_is_const[1]) {
    // Both inputs are constants. Leave this for constants propagation.
    return ::tensorflow::Status::OK();
  }
  const int constant_input_idx = input_is_const[0] ? 0 : 1;
  const int variable_input_idx = input_is_const[0] ? 1 : 0;
  CHECK(input_is_const[constant_input_idx]);
  CHECK(!input_is_const[variable_input_idx]);

  const auto& variable_input_array =
      model->GetArray(binary_op->inputs[variable_input_idx]);
  if (!variable_input_array.has_shape()) {
    AddMessageF(
        "Not moving %s because it's non-constant input shape is not resolved.",
        LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  if (!IsTailOfShape(
          model->GetArray(binary_op->inputs[constant_input_idx]).shape(),
          model->GetArray(binary_op->inputs[variable_input_idx]).shape())) {
    // Constant array shape must be the latter part of the variable shape.
    return ::tensorflow::Status::OK();
  }

  // RESHAPE OP CHECKS
  auto reshape_it =
      FindOpWithOutput(*model, binary_op->inputs[variable_input_idx]);
  if (reshape_it == model->operators.end()) {
    AddMessageF("Not moving %s because it's variable input is not connected.",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  Operator* reshape_op = reshape_it->get();
  if (reshape_op->type != OperatorType::kReshape) {
    AddMessageF("Not moving %s because the preceding %s is not a reshape op",
                LogName(*binary_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }
  const auto& reshape_input_array = model->GetArray(reshape_op->inputs[0]);
  if (!reshape_input_array.has_shape()) {
    AddMessageF(
        "Not moving %s because it's non-constant input shape is not resolved "
        "yet",
        LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }
  if (!IsTailOfShape(
          model->GetArray(binary_op->inputs[constant_input_idx]).shape(),
          model->GetArray(reshape_op->outputs[0]).shape())) {
    // Constant array shape must be the latter part of the binary op output
    // shape.
    return ::tensorflow::Status::OK();
  }

  // EXTRA CHECKS ON CONNECTING ARRAY
  for (const std::string& output_array : model->flags.output_arrays()) {
    if (binary_op->inputs[variable_input_idx] == output_array) {
      AddMessageF(
          "Not moving %s because the output of reshape op %s is an output op.",
          LogName(*binary_op), LogName(*reshape_op));
      return ::tensorflow::Status::OK();
    }
  }
  int count_ops_consuming_output =
      CountOpsWithInput(*model, binary_op->inputs[variable_input_idx]);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not moving %s because the output of reshape op %s is consumed by "
        "another op",
        LogName(*binary_op), LogName(*reshape_op));
    return ::tensorflow::Status::OK();
  }

  // SWAP ORDER OF BINARY AND RESHAPE OPS
  AddMessageF("Moving op %s before reshape op %s", LogName(*binary_op),
              LogName(*reshape_op));

  // Swap op input and outputs
  std::iter_swap(reshape_op->inputs.begin(),
                 binary_op->inputs.begin() + variable_input_idx);
  std::iter_swap(reshape_op->outputs.begin(), binary_op->outputs.begin());

  // Swap operator ordering
  std::iter_swap(binary_it, reshape_it);

  // Clear binary output shape so it will be re-propagated
  model->GetArray(binary_op->outputs[0]).clear_shape();

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
