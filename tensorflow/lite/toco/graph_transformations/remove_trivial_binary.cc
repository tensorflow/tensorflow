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
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

template <typename Scalar>
bool AreAllBufferElementsEqualTo(const std::vector<Scalar>& buffer_data,
                                 Scalar value) {
  for (const auto& x : buffer_data) {
    if (x != value) {
      return false;
    }
  }
  return true;
}
}  // namespace

// A binary operator is called trivial when exactly one of its operands is
// a constant and is such that the binary operation is equivalent to
// the identity operation on its other input.
// For example, an Add operator is trivial if
// one of its operands is constant 0, a Mul operator is trivial
// if one of its operands is constant 1, etc.
absl::Status RemoveTrivialBinaryOperator::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return absl::OkStatus();
  }

  CHECK_EQ(binary_op->inputs.size(), 2);

  // This graph transformation is only concerned with the case
  // when one input is constant and the other is not constant.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can resolve here.
    return absl::OkStatus();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants
    // propagation, not for us to handle here.
    return absl::OkStatus();
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  // If this was a broadcasting op we can't remove it as we need the broadcast.
  // It's possible we could replace it with a cheaper op, though.
  const auto& input_array_0 = model->GetArray(binary_op->inputs[0]);
  const auto& input_array_1 = model->GetArray(binary_op->inputs[1]);
  if (!input_array_0.has_shape() || !input_array_1.has_shape()) {
    // Both input shapes must be known.
    return absl::OkStatus();
  }
  if (input_array_0.shape().dimensions_count() ==
          input_array_1.shape().dimensions_count() &&
      input_array_0.shape() != input_array_1.shape()) {
    AddMessageF(
        "Preserving %s even though it's trivial as we need to broadcast "
        "(lhs %s, rhs %s)",
        LogName(*binary_op), ShapeToString(input_array_0.shape()),
        ShapeToString(input_array_1.shape()));
    return absl::OkStatus();
  }

  // Now check if the constant operand makes this binary
  // operator trivial.
  const auto& constant_input_array =
      model->GetArray(binary_op->inputs[index_of_constant_input]);
  // For now, we only handle floats here.
  if (constant_input_array.data_type != ArrayDataType::kFloat) {
    return absl::OkStatus();
  }
  const auto& constant_input_float_data =
      constant_input_array.GetBuffer<ArrayDataType::kFloat>().data;
  bool is_trivial = false;
  if (binary_op->type == OperatorType::kAdd) {
    is_trivial = AreAllBufferElementsEqualTo(constant_input_float_data, 0.f);
  } else if (binary_op->type == OperatorType::kSub) {
    is_trivial = index_of_constant_input == 1 &&
                 AreAllBufferElementsEqualTo(constant_input_float_data, 0.f);
  } else if (binary_op->type == OperatorType::kMul) {
    is_trivial = AreAllBufferElementsEqualTo(constant_input_float_data, 1.f);
  } else if (binary_op->type == OperatorType::kDiv) {
    is_trivial = index_of_constant_input == 1 &&
                 AreAllBufferElementsEqualTo(constant_input_float_data, 1.f);
  }

  is_trivial = is_trivial && binary_op->fused_activation_function ==
                                 FusedActivationFunctionType::kNone;

  if (!is_trivial) {
    return absl::OkStatus();
  }

  // Now we know that this node is trivial, so we can remove it.
  AddMessageF("Removing trivial %s", LogName(*binary_op));
  *modified = RemoveTrivialPassthroughOp(this, model, op_index);
  return absl::OkStatus();
}

}  // namespace toco
