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
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace {

template <typename T>
bool AreAllBufferElementsZero(const std::vector<T>& buffer_data) {
  for (auto x : buffer_data) {
    if (x != 0) {
      return false;
    }
  }
  return true;
}

template <ArrayDataType Type>
void FillArrayWithZeros(Array* array) {
  CHECK(array->data_type == Type);
  std::vector<DataType<Type>>& data = array->GetMutableBuffer<Type>().data;
  data.resize(RequiredBufferSizeForShape(array->shape()));
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = 0;
  }
}

}  // namespace

// Removes a multiplication by array of constant zeros by making the output
// array an array of constant zeros and removing the input arrays if they are no
// longer needed.
bool ResolveMultiplyByZero::Run(Model* model, std::size_t op_index) {
  const auto mul_it = model->operators.begin() + op_index;
  auto* mul_op = mul_it->get();
  if (mul_op->type != OperatorType::kMul) {
    return false;
  }
  const auto& output_array_name = mul_op->outputs[0];
  auto& output_array = model->GetArray(output_array_name);

  // Yield if the output shape is not known yet.
  if (!output_array.has_shape()) {
    return false;
  }

  // This transformation only handles the case where one operand is all 0's and
  // the other is non-constant. Other cases are handled by constant propagation
  // or the trivial binary removal pass.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, mul_op->inputs[0]),
      IsConstantParameterArray(*model, mul_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can resolve here.
    return false;
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants propagation, not
    // for us to handle here.
    return false;
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  const auto& constant_input_array =
      model->GetArray(mul_op->inputs[index_of_constant_input]);

  CHECK(constant_input_array.data_type == output_array.data_type);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kFloat>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kFloat>>(
              constant_input_data)) {
        return false;
      }
      FillArrayWithZeros<ArrayDataType::kFloat>(&output_array);
    } break;
    case ArrayDataType::kUint8: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kUint8>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kUint8>>(
              constant_input_data)) {
        return false;
      }
      FillArrayWithZeros<ArrayDataType::kUint8>(&output_array);
    } break;
    case ArrayDataType::kInt32: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kInt32>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kInt32>>(
              constant_input_data)) {
        return false;
      }
      FillArrayWithZeros<ArrayDataType::kInt32>(&output_array);
    } break;
    case ArrayDataType::kInt64: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kInt64>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kInt64>>(
              constant_input_data)) {
        return false;
      }
      FillArrayWithZeros<ArrayDataType::kInt64>(&output_array);
    } break;
    default:
      AddMessageF(
          "Cannot resolve multiply by 0 because of unsupported data type\n");
      return false;
  }

  // Erase input arrays to the multiply if no longer used
  if (IsDiscardableArray(*model, mul_op->inputs[0]) &&
      CountOpsWithInput(*model, mul_op->inputs[0]) == 1) {
    model->EraseArray(mul_op->inputs[0]);
  }
  if (IsDiscardableArray(*model, mul_op->inputs[1]) &&
      CountOpsWithInput(*model, mul_op->inputs[1]) == 1) {
    model->EraseArray(mul_op->inputs[1]);
  }

  // Erase the multiply operator.
  model->operators.erase(mul_it);

  return true;
}

}  // namespace toco
