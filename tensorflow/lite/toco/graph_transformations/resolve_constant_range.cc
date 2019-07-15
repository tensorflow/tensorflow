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
#include <cmath>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

template <ArrayDataType A, typename T>
void FillRangeOutput(const Array& start_array, const Array& limit_array,
                     const Array& delta_array, Array* output_array) {
  // Compute buffer contents
  T start = start_array.GetBuffer<A>().data[0];
  T limit = limit_array.GetBuffer<A>().data[0];
  T delta = delta_array.GetBuffer<A>().data[0];
  auto& buffer = output_array->GetMutableBuffer<A>();
  buffer.data.clear();
  int size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));
  for (int i = 0; i < size; ++i) {
    buffer.data.push_back(start + i * delta);
  }
  CHECK_EQ(std::floor((limit - start) / delta), buffer.data.size());
  CHECK_EQ(buffer.data.size(), output_array->shape().dims()[0]);
}

::tensorflow::Status ResolveConstantRange::Run(Model* model,
                                               std::size_t op_index,
                                               bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* base_op = it->get();
  if (base_op->type != OperatorType::kRange) {
    return ::tensorflow::Status::OK();
  }
  auto* op = static_cast<RangeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 3);
  const auto& start_array = model->GetArray(op->inputs[0]);
  if (!start_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return ::tensorflow::Status::OK();
  }
  const auto& limit_array = model->GetArray(op->inputs[1]);
  if (!limit_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return ::tensorflow::Status::OK();
  }
  const auto& delta_array = model->GetArray(op->inputs[2]);
  if (!delta_array.has_shape()) {
    // Yield until all input dims have been resolved.
    return ::tensorflow::Status::OK();
  }

  for (const auto& input : op->inputs) {
    if (!IsConstantParameterArray(*model, input)) {
      // yield if any input is mutable
      return ::tensorflow::Status::OK();
    }
  }

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(RequiredBufferSizeForShape(start_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(limit_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(delta_array.shape()), 1)
      << "Range op inputs must be scalar.";

  CHECK(start_array.data_type == ArrayDataType::kInt32 ||
        start_array.data_type == ArrayDataType::kFloat)
      << "Range op inputs must be int32 or float.";
  CHECK(limit_array.data_type == start_array.data_type)
      << "Range op inputs type must be equal.";
  CHECK(delta_array.data_type == start_array.data_type)
      << "Range op inputs type must be equal.";

  if (start_array.data_type == ArrayDataType::kInt32) {
    FillRangeOutput<ArrayDataType::kInt32, int32_t>(start_array, limit_array,
                                                    delta_array, &output_array);
  } else {
    FillRangeOutput<ArrayDataType::kFloat, float>(start_array, limit_array,
                                                  delta_array, &output_array);
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
