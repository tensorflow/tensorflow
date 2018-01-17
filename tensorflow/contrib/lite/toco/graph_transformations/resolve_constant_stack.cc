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
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void Stack(Model* model, StackOperator const& op) {
  auto& output_array = model->GetArray(op.outputs[0]);
  CHECK(output_array.data_type == Type);

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array.GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array.shape()));

  // Stack inputs into buffer
  CHECK_EQ(op.axis, 0) << "Stacking only supported along first axis";
  int dst_offset = 0;
  for (int i = 0; i < op.inputs.size(); i++) {
    // Append array data to output for each input array
    const auto& input_array = model->GetArray(op.inputs[i]);
    int input_size = RequiredBufferSizeForShape(input_array.shape());
    memcpy(&output_data[dst_offset], &input_array.GetBuffer<Type>().data[0],
           input_size * sizeof(Type));
    dst_offset += input_size;
  }
  CHECK_EQ(dst_offset, output_data.size());
}

}  // namespace

bool ResolveConstantStack::Run(Model* model, std::size_t op_index) {
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kStack) {
    return false;
  }
  const auto* op = static_cast<const StackOperator*>(base_op);

  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return false;
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return false;
  }

  for (const auto& input : op->inputs) {
    if (!IsConstantParameterArray(*model, input)) {
      // Yield if any input is mutable
      return false;
    }
  }

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Stack<ArrayDataType::kFloat>(model, *op);
      break;
    case ArrayDataType::kUint8:
      Stack<ArrayDataType::kUint8>(model, *op);
      break;
    case ArrayDataType::kInt32:
      Stack<ArrayDataType::kInt32>(model, *op);
      break;
    case ArrayDataType::kInt64:
      Stack<ArrayDataType::kInt64>(model, *op);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Stack op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  // Erase input arrays if no longer used
  for (const auto& input : op->inputs) {
    if (IsDiscardableArray(*model, input) &&
        CountOpsWithInput(*model, input) == 1) {
      model->arrays.erase(input);
    }
  }

  // Erase the operator
  model->operators.erase(it);
  return true;
}

}  // namespace toco
