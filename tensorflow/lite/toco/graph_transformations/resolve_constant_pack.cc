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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void Pack(Model* model, PackOperator const& op) {
  auto& output_array = model->GetArray(op.outputs[0]);
  CHECK(output_array.data_type == Type);

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array.GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array.shape()));

  // Pack inputs into buffer
  CHECK_EQ(op.axis, 0) << "Packing only supported along first axis";
  int dst_offset = 0;
  for (int i = 0; i < op.inputs.size(); i++) {
    // Append array data to output for each input array
    const auto& input_array = model->GetArray(op.inputs[i]);
    int input_size = RequiredBufferSizeForShape(input_array.shape());
    memcpy(&output_data[dst_offset], &input_array.GetBuffer<Type>().data[0],
           input_size * ElementSize(Type));
    dst_offset += input_size;
  }
  CHECK_EQ(dst_offset, output_data.size());
}

}  // namespace

::tensorflow::Status ResolveConstantPack::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kPack) {
    return ::tensorflow::Status::OK();
  }
  const auto* op = static_cast<const PackOperator*>(base_op);

  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::Status::OK();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return ::tensorflow::Status::OK();
  }

  for (const auto& input : op->inputs) {
    if (!IsConstantParameterArray(*model, input)) {
      // Yield if any input is mutable
      return ::tensorflow::Status::OK();
    }
  }

  int axis = op->axis;
  if (axis < 0) {
    // Handle negative axis
    axis += model->GetArray(op->inputs[0]).shape().dims().size();
  }
  CHECK_EQ(axis, 0) << "Packing only supported along 0th axis";

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Pack<ArrayDataType::kFloat>(model, *op);
      break;
    case ArrayDataType::kUint8:
      Pack<ArrayDataType::kUint8>(model, *op);
      break;
    case ArrayDataType::kInt32:
      Pack<ArrayDataType::kInt32>(model, *op);
      break;
    case ArrayDataType::kInt64:
      Pack<ArrayDataType::kInt64>(model, *op);
      break;
    case ArrayDataType::kComplex64:
      Pack<ArrayDataType::kComplex64>(model, *op);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Pack op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
