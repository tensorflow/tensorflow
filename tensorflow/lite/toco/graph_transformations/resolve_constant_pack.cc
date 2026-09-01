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
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void Pack(Model* model, const PackOperator& op) {
  auto& output_array = model->GetArray(op.outputs[0]);
  ABSL_CHECK(output_array.data_type == Type);

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array.GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array.shape()));

  // Pack inputs into buffer
  size_t dst_offset = 0;
  for (const auto& input : op.inputs) {
    // Append array data to output for each input array
    const auto& input_array = model->GetArray(input);
    size_t input_size = RequiredBufferSizeForShape(input_array.shape());
    ABSL_CHECK_GE(input_array.GetBuffer<Type>().data.size(), input_size);
    ABSL_CHECK_LE(dst_offset + input_size, output_data.size());
    memcpy(output_data.data() + dst_offset,
           input_array.GetBuffer<Type>().data.data(),
           input_size * ElementSize(Type));
    dst_offset += input_size;
  }
  ABSL_CHECK_EQ(dst_offset, output_data.size());
}

}  // namespace

absl::Status ResolveConstantPack::Run(Model* model, std::size_t op_index,
                                      bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kPack) {
    return absl::OkStatus();
  }
  const auto* op = static_cast<const PackOperator*>(base_op);

  ABSL_CHECK_GE(op->inputs.size(), 1);
  ABSL_CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return absl::OkStatus();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return absl::OkStatus();
  }

  for (const auto& input : op->inputs) {
    if (!IsConstantParameterArray(*model, input) ||
        !model->GetArray(input).has_shape()) {
      // Yield if any input is mutable or lacks a shape
      return absl::OkStatus();
    }
  }

  int axis = op->axis;
  if (axis < 0) {
    // Handle negative axis
    axis += model->GetArray(op->inputs[0]).shape().dims().size() + 1;
  }
  ABSL_CHECK_EQ(axis, 0) << "Packing only supported along 0th axis";

  ABSL_CHECK(!output_array.buffer);
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
  return absl::OkStatus();
}

}  // namespace toco
