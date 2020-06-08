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
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
bool Slice(SliceOperator const& op, Array const& input_array,
           Array* output_array) {
  // Implementation is taken from the tflite kernel.

  CHECK(input_array.data_type == Type);
  CHECK(output_array->data_type == Type);
  const auto& input_data = input_array.GetBuffer<Type>().data;

  // Create a buffer for the output array.
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));

  std::vector<int> size = op.size;
  if (size.size() != op.begin.size()) {
    // Broadcast the end positions.
    CHECK_EQ(op.size.size(), 1);
    int broadcast_size = size[0];
    while (size.size() < op.begin.size()) size.push_back(broadcast_size);
  }

  // Calculate begin and end indices along each dimension.
  CHECK_LE(op.begin.size(), 4);
  CHECK_LE(size.size(), 4);
  std::vector<int> begin = op.begin;
  std::vector<int> end;
  for (int i = 0; i < begin.size(); ++i) {
    int dim_size = size[i];
    if (dim_size == -1) {
      // -1 means the rest of the dimension.
      dim_size = input_array.shape().dims()[i] - begin[i];
    }
    CHECK_GE(dim_size, 1);
    end.push_back(begin[i] + dim_size - 1);
  }

  // Pad out so that we always have 4 dims, makes this loop easier.
  while (begin.size() < 4) begin.insert(begin.begin(), 0);
  while (end.size() < 4) end.insert(end.begin(), 0);
  Shape padded_shape = input_array.shape();
  while (padded_shape.dimensions_count() < 4) {
    padded_shape.mutable_dims()->insert(padded_shape.mutable_dims()->begin(),
                                        1);
  }

  auto* out_ptr = output_data.data();
  for (int in_b = begin[0]; in_b <= end[0]; ++in_b) {
    for (int in_h = begin[1]; in_h <= end[1]; ++in_h) {
      for (int in_w = begin[2]; in_w <= end[2]; ++in_w) {
        for (int in_d = begin[3]; in_d <= end[3]; ++in_d) {
          *out_ptr++ =
              input_data[Offset(padded_shape, {in_b, in_h, in_w, in_d})];
        }
      }
    }
  }

  return true;
}

}  // namespace

::tensorflow::Status ResolveConstantSlice::Run(Model* model,
                                               std::size_t op_index,
                                               bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kSlice) {
    return ::tensorflow::Status::OK();
  }

  const SliceOperator* op = static_cast<const SliceOperator*>(base_op);

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return ::tensorflow::Status::OK();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return ::tensorflow::Status::OK();
  }

  if (op->begin.empty() || op->size.empty()) {
    // Attributes have not resolved yet.
    return ::tensorflow::Status::OK();
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until the value shape has been resolved.
    return ::tensorflow::Status::OK();
  }
  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    // Yield until the value is constant.
    return ::tensorflow::Status::OK();
  }

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      if (!Slice<ArrayDataType::kFloat>(*op, input_array, &output_array)) {
        return ::tensorflow::Status::OK();
      }
      break;
    case ArrayDataType::kUint8:
      if (!Slice<ArrayDataType::kUint8>(*op, input_array, &output_array)) {
        return ::tensorflow::Status::OK();
      }
      break;
    case ArrayDataType::kInt32:
      if (!Slice<ArrayDataType::kInt32>(*op, input_array, &output_array)) {
        return ::tensorflow::Status::OK();
      }
      break;
    case ArrayDataType::kInt64:
      if (!Slice<ArrayDataType::kInt64>(*op, input_array, &output_array)) {
        return ::tensorflow::Status::OK();
      }
      break;
    case ArrayDataType::kComplex64:
      if (!Slice<ArrayDataType::kComplex64>(*op, input_array, &output_array)) {
        return ::tensorflow::Status::OK();
      }
      break;
    default:
      LOG(FATAL) << "Unsupported data type input to Slice op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
