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

#include "tensorflow/contrib/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void StridedSlice(StridedSliceOperator const& op, Array const& input_array,
                  Array* output_array) {
  // The TensorFlow documentation for StridedSlice is a bit ambiguous in places
  // (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/strided-slice).
  // Use the source code at /third_party/tensorflow/core/util/strided_op.cc as
  // "master documentation".

  CHECK(input_array.data_type == Type);
  CHECK(output_array->data_type == Type);
  CHECK_EQ(op.ellipsis_mask, 0);
  CHECK_EQ(op.new_axis_mask, 0);

  int num_input_axes = op.start_indices.size();
  CHECK_EQ(num_input_axes, op.stop_indices.size());
  CHECK_EQ(num_input_axes, op.strides.size());

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));

  // Initialize source coordinate
  Shape const& input_shape = input_array.shape();
  Buffer<Type> const& input_buffer = input_array.GetBuffer<Type>();
  std::vector<int> src_coord(op.start_indices.size());
  for (int axis = 0; axis < num_input_axes; axis++) {
    src_coord[axis] = tflite::strided_slice::StartForAxis(
        op.begin_mask, op.start_indices, op.strides, input_shape.dims().data(),
        axis);
  }

  // In order to handle any number (N) of dimensions, we copy elements one by
  // one and treat the source coordinate as an N digit number (src_coord here).
  // Each "digit" is incremented individually (by the stride). When it overflows
  // (becomes greater than the stop), that digit is reset and a carry flag is
  // used to increment the next digit.
  int dst_offset = 0;
  do {
    // Copy element.
    output_data[dst_offset] = input_buffer.data[Offset(input_shape, src_coord)];

    // Compute next source input coordinates.
    bool carry = true;
    for (int axis = 0; axis < num_input_axes; axis++) {
      int stride = op.strides[axis];
      // Increment this axis if we carried from the previous one
      if (carry) {
        src_coord[axis] += stride;
      }

      // Check if we've overflowed.
      int stop = tflite::strided_slice::StopForAxis(
          op.end_mask, op.stop_indices, op.strides, input_shape.dims().data(),
          axis);
      if (tflite::strided_slice::LoopCondition(src_coord[axis], stop, stride)) {
        // Reset axis and set carry
        src_coord[axis] = tflite::strided_slice::StartForAxis(
            op.begin_mask, op.start_indices, op.strides,
            input_shape.dims().data(), axis);
        carry = true;
      } else {
        carry = false;
      }
    }
    // increment destination buffer offset
    dst_offset++;
  } while (dst_offset < output_data.size());
}

}  // anonymous namespace

bool ResolveConstantStridedSlice::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kStridedSlice) {
    return false;
  }

  const StridedSliceOperator* op =
      static_cast<const StridedSliceOperator*>(base_op);

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

  if (op->start_indices.empty() || op->stop_indices.empty() ||
      op->strides.empty()) {
    // Attributes have not resolved yet.
    return false;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until the value shape has been resolved.
    return false;
  }
  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    // Yield until the value is constant.
    return false;
  }

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      StridedSlice<ArrayDataType::kFloat>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kUint8:
      StridedSlice<ArrayDataType::kUint8>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kInt32:
      StridedSlice<ArrayDataType::kInt32>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kInt64:
      StridedSlice<ArrayDataType::kInt64>(*op, input_array, &output_array);
      break;
    default:
      LOG(FATAL)
          << "Unsupported data type input to StridedSlice op with output \""
          << op->outputs[0] << "\"";
      break;
  }

  // Erase input array if no longer used
  if (IsDiscardableArray(*model, op->inputs[0]) &&
      CountOpsWithInput(*model, op->inputs[0]) == 1) {
    model->EraseArray(op->inputs[0]);
  }

  // Erase the operator
  model->operators.erase(it);

  return true;
}

}  // namespace toco
