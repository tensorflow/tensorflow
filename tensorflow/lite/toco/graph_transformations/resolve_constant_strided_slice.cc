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

#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void StridedSlice(StridedSliceOperator const& op, Array const& input_array,
                  Array* output_array) {
  // The TensorFlow documentation for StridedSlice is a bit ambiguous in places
  // (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/strided-slice).
  // Use the source code at tensorflow/core/util/strided_op.cc as
  // "master documentation".

  CHECK(input_array.data_type == Type);
  CHECK(output_array->data_type == Type);
  CHECK_EQ(op.ellipsis_mask, 0);
  CHECK_EQ(op.new_axis_mask, 0);

  int num_input_axes = op.start_indices.size();
  CHECK_EQ(num_input_axes, op.start_indices.size());
  CHECK_EQ(num_input_axes, op.stop_indices.size());
  CHECK_EQ(num_input_axes, op.strides.size());

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));

  // Initialize source coordinate
  Shape const& input_shape = input_array.shape();
  Buffer<Type> const& input_buffer = input_array.GetBuffer<Type>();
  std::vector<int> src_coord(num_input_axes);
  std::vector<int> stop_for_axis(num_input_axes);
  const auto strided_slice_params =
      tflite::strided_slice::BuildStridedSliceParams(
          op.begin_mask, op.end_mask, op.shrink_axis_mask, op.start_indices,
          op.stop_indices, op.strides);

  for (int axis = 0; axis < num_input_axes; axis++) {
    int start_index = tflite::strided_slice::StartForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis);
    src_coord[axis] = start_index;
    stop_for_axis[axis] = tflite::strided_slice::StopForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis,
        start_index);
  }

  // In order to handle any number (N) of dimensions, we copy elements one by
  // one and treat the source coordinate as an N digit number (src_coord here).
  // Each "digit" is incremented individually (by the stride). When it overflows
  // (becomes greater than the stop), that digit is reset and a carry flag is
  // used to increment the next digit.
  for (size_t dst_offset = 0; dst_offset < output_data.size(); ++dst_offset) {
    // Copy element.
    output_data[dst_offset] = input_buffer.data[Offset(input_shape, src_coord)];

    // Note we consider elements in the highest dimension are stored
    // contiguously. So, we increment the stride starting from the highest
    // dimension.
    for (int axis = num_input_axes - 1; axis >= 0; --axis) {
      int stride = op.strides[axis];
      src_coord[axis] += stride;

      // Check if we've overflowed. If not, we just break from the loop to
      // continue w/ the element copy. Otherwise, reset the starting coordinate
      // for this axis and move to the next lower axis.
      int stop = stop_for_axis[axis];
      if (!tflite::strided_slice::LoopCondition(src_coord[axis], stop,
                                                stride)) {
        break;
      }
      src_coord[axis] = tflite::strided_slice::StartForAxis(
          strided_slice_params, ToRuntimeShape(input_shape), axis);
    }
  }
}

}  // anonymous namespace

::tensorflow::Status ResolveConstantStridedSlice::Run(Model* model,
                                                      std::size_t op_index,
                                                      bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kStridedSlice) {
    return ::tensorflow::OkStatus();
  }

  const StridedSliceOperator* op =
      static_cast<const StridedSliceOperator*>(base_op);

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::OkStatus();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return ::tensorflow::OkStatus();
  }

  if (op->start_indices.empty() || op->stop_indices.empty() ||
      op->strides.empty()) {
    // Attributes have not resolved yet.
    return ::tensorflow::OkStatus();
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until the value shape has been resolved.
    return ::tensorflow::OkStatus();
  }
  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    // Yield until the value is constant.
    return ::tensorflow::OkStatus();
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
    case ArrayDataType::kComplex64:
      StridedSlice<ArrayDataType::kComplex64>(*op, input_array, &output_array);
      break;
    default:
      LOG(FATAL)
          << "Unsupported data type input to StridedSlice op with output \""
          << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
