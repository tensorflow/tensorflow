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
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

int PadAttributeArray(Array* attribute_array, std::vector<int> pad_values,
                      int mask) {
  int attribute_dim_count = attribute_array->shape().dims(0);
  int dim_count = pad_values.size();
  if (attribute_dim_count < dim_count) {
    Shape strided_slice_shape = Shape({dim_count});
    attribute_array->copy_shape(strided_slice_shape);
    Buffer<ArrayDataType::kInt32>* buffer =
        &(attribute_array->GetMutableBuffer<ArrayDataType::kInt32>());
    buffer->data.resize(RequiredBufferSizeForShape(strided_slice_shape));
    for (int i = attribute_dim_count; i < dim_count; i++) {
      buffer->data[i] = pad_values[i];
      mask |= 1 << i;
    }
  }
  return mask;
}

::tensorflow::Status ResolveStridedSliceAttributes::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
  *modified = false;
  const auto slice_it = model->operators.begin() + op_index;
  auto* slice_op = slice_it->get();
  if (slice_op->type != OperatorType::kStridedSlice) return absl::OkStatus();

  auto* op = static_cast<StridedSliceOperator*>(slice_op);
  if (!op->start_indices.empty()) {
    // We have already resolved these attributes
    return absl::OkStatus();
  }

  CHECK_EQ(op->inputs.size(), 4);
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // We require the dimensionality of the input to pad the indices
    return absl::OkStatus();
  }

  auto& start_array = model->GetArray(op->inputs[1]);
  if (!start_array.has_shape()) return absl::OkStatus();
  if (toco::RequiredBufferSizeForShape(start_array.shape()) > 4) {
    // Only 1-4D arrays are supported for now.
    return absl::OkStatus();
  }

  auto& stop_array = model->GetArray(op->inputs[2]);
  if (!stop_array.has_shape()) return absl::OkStatus();

  auto& stride_array = model->GetArray(op->inputs[3]);
  if (!stride_array.has_shape()) return absl::OkStatus();

  if (!IsConstantParameterArray(*model, op->inputs[1])) return absl::OkStatus();
  if (!IsConstantParameterArray(*model, op->inputs[2])) return absl::OkStatus();
  if (!IsConstantParameterArray(*model, op->inputs[3])) return absl::OkStatus();

  int num_input_axes = input_array.shape().dimensions_count();
  int start_indices_size = start_array.shape().dims(0);
  int stop_indices_size = stop_array.shape().dims(0);
  int stride_indices_size = stride_array.shape().dims(0);

  CHECK_GE(start_indices_size, 1);
  CHECK_LE(start_indices_size, 4);
  CHECK_LE(stop_indices_size, 4);
  CHECK_LE(stride_indices_size, 4);

  // The TensorFlow documentation is not explicit on how it handles fewer
  // supplied indices than dimensions, but they are accepted. We emulate TF's
  // behavior by fully iterating over each omitted dimension.
  CHECK_LE(start_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " start indices";
  CHECK_LE(stop_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " stop indices";
  CHECK_LE(stride_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " strides";

  // Ideally, we would remove the input arrays after they have been resolved.
  // However, we must then reconstitute these input arrays for all supported
  // export formats. For now, leave the arrays so we don't have to modify our
  // exporters. Ideally, we wouldn't have op attributes, and would work directly
  // with the input arrays.
  std::vector<int> begin_pad_values(num_input_axes, 0);
  op->begin_mask =
      PadAttributeArray(&start_array, begin_pad_values, op->begin_mask);
  op->end_mask =
      PadAttributeArray(&stop_array, input_array.shape().dims(), op->end_mask);
  std::vector<int> stride_pad_values(num_input_axes, 1);
  PadAttributeArray(&stride_array, stride_pad_values, 0);

  op->start_indices = start_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->stop_indices = stop_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->strides = stride_array.GetBuffer<ArrayDataType::kInt32>().data;

  *modified = true;
  return absl::OkStatus();
}
}  // namespace toco
