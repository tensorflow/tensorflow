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
#include "tensorflow/lite/toco/graph_transformations/lstm_utils.h"

#include <string>

namespace toco {

void CreateOptionalArray(Model* model, std::string* input_array_buffer,
                         const std::string& array_name) {
  *input_array_buffer = array_name;
  model->CreateOptionalArray(array_name);
}

void CopyArrayData(const Buffer<ArrayDataType::kFloat>& src_buffer,
                   int src_stride, int src_start_idx1, int src_start_idx2,
                   Buffer<ArrayDataType::kFloat>* dst_buffer, int dst_stride,
                   int dst_start_idx1, int dst_start_idx2, int dim1_copy_size,
                   int dim2_copy_size) {
  int src_offset = src_start_idx1 * src_stride + src_start_idx2;
  int dst_offset = dst_start_idx1 * dst_stride + dst_start_idx2;
  for (int i = 0; i < dim1_copy_size; i++) {
    for (int j = 0; j < dim2_copy_size; j++) {
      int idx_src = src_offset + i * src_stride + j;
      int idx_dst = dst_offset + i * dst_stride + j;
      dst_buffer->data[idx_dst] = src_buffer.data[idx_src];
    }
  }
}

Buffer<ArrayDataType::kFloat>* CreateFloatArrayBuffer(Model* model,
                                                      std::string* array_name,
                                                      const Shape& shape) {
  *array_name = AvailableArrayName(*model, *array_name);
  auto& array = model->GetOrCreateArray(*array_name);
  array.data_type = ArrayDataType::kFloat;
  array.copy_shape(shape);
  Buffer<ArrayDataType::kFloat>* buffer =
      &(array.GetMutableBuffer<ArrayDataType::kFloat>());
  buffer->data.resize(RequiredBufferSizeForShape(shape));
  return buffer;
}

void CopySubArrayToArray(Model* model, std::string* array_name,
                         const std::string& tensor_name, int dim1_size,
                         int dim2_size, const Array& original_array,
                         int start_idx1, int start_idx2) {
  // Determine whether it's bias or not, create shape, buffer.
  bool is_bias = dim2_size == 1;
  Shape shape = is_bias ? Shape({dim1_size}) : Shape({dim1_size, dim2_size});
  Buffer<ArrayDataType::kFloat>* buffer =
      CreateFloatArrayBuffer(model, array_name, shape);
  auto& orig_buffer = original_array.GetBuffer<ArrayDataType::kFloat>();

  // Copy data from big tensor.
  CopyArrayData(orig_buffer, is_bias ? 1 : original_array.shape().dims(1),
                start_idx1, start_idx2, buffer, dim2_size, 0, 0, dim1_size,
                dim2_size);
}

void CopyArrayToSubArray(Buffer<ArrayDataType::kFloat>& tensor_buffer,
                         int tensor_stride, const Array& sub_array,
                         int start_idx1, int start_idx2) {
  // Get tensor data.
  bool is_bias = sub_array.shape().dims().size() == 1;
  int dim1_copy_size = sub_array.shape().dims()[0];
  int dim2_copy_size = is_bias ? 1 : sub_array.shape().dims(1);
  auto& sub_buffer = sub_array.GetBuffer<ArrayDataType::kFloat>();

  // Copy data from sub tensor.
  CopyArrayData(sub_buffer, dim2_copy_size, 0, 0, &tensor_buffer,
                is_bias ? 1 : tensor_stride, start_idx1, start_idx2,
                dim1_copy_size, dim2_copy_size);
}

bool GetMatchingRnnArray(Model* model,
                         const std::string& back_edge_source_array,
                         std::string* rnn_array) {
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (rnn_state.back_edge_source_array() == back_edge_source_array) {
      *rnn_array = rnn_state.state_array();
      return true;
    }
  }
  return false;
}

}  // namespace toco
