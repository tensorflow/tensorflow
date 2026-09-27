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
#ifndef TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_LSTM_UTILS_H_
#define TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_LSTM_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/toco/model.h"

namespace toco {

// For consistency with the parameters defined in extended LstmCell's kernel
// (tensorflow/lite/kernels/lstm.cc),
// use kCamelCase for these constants.

enum ExtendedLstmCellInputs {
  kInputTensor = 0,
  kInputToInputWeightsTensor = 1,  // Optional
  kInputToForgetWeightsTensor = 2,
  kInputToCellWeightsTensor = 3,
  kInputToOutputWeightsTensor = 4,
  kRecurrentToInputWeightsTensor = 5,  // Optional
  kRecurrentToForgetWeightsTensor = 6,
  kRecurrentToCellWeightsTensor = 7,
  kRecurrentToOutputWeightsTensor = 8,
  kCellToInputWeightsTensor = 9,    // Optional
  kCellToForgetWeightsTensor = 10,  // Optional
  kCellToOutputWeightsTensor = 11,  // Optional
  kInputGateBiasTensor = 12,        // Optional
  kForgetGateBiasTensor = 13,
  kCellGateBiasTensor = 14,
  kOutputGateBiasTensor = 15,
  kProjectionWeightsTensor = 16,  // Optional
  kProjectionBiasTensor = 17,     // Optional
  kInputActivationStateTensor = 18,
  // The op can handle 18 inputs or 20 inputs.
  kInputCellStateTensor = 19,
  kExtendedLstmInputCount = 20,
};

enum ExtendedLstmCellOutputs {
  kOutputStateTensor = 0,
  kCellStateTensor = 1,
  kOutputTensor = 2,
  kExtendedLstmOutputCount = 3
};

// Creates an optional array in the model and populates the input array buffer
// with its name.
void CreateOptionalArray(Model* model, std::string* input_array_buffer,
                         absl::string_view array_name);

// Creates a new float array with the specified shape in the model's array map
// and returns a non-owning pointer to its mutable buffer.
Buffer<ArrayDataType::kFloat>* CreateFloatArrayBuffer(Model* model,
                                                      std::string* array_name,
                                                      const Shape& shape);

// Copies a 2D submatrix (or 1D vector, where the second dimension size is 1)
// from a source buffer to a destination buffer. The source and destination
// strides specify the total width (second dimension size) of the respective
// buffers, and the start indices define the top-left offset of the copy region.
void CopyArrayData(const Buffer<ArrayDataType::kFloat>& src_buffer,
                   int src_stride, int src_start_idx1, int src_start_idx2,
                   Buffer<ArrayDataType::kFloat>* dst_buffer, int dst_stride,
                   int dst_start_idx1, int dst_start_idx2, int dim1_copy_size,
                   int dim2_copy_size);

// Creates a smaller array in the model and populates it with a submatrix
// region copied from the original array.
void CopySubArrayToArray(Model* model, std::string* array_name,
                         absl::string_view tensor_name, int dim1_size,
                         int dim2_size, const Array& original_array,
                         int start_idx1, int start_idx2);

// Copies data from a subarray into a submatrix region of a larger tensor
// buffer.
void CopyArrayToSubArray(Buffer<ArrayDataType::kFloat>& tensor_buffer,
                         int tensor_stride, const Array& sub_array,
                         int start_idx1, int start_idx2);

// Searches the model's rnn_states flags for an entry matching the back-edge
// source array. Returns true and populates rnn_array with the state array name
// if a match is found; otherwise returns false.
bool GetMatchingRnnArray(Model* model, absl::string_view back_edge_source_array,
                         std::string* rnn_array);

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_LSTM_UTILS_H_
