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

#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

// For consistency with the parameters defined in extended LstmCell's kernel
// (tensorflow/lite/kernels/lstm.cc),
// use lowercase for these constants.

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

// Create optional array used for optional tensor in ExtendedLstmCell inputs.
void CreateOptionalArray(Model* model, std::string* input_array_buffer,
                         const std::string& array_name);

// Create float array and get its buffer.
Buffer<ArrayDataType::kFloat>* CreateFloatArrayBuffer(Model* model,
                                                      std::string* array_name,
                                                      const Shape& shape);

// Copy data from one array to the other one (supports 1D and 2D array),
// for 1D array, the 2nd dim's size is 1.
// Arguments:
//   src_buffer: the source buffer
//   src_stride: the stride of source buffer, i.e., 2nd dim's size
//   src_start_idx1: the 1st dim index of start point in src matrix
//   src_start_idx2: the 2nd dim index of start point in src matrix
//   dst_buffer: the destination buffer
//   dst_stride: the stride of destination buffer, i.e., 2nd dim's size
//   dst_start_idx1: the 1st dim index of start point in dst matrix
//   dst_start_idx2: the 2nd dim index of start point in dst matrix
//   dim1_copy_size: 1st dim size of copy data
//   dim2_copy_size: 2nd dim size of copy data
void CopyArrayData(const Buffer<ArrayDataType::kFloat>& src_buffer,
                   int src_stride, int src_start_idx1, int src_start_idx2,
                   Buffer<ArrayDataType::kFloat>* dst_buffer, int dst_stride,
                   int dst_start_idx1, int dst_start_idx2, int dim1_copy_size,
                   int dim2_copy_size);

// Copy a subset of array data and create a smaller array,
// mostly used for spliting weights and bias for Lstm cell.
void CopySubArrayToArray(Model* model, std::string* array_name,
                         const std::string& tensor_name, int dim1_size,
                         int dim2_size, const Array& original_array,
                         int start_idx1, int start_idx2);

// Copy array data to a large array's submatrix,
// mostly used for merging weights and bias for Lstm cell.
void CopyArrayToSubArray(Buffer<ArrayDataType::kFloat>& tensor_buffer,
                         int tensor_stride, const Array& sub_array,
                         int start_idx1, int start_idx2);

// Get mating rnn array inputs using rnn_states flag.
bool GetMatchingRnnArray(Model* model,
                         const std::string& back_edge_source_array,
                         std::string* rnn_array);

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_LSTM_UTILS_H_
