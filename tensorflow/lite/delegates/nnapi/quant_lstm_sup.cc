/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"

#include <algorithm>

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegate {
namespace nnapi {

// The function extracts a submatrix of the weights at a given row
// and column offsets from  a 2D matrix
void ExtractQuantLstmWeightsSubmatrix(const TfLiteIntArray* submatrix_dims,
                                      const int32_t offset_row,
                                      const int32_t offset_column,
                                      const TfLiteIntArray* weight_dims,
                                      const uint8_t* weights,
                                      std::vector<uint8_t>* submatrix) {
  auto const& submatrix_rows = submatrix_dims->data[0];
  auto const& submatrix_cols = submatrix_dims->data[1];
  auto const& weight_cols = weight_dims->data[1];

  submatrix->resize(NumElements(submatrix_dims));

  for (uint32_t i = 0; i < submatrix_rows * submatrix_cols; ++i) {
    const uint32_t row = i / submatrix_cols;
    const uint32_t column = i % submatrix_cols;
    (*submatrix)[i] =
        weights[(row + offset_row) * weight_cols + column + offset_column];
  }
}

inline int OutputDepth(const TfLiteIntArray* weight_dims) {
  return weight_dims->data[0] / 4;
}

inline int InputDepth(const TfLiteIntArray* weight_dims) {
  return weight_dims->data[1] - OutputDepth(weight_dims);
}

void SetWeightSubmatrixDims(const TfLiteIntArray* weight_dims,
                            TfLiteIntArray* recurrent_submatrix_dims,
                            TfLiteIntArray* input_submatrix_dims) {
  const auto input_depth = InputDepth(weight_dims);
  const auto output_depth = OutputDepth(weight_dims);

  recurrent_submatrix_dims->data[0] = output_depth;
  recurrent_submatrix_dims->data[1] = output_depth;

  input_submatrix_dims->data[0] = output_depth;
  input_submatrix_dims->data[1] = input_depth;
}

// Doing exactly the opposite work of QuantizedLSTMCell::concatenateWeights
// in NNAPI, decomposing the concat_weights tensor data into its 8 components
// according to the following diagram
//
// +-----------------------------------+
// | recurrentToInput  | inputToInput  |
// |-------------------+---------------|
// | recurrentToCell   | inputToCell   |
// |-------------------+---------------|
// | recurrentToForget | inputToForget |
// |-------------------+---------------|
// | recurrentToOutput | inputToOutput |
// +-----------------------------------+
void DecomposeQuantLstmWeightsTensor(const uint8_t* concat_weights,
                                     const TfLiteIntArray* weight_dims,
                                     std::vector<uint8_t>* recurrent_to_input,
                                     std::vector<uint8_t>* input_to_input,
                                     std::vector<uint8_t>* recurrent_to_cell,
                                     std::vector<uint8_t>* input_to_cell,
                                     std::vector<uint8_t>* recurrent_to_forget,
                                     std::vector<uint8_t>* input_to_forget,
                                     std::vector<uint8_t>* recurrent_to_output,
                                     std::vector<uint8_t>* input_to_output) {
  const auto output_depth = OutputDepth(weight_dims);

  TfLiteIntArray* recurrent_submatrix_dims = TfLiteIntArrayCreate(2);
  TfLiteIntArray* input_submatrix_dims = TfLiteIntArrayCreate(2);
  SetWeightSubmatrixDims(weight_dims, recurrent_submatrix_dims,
                         input_submatrix_dims);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 0 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_input);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 0 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_input);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 1 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_cell);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 1 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_cell);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 2 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_forget);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 2 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_forget);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 3 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_output);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 3 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_output);

  TfLiteIntArrayFree(recurrent_submatrix_dims);
  TfLiteIntArrayFree(input_submatrix_dims);
}

void DecomposeBiasTensor(const int32_t* biases, int bias_size,
                         std::vector<int32_t>* input_bias,
                         std::vector<int32_t>* cell_bias,
                         std::vector<int32_t>* forget_bias,
                         std::vector<int32_t>* output_bias) {
  input_bias->resize(bias_size);
  std::copy(biases, biases + bias_size, input_bias->begin());

  cell_bias->resize(bias_size);
  std::copy(biases + bias_size, biases + 2 * bias_size, cell_bias->begin());

  forget_bias->resize(bias_size);
  std::copy(biases + 2 * bias_size, biases + 3 * bias_size,
            forget_bias->begin());

  output_bias->resize(bias_size);
  std::copy(biases + 3 * bias_size, biases + 4 * bias_size,
            output_bias->begin());
}

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite
