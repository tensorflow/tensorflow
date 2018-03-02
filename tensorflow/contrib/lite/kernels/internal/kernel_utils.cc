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
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"

namespace tflite {
namespace kernel_utils {

void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch) {
  // Output = bias
  tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                        output_ptr_batch);
  // Output += input * input_weights
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_weights_ptr, num_units, input_size, input_ptr_batch, batch_size,
      output_ptr_batch, /*result_stride=*/1);
  // Output += recurrent_weights * hidden_state
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_weights_ptr, num_units, num_units, hidden_state_ptr_batch,
      batch_size, output_ptr_batch, /*result_stride=*/1);
  // Output = activation(Output) and update hidden_state
  tensor_utils::ApplyActivationToVector(
      output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
  tensor_utils::VectorBatchVectorAssign(output_ptr_batch, num_units, batch_size,
                                        hidden_state_ptr_batch);
}

}  // namespace kernel_utils
}  // namespace tflite
