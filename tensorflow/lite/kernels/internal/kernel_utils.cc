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
#include "tensorflow/lite/kernels/internal/kernel_utils.h"

#include <algorithm>

#include "tensorflow/lite/kernels/internal/tensor_utils.h"

namespace tflite {
namespace kernel_utils {

void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch) {
  RnnBatchStep(input_ptr_batch, input_weights_ptr,
               /*aux_input_ptr_batch=*/nullptr,
               /*aux_input_weights_ptr=*/nullptr, recurrent_weights_ptr,
               bias_ptr, input_size, /*aux_input_size=*/0, num_units,
               batch_size, output_batch_leading_dim, activation,
               hidden_state_ptr_batch, output_ptr_batch);
}

void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* aux_input_ptr_batch,
                  const float* aux_input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int aux_input_size, int num_units,
                  int batch_size, int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch) {
  // Since the output batch rows may not be contiguous (output_batch_leading_dim
  // != n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == num_units) {
    // Output = bias
    tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                          output_ptr_batch);

    // Output += input * input_weights
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_weights_ptr, num_units, input_size, input_ptr_batch, batch_size,
        output_ptr_batch);

    // Output += aux_input * aux_input_weights (if they are not empty).
    if (aux_input_size > 0) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_weights_ptr, num_units, aux_input_size, aux_input_ptr_batch,
          batch_size, output_ptr_batch);
    }

    // Output += recurrent_weights * hidden_state
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_weights_ptr, num_units, num_units, hidden_state_ptr_batch,
        batch_size, output_ptr_batch);

    // Output = activation(Output) and update hidden_state
    tensor_utils::ApplyActivationToVector(
        output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
    std::copy_n(output_ptr_batch, num_units * batch_size,
                hidden_state_ptr_batch);
  } else {
    // Output = bias
    for (int k = 0; k < batch_size; k++) {
      std::copy_n(bias_ptr, num_units,
                  output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output += input * input_weights
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_weights_ptr, num_units, input_size,
          input_ptr_batch + k * input_size, /*n_batch=*/1,
          output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output += aux_input * aux_input_weights (if they are not empty).
    if (aux_input_size > 0) {
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            aux_input_weights_ptr, num_units, aux_input_size,
            aux_input_ptr_batch + k * aux_input_size,
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
      }
    }

    // Output += recurrent_weights * hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_weights_ptr, num_units, num_units,
          hidden_state_ptr_batch + k * num_units,
          /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output = activation(Output) and update hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::ApplyActivationToVector(
          output_ptr_batch + k * output_batch_leading_dim, num_units,
          activation, output_ptr_batch + k * output_batch_leading_dim);
      std::copy_n(output_ptr_batch + k * output_batch_leading_dim, num_units,
                  hidden_state_ptr_batch + k * num_units);
    }
  }
}

void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const int8_t* recurrent_weights_ptr,
    float recurrent_weights_scale, const float* bias_ptr, int input_size,
    int num_units, int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch) {
  RnnBatchStep(input_ptr_batch, input_weights_ptr, input_weights_scale,
               /*aux_input_ptr_batch=*/nullptr,
               /*aux_input_weights_ptr=*/nullptr,
               /*aux_input_weights_scale=*/0.0f, recurrent_weights_ptr,
               recurrent_weights_scale, bias_ptr, input_size,
               /*aux_input_size=*/0, num_units, batch_size,
               output_batch_leading_dim, activation, quantized_input_ptr_batch,
               /*aux_quantized_input_ptr_batch=*/nullptr,
               quantized_hidden_state_ptr_batch, scaling_factors,
               hidden_state_ptr_batch, output_ptr_batch);
}

void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const float* aux_input_ptr_batch,
    const int8_t* aux_input_weights_ptr, float aux_input_weights_scale,
    const int8_t* recurrent_weights_ptr, float recurrent_weights_scale,
    const float* bias_ptr, int input_size, int aux_input_size, int num_units,
    int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* aux_quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch) {
  // Since the output batch rows may not be contiguous (output_batch_leading_dim
  // != n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == num_units) {
    // Output = bias
    tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                          output_ptr_batch);

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
      // Quantize input from float to uint8 + quantization params (scaling
      // factor).
      float unused_min, unused_max;
      // TODO(mirkov,raziel): replace this for-loop with a MACRO (or function)
      // whichever is faster.
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * input_size;
        tensor_utils::SymmetricQuantizeFloats(
            input_ptr_batch + offset, input_size,
            quantized_input_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= input_weights_scale;
      }

      // Output += input * input_weights
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_weights_ptr, num_units, input_size, quantized_input_ptr_batch,
          scaling_factors, batch_size, output_ptr_batch);
    }

    if (aux_input_ptr_batch &&
        !tensor_utils::IsZeroVector(aux_input_ptr_batch,
                                    batch_size * aux_input_size)) {
      float unused_min, unused_max;
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * aux_input_size;
        tensor_utils::SymmetricQuantizeFloats(
            aux_input_ptr_batch + offset, aux_input_size,
            aux_quantized_input_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= aux_input_weights_scale;
      }

      // Output += aux_input * aux_input_weights
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_weights_ptr, num_units, aux_input_size,
          aux_quantized_input_ptr_batch, scaling_factors, batch_size,
          output_ptr_batch);
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(hidden_state_ptr_batch,
                                    batch_size * num_units)) {
      // Quantize hidden_state
      float unused_min, unused_max;
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * num_units;
        tensor_utils::SymmetricQuantizeFloats(
            hidden_state_ptr_batch + offset, num_units,
            quantized_hidden_state_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= recurrent_weights_scale;
      }

      // Output += recurrent_weights * hidden_state
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_weights_ptr, num_units, num_units,
          quantized_hidden_state_ptr_batch, scaling_factors, batch_size,
          output_ptr_batch);
    }

    // Output = activation(Output) and update hidden_state
    tensor_utils::ApplyActivationToVector(
        output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
    std::copy_n(output_ptr_batch, num_units * batch_size,
                hidden_state_ptr_batch);
  } else {
    // Output = bias
    for (int k = 0; k < batch_size; k++) {
      std::copy_n(bias_ptr, num_units,
                  output_ptr_batch + k * output_batch_leading_dim);
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
      // Quantize input from float to uint8 + quantization params (scaling
      // factor).
      float unused_min, unused_max;
      // TODO(mirkov,raziel): replace this for-loop with a MACRO (or function)
      // whichever is faster.
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * input_size;
        tensor_utils::SymmetricQuantizeFloats(
            input_ptr_batch + offset, input_size,
            quantized_input_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= input_weights_scale;
      }

      // Output += input * input_weights
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            input_weights_ptr, num_units, input_size,
            quantized_input_ptr_batch + k * input_size, &scaling_factors[k],
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
      }
    }

    if (aux_input_ptr_batch &&
        !tensor_utils::IsZeroVector(aux_input_ptr_batch,
                                    batch_size * aux_input_size)) {
      float unused_min, unused_max;
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * aux_input_size;
        tensor_utils::SymmetricQuantizeFloats(
            aux_input_ptr_batch + offset, aux_input_size,
            aux_quantized_input_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= aux_input_weights_scale;
      }

      // Output += aux_input * aux_input_weights
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            aux_input_weights_ptr, num_units, aux_input_size,
            aux_quantized_input_ptr_batch + k * aux_input_size,
            &scaling_factors[k],
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
      }
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(hidden_state_ptr_batch,
                                    batch_size * num_units)) {
      // Quantize hidden_state
      float unused_min, unused_max;
      for (int b = 0; b < batch_size; ++b) {
        const int offset = b * num_units;
        tensor_utils::SymmetricQuantizeFloats(
            hidden_state_ptr_batch + offset, num_units,
            quantized_hidden_state_ptr_batch + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
        scaling_factors[b] *= recurrent_weights_scale;
      }

      // Output += recurrent_weights * hidden_state
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            recurrent_weights_ptr, num_units, num_units,
            quantized_hidden_state_ptr_batch + k * num_units,
            &scaling_factors[k],
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
      }
    }

    // Output = activation(Output) and update hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::ApplyActivationToVector(
          output_ptr_batch + k * output_batch_leading_dim, num_units,
          activation, output_ptr_batch + k * output_batch_leading_dim);
      std::copy_n(output_ptr_batch + k * output_batch_leading_dim, num_units,
                  hidden_state_ptr_batch + k * num_units);
    }
  }
}

}  // namespace kernel_utils
}  // namespace tflite
