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
#include "tensorflow/contrib/lite/kernels/internal/kernel_utils.h"

#include <algorithm>

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

void RnnBatchStep(const float* input_ptr_batch, const int8_t* input_weights_ptr,
                  float input_weights_scale,
                  const int8_t* recurrent_weights_ptr,
                  float recurrent_weights_scale, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  TfLiteFusedActivation activation,
                  int8_t* quantized_input_ptr_batch,
                  int8_t* quantized_hidden_state_ptr_batch,
                  float* scaling_factors, float* hidden_state_ptr_batch,
                  float* output_ptr_batch) {
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
        scaling_factors, batch_size, output_ptr_batch, /*result_stride=*/1);
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
        output_ptr_batch, /*result_stride=*/1);
  }

  // Output = activation(Output) and update hidden_state
  tensor_utils::ApplyActivationToVector(
      output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
  tensor_utils::VectorBatchVectorAssign(output_ptr_batch, num_units, batch_size,
                                        hidden_state_ptr_batch);
}

void LstmStep(
    const float* input_ptr_batch, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr, const float* input_gate_bias_ptr,
    const float* forget_gate_bias_ptr, const float* cell_bias_ptr,
    const float* output_gate_bias_ptr, const float* projection_weights_ptr,
    const float* projection_bias_ptr, const TfLiteLSTMParams* params,
    int n_batch, int n_cell, int n_input, int n_output, float* output_state_ptr,
    float* cell_state_ptr, float* input_gate_scratch,
    float* forget_gate_scratch, float* cell_scratch, float* output_gate_scratch,
    float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  const bool use_peephole = (cell_to_output_weights_ptr != nullptr);
  // Initialize scratch buffers with bias.
  if (!use_cifg) {
    tensor_utils::VectorBatchVectorAssign(input_gate_bias_ptr, n_cell, n_batch,
                                          input_gate_scratch);
  }
  tensor_utils::VectorBatchVectorAssign(forget_gate_bias_ptr, n_cell, n_batch,
                                        forget_gate_scratch);
  tensor_utils::VectorBatchVectorAssign(cell_bias_ptr, n_cell, n_batch,
                                        cell_scratch);
  tensor_utils::VectorBatchVectorAssign(output_gate_bias_ptr, n_cell, n_batch,
                                        output_gate_scratch);

  // For each batch and cell: compute input_weight * input.
  if (!use_cifg) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_input_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
        input_gate_scratch, /*result_stride=*/1);
  }

  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_forget_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      forget_gate_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_cell_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      cell_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input_to_output_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      output_gate_scratch, /*result_stride=*/1);

  // For each batch and cell: compute recurrent_weight * output_state.
  if (!use_cifg) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_input_weights_ptr, n_cell, n_output, output_state_ptr,
        n_batch, input_gate_scratch, /*result_stride=*/1);
  }
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_forget_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, forget_gate_scratch,
      /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_cell_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, cell_scratch, /*result_stride=*/1);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_output_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, output_gate_scratch,
      /*result_stride=*/1);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          cell_to_input_weights_ptr, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_forget_weights_ptr, n_cell, cell_state_ptr, n_batch,
        forget_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                     forget_gate_scratch);

  // For each batch and cell: update the cell.
  tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch, cell_state_ptr,
                                         n_batch * n_cell, cell_state_ptr);
  tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                        params->activation, cell_scratch);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                             forget_gate_scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_ptr);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_ptr);
  }
  if (params->cell_clip > 0.0) {
    tensor_utils::ClipVector(cell_state_ptr, n_batch * n_cell,
                             params->cell_clip, cell_state_ptr);
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_output_weights_ptr, n_cell, cell_state_ptr, n_batch,
        output_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                     output_gate_scratch);
  tensor_utils::ApplyActivationToVector(cell_state_ptr, n_batch * n_cell,
                                        params->activation, cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                         n_batch * n_cell, output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);
  if (use_projection_weight) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                            n_batch, output_ptr_batch);
    } else {
      tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        projection_weights_ptr, n_output, n_cell, output_gate_scratch, n_batch,
        output_ptr_batch, /*result_stride=*/1);
    if (params->proj_clip > 0.0) {
      tensor_utils::ClipVector(output_ptr_batch, n_batch * n_output,
                               params->proj_clip, output_ptr_batch);
    }
  } else {
    tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             output_ptr_batch);
  }
  tensor_utils::CopyVector(output_ptr_batch, n_batch * n_output,
                           output_state_ptr);
}

// TODO(alanchiao): move this to tensor_utils.
void VectorMultiply(const int8_t* vector, const int v_size, const float scale,
                    float* result) {
  for (int i = 0; i < v_size; ++i) {
    *result++ = scale * *vector++;
  }
}

void LstmStep(
    const float* input_ptr_batch, const int8_t* input_to_input_weights_ptr,
    float input_to_input_weights_scale,
    const int8_t* input_to_forget_weights_ptr,
    float input_to_forget_weights_scale,
    const int8_t* input_to_cell_weights_ptr, float input_to_cell_weights_scale,
    const int8_t* input_to_output_weights_ptr,
    float input_to_output_weights_scale,
    const int8_t* recurrent_to_input_weights_ptr,
    float recurrent_to_input_weights_scale,
    const int8_t* recurrent_to_forget_weights_ptr,
    float recurrent_to_forget_weights_scale,
    const int8_t* recurrent_to_cell_weights_ptr,
    float recurrent_to_cell_weights_scale,
    const int8_t* recurrent_to_output_weights_ptr,
    float recurrent_to_output_weights_scale,
    const int8_t* cell_to_input_weights_ptr, float cell_to_input_weights_scale,
    const int8_t* cell_to_forget_weights_ptr,
    float cell_to_forget_weights_scale,
    const int8_t* cell_to_output_weights_ptr,
    float cell_to_output_weights_scale, const float* input_gate_bias_ptr,
    const float* forget_gate_bias_ptr, const float* cell_bias_ptr,
    const float* output_gate_bias_ptr, const int8_t* projection_weights_ptr,
    float projection_weights_scale, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_output, float* input_gate_scratch, float* forget_gate_scratch,
    float* cell_scratch, float* output_gate_scratch, float* scaling_factors,
    float* product_scaling_factors, float* recovered_cell_weights,
    int8_t* quantized_input_ptr_batch, int8_t* quantized_output_state_ptr,
    int8_t* quantized_cell_state_ptr, float* output_state_ptr,
    float* cell_state_ptr, float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  const bool use_peephole = (cell_to_output_weights_ptr != nullptr);
  // Initialize scratch buffers with bias.
  if (!use_cifg) {
    tensor_utils::VectorBatchVectorAssign(input_gate_bias_ptr, n_cell, n_batch,
                                          input_gate_scratch);
  }
  tensor_utils::VectorBatchVectorAssign(forget_gate_bias_ptr, n_cell, n_batch,
                                        forget_gate_scratch);
  tensor_utils::VectorBatchVectorAssign(cell_bias_ptr, n_cell, n_batch,
                                        cell_scratch);
  tensor_utils::VectorBatchVectorAssign(output_gate_bias_ptr, n_cell, n_batch,
                                        output_gate_scratch);

  if (!tensor_utils::IsZeroVector(input_ptr_batch, n_batch * n_input)) {
    // Save quantization and matmul computation for all zero input.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_input;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr_batch + offset, n_input, quantized_input_ptr_batch + offset,
          &unused_min, &unused_max, &scaling_factors[b]);
    }
    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * input_to_input_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_to_input_weights_ptr, n_cell, n_input,
          quantized_input_ptr_batch, product_scaling_factors, n_batch,
          input_gate_scratch, /*result_stride=*/1);
    }

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_forget_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_forget_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, forget_gate_scratch,
        /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_cell_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_cell_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, cell_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * input_to_output_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_output_weights_ptr, n_cell, n_input, quantized_input_ptr_batch,
        product_scaling_factors, n_batch, output_gate_scratch,
        /*result_stride=*/1);
  }

  if (!tensor_utils::IsZeroVector(output_state_ptr, n_batch * n_output)) {
    // Save quantization and matmul computation for all zero input.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_output;
      tensor_utils::SymmetricQuantizeFloats(output_state_ptr + offset, n_output,
                                            quantized_output_state_ptr + offset,
                                            &unused_min, &unused_max,
                                            &scaling_factors[b]);
    }
    // For each batch and cell: compute recurrent_weight * output_state.
    if (!use_cifg) {
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * recurrent_to_input_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_to_input_weights_ptr, n_cell, n_output,
          quantized_output_state_ptr, product_scaling_factors, n_batch,
          input_gate_scratch, /*result_stride=*/1);
    }

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_forget_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_forget_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        forget_gate_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_cell_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_cell_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        cell_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * recurrent_to_output_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_output_weights_ptr, n_cell, n_output,
        quantized_output_state_ptr, product_scaling_factors, n_batch,
        output_gate_scratch, /*result_stride=*/1);
  }

  // Save quantization and matmul computation for all zero input.
  bool is_cell_state_all_zeros =
      tensor_utils::IsZeroVector(cell_state_ptr, n_batch * n_cell);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole && !is_cell_state_all_zeros) {
      VectorMultiply(cell_to_input_weights_ptr, n_cell,
                     1. / cell_to_input_weights_scale, recovered_cell_weights);
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          recovered_cell_weights, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole && !is_cell_state_all_zeros) {
    VectorMultiply(cell_to_forget_weights_ptr, n_cell,
                   1. / cell_to_forget_weights_scale, recovered_cell_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_cell_weights, n_cell, cell_state_ptr, n_batch,
        forget_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                     forget_gate_scratch);

  // For each batch and cell: update the cell.
  tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch, cell_state_ptr,
                                         n_batch * n_cell, cell_state_ptr);
  tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                        params->activation, cell_scratch);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                             forget_gate_scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_ptr);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_ptr);
  }
  if (params->cell_clip > 0.0) {
    tensor_utils::ClipVector(cell_state_ptr, n_batch * n_cell,
                             params->cell_clip, cell_state_ptr);
  }

  is_cell_state_all_zeros =
      tensor_utils::IsZeroVector(cell_state_ptr, n_batch * n_cell);
  // For each batch and cell: update the output gate.
  if (use_peephole && !is_cell_state_all_zeros) {
    VectorMultiply(cell_to_output_weights_ptr, n_cell,
                   1. / cell_to_output_weights_scale, recovered_cell_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_cell_weights, n_cell, cell_state_ptr, n_batch,
        output_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                     output_gate_scratch);
  tensor_utils::ApplyActivationToVector(cell_state_ptr, n_batch * n_cell,
                                        params->activation, cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                         n_batch * n_cell, output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);
  if (use_projection_weight) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                            n_batch, output_ptr_batch);
    } else {
      tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
    }
    if (!tensor_utils::IsZeroVector(output_gate_scratch, n_batch * n_cell)) {
      // Save quantization and matmul computation for all zero input.
      float unused_min, unused_max;
      for (int b = 0; b < n_batch; ++b) {
        const int offset = b * n_cell;
        tensor_utils::SymmetricQuantizeFloats(
            output_gate_scratch + offset, n_cell,
            quantized_cell_state_ptr + offset, &unused_min, &unused_max,
            &scaling_factors[b]);
      }
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * projection_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights_ptr, n_output, n_cell, quantized_cell_state_ptr,
          product_scaling_factors, n_batch, output_ptr_batch,
          /*result_stride=*/1);
    }
    if (params->proj_clip > 0.0) {
      tensor_utils::ClipVector(output_ptr_batch, n_batch * n_output,
                               params->proj_clip, output_ptr_batch);
    }
  } else {
    tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                             output_ptr_batch);
  }
  tensor_utils::CopyVector(output_ptr_batch, n_batch * n_output,
                           output_state_ptr);
}

}  // namespace kernel_utils
}  // namespace tflite
