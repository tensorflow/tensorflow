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
#include "tensorflow/lite/kernels/lstm_eval.h"

#include <cstdint>

#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace lstm_eval {

namespace {

// Performs an LSTM batch inference step for input specified by input_ptr_batch.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_output: the output size.
//  - output_batch_leading_dim: the leading dimension of the output buffer.
//
// The pointers to the cell and output state and the output are updated.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many cell and output states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to n_output. It is usually not
// when we want to store the LSTM output into a slice of the output tensor, e.g.
// for bidirectional LSTMs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
inline void LstmStepWithAuxInput(
    const float* input_ptr_batch, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr, const float* aux_input_ptr_batch,
    const float* aux_input_to_input_weights_ptr,
    const float* aux_input_to_forget_weights_ptr,
    const float* aux_input_to_cell_weights_ptr,
    const float* aux_input_to_output_weights_ptr,
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
    int n_batch, int n_cell, int n_input, int n_aux_input, int n_output,
    int output_batch_leading_dim, float* output_state_ptr,
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

  // If auxiliary input is available then compute aux_input_weight * aux_input
  if (aux_input_ptr_batch != nullptr) {
    if (!use_cifg) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_to_input_weights_ptr, n_cell, n_aux_input,
          aux_input_ptr_batch, n_batch, input_gate_scratch,
          /*result_stride=*/1);
    }

    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_forget_weights_ptr, n_cell, n_aux_input,
        aux_input_ptr_batch, n_batch, forget_gate_scratch, /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_cell_weights_ptr, n_cell, n_aux_input, aux_input_ptr_batch,
        n_batch, cell_scratch, /*result_stride=*/1);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_output_weights_ptr, n_cell, n_aux_input,
        aux_input_ptr_batch, n_batch, output_gate_scratch, /*result_stride=*/1);
  }

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

  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);

  // For each batch: update the projection and output_state. Note that since
  // the output batch rows may not be contiguous (output_batch_leading_dim !=
  // n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == n_output) {
    if (use_projection_weight) {
      if (use_projection_bias) {
        tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                              n_batch, output_ptr_batch);
      } else {
        tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights_ptr, n_output, n_cell, output_gate_scratch,
          n_batch, output_ptr_batch, /*result_stride=*/1);
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
  } else {
    if (use_projection_weight) {
      if (use_projection_bias) {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::CopyVector(
              projection_bias_ptr, n_output,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      } else {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::ZeroVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output);
        }
      }
      for (int k = 0; k < n_batch; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            projection_weights_ptr, n_output, n_cell,
            output_gate_scratch + k * n_cell,
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim,
            /*result_stride=*/1);
        if (params->proj_clip > 0.0) {
          tensor_utils::ClipVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output,
              params->proj_clip,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      }
    } else {
      for (int k = 0; k < n_batch; k++) {
        tensor_utils::CopyVector(
            output_gate_scratch + k * n_output, n_output,
            output_ptr_batch + k * output_batch_leading_dim);
      }
    }
    for (int k = 0; k < n_batch; k++) {
      tensor_utils::CopyVector(output_ptr_batch + k * output_batch_leading_dim,
                               n_output, output_state_ptr + k * n_output);
    }
  }
}

// Same as above but with quantized weight matrices. In detail:
// Input of size 'n_batch * n_input':
//   input_ptr_batch
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional (can be nullptr)
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_input_weights
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Weight scales (scalars) for each of the weights above.
//   input_to_input_weights_scale      - optional
//   input_to_forget_weights_scale
//   input_to_cell_weights_scale
//   input_to_output_weights_scale
//   recurrent_to_input_weights_scale  - optional
//   recurrent_to_forget_weights_scale
//   recurrent_to_cell_weights_scale
//   recurrent_to_output_weights_scale
//   cell_to_input_weights_scale,
//   cell_to_forget_weights_scale,
//   cell_to_output_weights_scale,
//   projection_weights_scale          - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Temporary pre-allocated storage for quantized values:
//   quantized_input_ptr_batch (same size as input_ptr_batch)
//   quantized_output_state_ptr (same size as output_state_ptr)
//   quantized_cell_state_ptr (same size as cell_state_ptr)
// Temporary pre-allocated storage for recovered values:
//   recovered_cell_weights (same size as cell_to_*_weights)
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr_batch - size 'n_batch * output_batch_leading_dim'
inline void LstmStepWithAuxInput(
    const float* input_ptr_batch, const int8_t* input_to_input_weights_ptr,
    float input_to_input_weights_scale,
    const int8_t* input_to_forget_weights_ptr,
    float input_to_forget_weights_scale,
    const int8_t* input_to_cell_weights_ptr, float input_to_cell_weights_scale,
    const int8_t* input_to_output_weights_ptr,
    float input_to_output_weights_scale, const float* aux_input_ptr_batch,
    const int8_t* aux_input_to_input_weights_ptr,
    float aux_input_to_input_weights_scale,
    const int8_t* aux_input_to_forget_weights_ptr,
    float aux_input_to_forget_weights_scale,
    const int8_t* aux_input_to_cell_weights_ptr,
    float aux_input_to_cell_weights_scale,
    const int8_t* aux_input_to_output_weights_ptr,
    float aux_input_to_output_weights_scale,
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
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* input_gate_scratch, float* forget_gate_scratch, float* cell_scratch,
    float* output_gate_scratch, float* scaling_factors,
    float* product_scaling_factors, float* recovered_cell_weights,
    int8_t* quantized_input_ptr_batch, int8_t* quantized_aux_input_ptr_batch,
    int8_t* quantized_output_state_ptr, int8_t* quantized_cell_state_ptr,
    float* output_state_ptr, float* cell_state_ptr, float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we
  // can check the existense of only one to the get the condition.
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

  if (aux_input_ptr_batch != nullptr &&
      !tensor_utils::IsZeroVector(aux_input_ptr_batch, n_batch * n_input)) {
    // Save quantization and matmul computation for all zero input.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_input;
      tensor_utils::SymmetricQuantizeFloats(
          aux_input_ptr_batch + offset, n_input,
          quantized_aux_input_ptr_batch + offset, &unused_min, &unused_max,
          &scaling_factors[b]);
    }
    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
      for (int b = 0; b < n_batch; ++b) {
        product_scaling_factors[b] =
            scaling_factors[b] * aux_input_to_input_weights_scale;
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_to_input_weights_ptr, n_cell, n_input,
          quantized_aux_input_ptr_batch, product_scaling_factors, n_batch,
          input_gate_scratch, /*result_stride=*/1);
    }

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * aux_input_to_forget_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_forget_weights_ptr, n_cell, n_input,
        quantized_aux_input_ptr_batch, product_scaling_factors, n_batch,
        forget_gate_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * aux_input_to_cell_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_cell_weights_ptr, n_cell, n_input,
        quantized_aux_input_ptr_batch, product_scaling_factors, n_batch,
        cell_scratch, /*result_stride=*/1);

    for (int b = 0; b < n_batch; ++b) {
      product_scaling_factors[b] =
          scaling_factors[b] * aux_input_to_output_weights_scale;
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_output_weights_ptr, n_cell, n_input,
        quantized_aux_input_ptr_batch, product_scaling_factors, n_batch,
        output_gate_scratch, /*result_stride=*/1);
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
      tensor_utils::VectorScalarMultiply(cell_to_input_weights_ptr, n_cell,
                                         cell_to_input_weights_scale,
                                         recovered_cell_weights);
      tensor_utils::VectorBatchVectorCwiseProductAccumulate(
          recovered_cell_weights, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole && !is_cell_state_all_zeros) {
    tensor_utils::VectorScalarMultiply(cell_to_forget_weights_ptr, n_cell,
                                       cell_to_forget_weights_scale,
                                       recovered_cell_weights);
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
    tensor_utils::VectorScalarMultiply(cell_to_output_weights_ptr, n_cell,
                                       cell_to_output_weights_scale,
                                       recovered_cell_weights);
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

  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);

  // For each batch: update the projection and output_state. Note that since
  // the output batch rows may not be contiguous (output_batch_leading_dim !=
  // n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == n_output) {
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
  } else {
    if (use_projection_weight) {
      if (use_projection_bias) {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::CopyVector(
              projection_bias_ptr, n_output,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      } else {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::ZeroVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output);
        }
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
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::MatrixBatchVectorMultiplyAccumulate(
              projection_weights_ptr, n_output, n_cell,
              quantized_cell_state_ptr + k * n_cell,
              &product_scaling_factors[k],
              /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim,
              /*result_stride=*/1);
        }
      }
      if (params->proj_clip > 0.0) {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::ClipVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output,
              params->proj_clip,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      }
    } else {
      for (int k = 0; k < n_batch; k++) {
        tensor_utils::CopyVector(
            output_gate_scratch + k * n_output, n_output,
            output_ptr_batch + k * output_batch_leading_dim);
      }
    }
    for (int k = 0; k < n_batch; k++) {
      tensor_utils::CopyVector(output_ptr_batch + k * output_batch_leading_dim,
                               n_output, output_state_ptr + k * n_output);
    }
  }
}
}  // namespace

TfLiteStatus EvalFloat(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights, const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteTensor* scratch_buffer,
    TfLiteTensor* activation_state, TfLiteTensor* cell_state,
    TfLiteTensor* output) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  int max_time, n_batch;
  if (input->dims->size == 3) {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  } else {
    max_time = 1;
    n_batch = input->dims->data[0];
  }
  const int n_input = input->dims->data[input->dims->size - 1];
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = scratch_buffer->data.f;
    forget_gate_scratch = scratch_buffer->data.f + n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer->data.f;
    cell_scratch = scratch_buffer->data.f + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 3 * n_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  const float* input_to_input_weights_ptr =
      (use_cifg) ? nullptr : input_to_input_weights->data.f;
  const float* recurrent_to_input_weights_ptr =
      (use_cifg) ? nullptr : recurrent_to_input_weights->data.f;
  const float* input_gate_bias_ptr =
      (use_cifg) ? nullptr : input_gate_bias->data.f;
  const float* cell_to_input_weights_ptr =
      (use_peephole && !use_cifg) ? cell_to_input_weights->data.f : nullptr;
  const float* cell_to_forget_weights_ptr =
      (use_peephole) ? cell_to_forget_weights->data.f : nullptr;
  const float* cell_to_output_weights_ptr =
      (use_peephole) ? cell_to_output_weights->data.f : nullptr;
  const float* projection_weights_ptr =
      (projection_weights == nullptr) ? nullptr : projection_weights->data.f;
  const float* projection_bias_ptr =
      (projection_bias == nullptr) ? nullptr : projection_bias->data.f;

  float* aux_input_ptr = nullptr;
  float* aux_input_to_input_weights_ptr = nullptr;
  float* aux_input_to_forget_weights_ptr = nullptr;
  float* aux_input_to_cell_weights_ptr = nullptr;
  float* aux_input_to_output_weights_ptr = nullptr;
  if (aux_input_size > 0) {
    if (!use_cifg) {
      aux_input_to_input_weights_ptr = aux_input_to_input_weights->data.f;
    }
    aux_input_to_forget_weights_ptr = aux_input_to_forget_weights->data.f;
    aux_input_to_cell_weights_ptr = aux_input_to_cell_weights->data.f;
    aux_input_to_output_weights_ptr = aux_input_to_output_weights->data.f;
  }

  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    // Loop through the sequence.
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      // If this is the forward_sequence, step forward, otherwise step
      // backwards.
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr_batch = input->data.f + t_rel * input_step;
      if (aux_input) {
        aux_input_ptr = aux_input->data.f + t_rel * input_step;
      }
      float* output_ptr_time =
          output->data.f + t_rel * output_step + output_offset;

      LstmStepWithAuxInput(
          input_ptr_batch, input_to_input_weights_ptr,
          input_to_forget_weights->data.f, input_to_cell_weights->data.f,
          input_to_output_weights->data.f, aux_input_ptr,
          aux_input_to_input_weights_ptr, aux_input_to_forget_weights_ptr,
          aux_input_to_cell_weights_ptr, aux_input_to_output_weights_ptr,
          recurrent_to_input_weights_ptr, recurrent_to_forget_weights->data.f,
          recurrent_to_cell_weights->data.f,
          recurrent_to_output_weights->data.f, cell_to_input_weights_ptr,
          cell_to_forget_weights_ptr, cell_to_output_weights_ptr,
          input_gate_bias_ptr, forget_gate_bias->data.f, cell_bias->data.f,
          output_gate_bias->data.f, projection_weights_ptr, projection_bias_ptr,
          params, n_batch, n_cell, n_input, aux_input_size, n_output,
          output_batch_leading_dim, activation_state->data.f,
          cell_state->data.f, input_gate_scratch, forget_gate_scratch,
          cell_scratch, output_gate_scratch, output_ptr_time);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr = input->data.f + time_offset * input_step;
        if (aux_input) {
          aux_input_ptr = aux_input->data.f + time_offset * input_step;
        }
        float* output_ptr =
            output->data.f + time_offset * output_step + output_offset;

        // Offset the {activation,cell}_state pointers to the right batch.
        float* activation_state_ptr =
            activation_state->data.f + b * output_batch_leading_dim;
        float* cell_state_ptr = cell_state->data.f + b * n_cell;
        // Offset the scratch pointers to the right batch.
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_scratch_ptr = cell_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        LstmStepWithAuxInput(
            input_ptr, input_to_input_weights_ptr,
            input_to_forget_weights->data.f, input_to_cell_weights->data.f,
            input_to_output_weights->data.f, aux_input_ptr,
            aux_input_to_input_weights_ptr, aux_input_to_forget_weights_ptr,
            aux_input_to_cell_weights_ptr, aux_input_to_output_weights_ptr,
            recurrent_to_input_weights_ptr, recurrent_to_forget_weights->data.f,
            recurrent_to_cell_weights->data.f,
            recurrent_to_output_weights->data.f, cell_to_input_weights_ptr,
            cell_to_forget_weights_ptr, cell_to_output_weights_ptr,
            input_gate_bias_ptr, forget_gate_bias->data.f, cell_bias->data.f,
            output_gate_bias->data.f, projection_weights_ptr,
            projection_bias_ptr, params, /*n_batch=*/1, n_cell, n_input,
            aux_input_size, n_output, output_batch_leading_dim,
            activation_state_ptr, cell_state_ptr, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_scratch_ptr, output_gate_scratch_ptr,
            output_ptr);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights, const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteTensor* scratch_buffer,
    TfLiteTensor* scaling_factors, TfLiteTensor* prod_scaling_factors,
    TfLiteTensor* recovered_cell_weights, TfLiteTensor* input_quantized,
    TfLiteTensor* aux_input_quantized, TfLiteTensor* output_state_quantized,
    TfLiteTensor* cell_state_quantized, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;
  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = scratch_buffer->data.f;
    forget_gate_scratch = scratch_buffer->data.f + n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer->data.f;
    cell_scratch = scratch_buffer->data.f + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 3 * n_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  int8_t* input_to_input_weights_ptr = nullptr;
  float input_to_input_weights_scale = 1.0f;
  int8_t* recurrent_to_input_weights_ptr = nullptr;
  float recurrent_to_input_weights_scale = 1.0f;
  float* input_gate_bias_ptr = nullptr;
  if (!use_cifg) {
    input_to_input_weights_ptr =
        reinterpret_cast<int8_t*>(input_to_input_weights->data.uint8);
    recurrent_to_input_weights_ptr =
        reinterpret_cast<int8_t*>(recurrent_to_input_weights->data.uint8);
    input_gate_bias_ptr = input_gate_bias->data.f;
    input_to_input_weights_scale = input_to_input_weights->params.scale;
    recurrent_to_input_weights_scale = recurrent_to_input_weights->params.scale;
  }

  int8_t* cell_to_input_weights_ptr = nullptr;
  int8_t* cell_to_forget_weights_ptr = nullptr;
  int8_t* cell_to_output_weights_ptr = nullptr;
  float cell_to_input_weights_scale = 1.0f;
  float cell_to_forget_weights_scale = 1.0f;
  float cell_to_output_weights_scale = 1.0f;
  if (use_peephole) {
    if (!use_cifg) {
      cell_to_input_weights_ptr =
          reinterpret_cast<int8_t*>(cell_to_input_weights->data.uint8);
      cell_to_input_weights_scale = cell_to_input_weights->params.scale;
    }
    cell_to_forget_weights_ptr =
        reinterpret_cast<int8_t*>(cell_to_forget_weights->data.uint8);
    cell_to_output_weights_ptr =
        reinterpret_cast<int8_t*>(cell_to_output_weights->data.uint8);
    cell_to_forget_weights_scale = cell_to_forget_weights->params.scale;
    cell_to_output_weights_scale = cell_to_output_weights->params.scale;
  }

  const int8_t* projection_weights_ptr =
      (projection_weights == nullptr)
          ? nullptr
          : reinterpret_cast<int8_t*>(projection_weights->data.uint8);
  const float projection_weights_scale =
      (projection_weights == nullptr) ? 1.0f : projection_weights->params.scale;
  const float* projection_bias_ptr =
      (projection_bias == nullptr) ? nullptr : projection_bias->data.f;

  // Required tensors, pointers are non-null.
  const int8_t* input_to_forget_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_forget_weights->data.uint8);
  const float input_to_forget_weights_scale =
      input_to_forget_weights->params.scale;
  const int8_t* input_to_cell_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_cell_weights->data.uint8);
  const float input_to_cell_weights_scale = input_to_cell_weights->params.scale;
  const int8_t* input_to_output_weights_ptr =
      reinterpret_cast<int8_t*>(input_to_output_weights->data.uint8);
  const float input_to_output_weights_scale =
      input_to_output_weights->params.scale;
  const int8_t* recurrent_to_forget_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_forget_weights->data.uint8);
  const float recurrent_to_forget_weights_scale =
      recurrent_to_forget_weights->params.scale;
  const int8_t* recurrent_to_cell_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_cell_weights->data.uint8);
  const float recurrent_to_cell_weights_scale =
      recurrent_to_cell_weights->params.scale;
  const int8_t* recurrent_to_output_weights_ptr =
      reinterpret_cast<int8_t*>(recurrent_to_output_weights->data.uint8);
  const float recurrent_to_output_weights_scale =
      recurrent_to_output_weights->params.scale;
  const float* forget_gate_bias_ptr = forget_gate_bias->data.f;
  const float* cell_bias_ptr = cell_bias->data.f;
  const float* output_gate_bias_ptr = output_gate_bias->data.f;

  // Temporary storage for quantized values and scaling factors.
  int8_t* quantized_input_ptr =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);
  int8_t* quantized_aux_input_ptr =
      (aux_input_quantized == nullptr)
          ? nullptr
          : reinterpret_cast<int8_t*>(aux_input_quantized->data.uint8);
  int8_t* quantized_output_state_ptr =
      reinterpret_cast<int8_t*>(output_state_quantized->data.uint8);
  int8_t* quantized_cell_state_ptr =
      reinterpret_cast<int8_t*>(cell_state_quantized->data.uint8);
  float* scaling_factors_ptr = scaling_factors->data.f;
  float* prod_scaling_factors_ptr = prod_scaling_factors->data.f;
  float* recovered_cell_weights_ptr = recovered_cell_weights->data.f;

  // Auxiliary input and weights.
  float* aux_input_ptr = nullptr;
  int8_t* aux_input_to_input_weights_ptr = nullptr;
  int8_t* aux_input_to_forget_weights_ptr = nullptr;
  int8_t* aux_input_to_cell_weights_ptr = nullptr;
  int8_t* aux_input_to_output_weights_ptr = nullptr;
  float aux_input_to_input_weights_scale = 0.0f;
  float aux_input_to_forget_weights_scale = 0.0f;
  float aux_input_to_cell_weights_scale = 0.0f;
  float aux_input_to_output_weights_scale = 0.0f;
  if (aux_input_size > 0) {
    if (!use_cifg) {
      aux_input_to_input_weights_ptr =
          reinterpret_cast<int8_t*>(aux_input_to_input_weights->data.uint8);
    }
    aux_input_to_forget_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_forget_weights->data.uint8);
    aux_input_to_cell_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_cell_weights->data.uint8);
    aux_input_to_output_weights_ptr =
        reinterpret_cast<int8_t*>(aux_input_to_output_weights->data.uint8);
    if (!use_cifg) {
      aux_input_to_input_weights_scale =
          aux_input_to_input_weights->params.scale;
    }
    aux_input_to_forget_weights_scale =
        aux_input_to_forget_weights->params.scale;
    aux_input_to_cell_weights_scale = aux_input_to_cell_weights->params.scale;
    aux_input_to_output_weights_scale =
        aux_input_to_output_weights->params.scale;
  }

  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    // Feed the sequence into the LSTM step-by-step.
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      // If this is the forward_sequence, step forward, otherwise step
      // backwards.
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr_batch = input->data.f + t_rel * input_step;
      if (aux_input) {
        aux_input_ptr = aux_input->data.f + t_rel * input_step;
      }
      float* output_ptr_batch =
          output->data.f + t_rel * output_step + output_offset;

      LstmStepWithAuxInput(
          input_ptr_batch, input_to_input_weights_ptr,
          input_to_input_weights_scale, input_to_forget_weights_ptr,
          input_to_forget_weights_scale, input_to_cell_weights_ptr,
          input_to_cell_weights_scale, input_to_output_weights_ptr,
          input_to_output_weights_scale, aux_input_ptr,
          aux_input_to_input_weights_ptr, aux_input_to_input_weights_scale,
          aux_input_to_forget_weights_ptr, aux_input_to_forget_weights_scale,
          aux_input_to_cell_weights_ptr, aux_input_to_cell_weights_scale,
          aux_input_to_output_weights_ptr, aux_input_to_output_weights_scale,
          recurrent_to_input_weights_ptr, recurrent_to_input_weights_scale,
          recurrent_to_forget_weights_ptr, recurrent_to_forget_weights_scale,
          recurrent_to_cell_weights_ptr, recurrent_to_cell_weights_scale,
          recurrent_to_output_weights_ptr, recurrent_to_output_weights_scale,
          cell_to_input_weights_ptr, cell_to_input_weights_scale,
          cell_to_forget_weights_ptr, cell_to_forget_weights_scale,
          cell_to_output_weights_ptr, cell_to_output_weights_scale,
          input_gate_bias_ptr, forget_gate_bias_ptr, cell_bias_ptr,
          output_gate_bias_ptr, projection_weights_ptr,
          projection_weights_scale, projection_bias_ptr, params, n_batch,
          n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
          input_gate_scratch, forget_gate_scratch, cell_scratch,
          output_gate_scratch, scaling_factors_ptr, prod_scaling_factors_ptr,
          recovered_cell_weights_ptr, quantized_input_ptr,
          quantized_aux_input_ptr, quantized_output_state_ptr,
          quantized_cell_state_ptr, output_state->data.f, cell_state->data.f,
          output_ptr_batch);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr = input->data.f + time_offset * input_step;
        if (aux_input) {
          aux_input_ptr = aux_input->data.f + time_offset * input_step;
        }
        float* output_ptr =
            output->data.f + time_offset * output_step + output_offset;

        // Offset the {output,cell}_state pointers to the right batch.
        float* output_state_ptr =
            output_state->data.f + b * output_batch_leading_dim;
        float* cell_state_ptr = cell_state->data.f + b * n_cell;
        // Offset the scratch pointers to the right batch.
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_scratch_ptr = cell_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        LstmStepWithAuxInput(
            input_ptr, input_to_input_weights_ptr, input_to_input_weights_scale,
            input_to_forget_weights_ptr, input_to_forget_weights_scale,
            input_to_cell_weights_ptr, input_to_cell_weights_scale,
            input_to_output_weights_ptr, input_to_output_weights_scale,
            aux_input_ptr, aux_input_to_input_weights_ptr,
            aux_input_to_input_weights_scale, aux_input_to_forget_weights_ptr,
            aux_input_to_forget_weights_scale, aux_input_to_cell_weights_ptr,
            aux_input_to_cell_weights_scale, aux_input_to_output_weights_ptr,
            aux_input_to_output_weights_scale, recurrent_to_input_weights_ptr,
            recurrent_to_input_weights_scale, recurrent_to_forget_weights_ptr,
            recurrent_to_forget_weights_scale, recurrent_to_cell_weights_ptr,
            recurrent_to_cell_weights_scale, recurrent_to_output_weights_ptr,
            recurrent_to_output_weights_scale, cell_to_input_weights_ptr,
            cell_to_input_weights_scale, cell_to_forget_weights_ptr,
            cell_to_forget_weights_scale, cell_to_output_weights_ptr,
            cell_to_output_weights_scale, input_gate_bias_ptr,
            forget_gate_bias_ptr, cell_bias_ptr, output_gate_bias_ptr,
            projection_weights_ptr, projection_weights_scale,
            projection_bias_ptr, params,
            /*n_batch=*/1, n_cell, n_input, aux_input_size, n_output,
            output_batch_leading_dim, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_scratch_ptr, output_gate_scratch_ptr,
            scaling_factors_ptr, prod_scaling_factors_ptr,
            recovered_cell_weights_ptr, quantized_input_ptr,
            quantized_aux_input_ptr, quantized_output_state_ptr,
            quantized_cell_state_ptr, output_state_ptr, cell_state_ptr,
            output_ptr);
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace lstm_eval
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
