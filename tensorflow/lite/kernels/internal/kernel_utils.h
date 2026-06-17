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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_

#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace tflite {
namespace kernel_utils {

// Performs an RNN batch inference step for inputs specified by input_ptr_batch.
// The RNN cell is specified by the pointers to its input and recurrent weights,
// and biases, along with the input size, number of units, activation.
//
// The pointers to the hidden state and the output are updated as a result.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many outputs and hidden states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to num_units. It is usually
// not when we want to store the RNN output into a slice of the output tensor,
// e.g. for bidirectional RNNs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch);

// Same as above but includes an auxiliary input with the corresponding weights.
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* aux_input_ptr_batch,
                  const float* aux_input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int aux_input_size, int num_units,
                  int batch_size, int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch);

// Performs a quantized RNN batch inference step. Same as above, but for
// quantization purposes, we also pass in quantized_hidden_state_ptr_batch and
// quantized_input_ptr_batch pointers for temporary storage of the quantized
// values of hidden_state_ptr_batch and input_ptr_batch, respectively.
// These temporary storages are expected to be preallocated to the same size as
// the respective pointers.
// An additional preallocated temporary storage 'scaling_factors' (of size
// batch_size) is used to store the scaling factors of the quantization (used
// for recovery).
// {input,recurrent}_weights_scale params are used for dequantization/recovery.
void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const int8_t* recurrent_weights_ptr,
    float recurrent_weights_scale, const float* bias_ptr, int input_size,
    int num_units, int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums);

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
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums);

}  // namespace kernel_utils
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
