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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_

#include "tensorflow/contrib/lite/builtin_op_data.h"

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
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
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
void RnnBatchStep(const float* input_ptr_batch, const int8_t* input_weights_ptr,
                  float input_weights_scale,
                  const int8_t* recurrent_weights_ptr,
                  float recurrent_weights_scale, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  TfLiteFusedActivation activation,
                  int8_t* quantized_input_ptr_batch,
                  int8_t* quantized_hidden_state_ptr_batch,
                  float* scaling_factors, float* hidden_state_ptr_batch,
                  float* output_ptr_batch);

// Performs an LSTM batch inference step for input specified by input_ptr_batch.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_output: the output size.
//
// The pointers to the cell and output state and the output are updated. Unless
// projection is specified output and output state contain the same data.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many cell and output states.
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
    float* output_ptr_batch);

}  // namespace kernel_utils
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
