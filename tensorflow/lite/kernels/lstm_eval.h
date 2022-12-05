/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_LSTM_EVAL_H_
#define TENSORFLOW_LITE_KERNELS_LSTM_EVAL_H_

#include <cstdint>
#include <memory>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace lstm_eval {

// Pamameters for integer LSTM.
// Consider split this into two Integer Parameters if more fields are added.
struct IntegerLstmParameter {
  int32_t effective_input_to_input_scale_a;
  int32_t effective_input_to_input_scale_b;
  int32_t effective_recurrent_to_input_scale_a;
  int32_t effective_recurrent_to_input_scale_b;
  int32_t effective_cell_to_input_scale_a;
  int32_t effective_cell_to_input_scale_b;
  int32_t effective_input_to_forget_scale_a;
  int32_t effective_input_to_forget_scale_b;
  int32_t effective_recurrent_to_forget_scale_a;
  int32_t effective_recurrent_to_forget_scale_b;
  int32_t effective_cell_to_forget_scale_a;
  int32_t effective_cell_to_forget_scale_b;
  int32_t effective_input_to_cell_scale_a;
  int32_t effective_input_to_cell_scale_b;
  int32_t effective_recurrent_to_cell_scale_a;
  int32_t effective_recurrent_to_cell_scale_b;
  int32_t effective_input_to_output_scale_a;
  int32_t effective_input_to_output_scale_b;
  int32_t effective_recurrent_to_output_scale_a;
  int32_t effective_recurrent_to_output_scale_b;
  int32_t effective_cell_to_output_scale_a;
  int32_t effective_cell_to_output_scale_b;
  int32_t effective_proj_scale_a;
  int32_t effective_proj_scale_b;
  int32_t effective_hidden_scale_a;
  int32_t effective_hidden_scale_b;
  int32_t layer_norm_input_scale_a;
  int32_t layer_norm_input_scale_b;
  int32_t layer_norm_forget_scale_a;
  int32_t layer_norm_forget_scale_b;
  int32_t layer_norm_cell_scale_a;
  int32_t layer_norm_cell_scale_b;
  int32_t layer_norm_output_scale_a;
  int32_t layer_norm_output_scale_b;
  // Quantized clip value for cell and projection. Zero value means no clipping.
  int16_t quantized_cell_clip;
  int8_t quantized_proj_clip;
  int32_t hidden_zp;
  int32_t cell_scale;

  int32_t input_variance_guard;
  int32_t forget_variance_guard;
  int32_t cell_variance_guard;
  int32_t output_variance_guard;

  // Pre-calculate bias + zero_point * weight.
  // Unabled to use temporary tensors since those are used in Prepare() and
  // scratch buffer is only allocated after Preapre().
  std::unique_ptr<int32_t[]> input_to_forget_effective_bias;
  std::unique_ptr<int32_t[]> recurrent_to_forget_effective_bias;
  std::unique_ptr<int32_t[]> input_to_cell_effective_bias;
  std::unique_ptr<int32_t[]> recurrent_to_cell_effective_bias;
  std::unique_ptr<int32_t[]> input_to_output_effective_bias;
  std::unique_ptr<int32_t[]> recurrent_to_output_effective_bias;
  std::unique_ptr<int32_t[]> input_to_input_effective_bias;
  std::unique_ptr<int32_t[]> recurrent_to_input_effective_bias;
  std::unique_ptr<int32_t[]> projection_effective_bias;

  // Scale and zero point for intermediate tensors.
  // Used only in the 8x8_8 case.
  int32_t intermediate_scale_a[8];
  int32_t intermediate_scale_b[8];
  int32_t intermediate_zp[12];
};

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
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteTensor* scratch_buffer, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output, CpuBackendContext* context);

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_input_weights_ledger,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_forget_weights_ledger,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_cell_weights_ledger,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* input_to_output_weights_ledger,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_input_weights_ledger,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_forget_weights_ledger,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_cell_weights_ledger,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* recurrent_to_output_weights_ledger,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights,
    const TfLiteTensor* projection_weights_ledger,
    const TfLiteTensor* projection_bias, const TfLiteLSTMParams* params,
    bool forward_sequence, bool time_major, int output_offset,
    TfLiteTensor* scratch_buffer, TfLiteTensor* input_sf,
    TfLiteTensor* aux_input_sf, TfLiteTensor* output_state_sf,
    TfLiteTensor* prod_scaling_factors, TfLiteTensor* recovered_cell_weights,
    TfLiteTensor* input_quantized, TfLiteTensor* aux_input_quantized,
    TfLiteTensor* output_state_quantized, TfLiteTensor* cell_state_quantized,
    TfLiteTensor* output_state, TfLiteTensor* cell_state,
    TfLiteTensor* output_scratch_buffer, TfLiteTensor* output,
    TfLiteTensor* input_zp, TfLiteTensor* aux_input_zp,
    TfLiteTensor* output_state_zp, TfLiteTensor* row_sums, int row_sums_size,
    bool* compute_row_sums, CpuBackendContext* context);

TfLiteStatus EvalInteger8x8_16(
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
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* output_state, TfLiteTensor* cell_state, TfLiteTensor* output,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    CpuBackendContext* context);

TfLiteStatus EvalInteger8x8_8(
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
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    TfLiteTensor* scratch6, TfLiteTensor* scratch7);

}  // namespace lstm_eval
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_LSTM_EVAL_H_
