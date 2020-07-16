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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_eval.h"
#include "tensorflow/lite/kernels/lstm_shared.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace lstm {

struct OpData {
  // Which kernel type to use. Full kernel (24 inputs) or basic kernel (5
  // inputs).
  // Please note the 20-input full kernel is deprecated and only kept
  // here for backward compatibility.
  TfLiteLSTMKernelType kernel_type;

  // If the lstm is layer norm.
  bool use_layer_norm;

  // These fields are only used by full kernel.
  int scratch_tensor_index;
  lstm_eval::IntegerLstmParameter integer_lstm_param;
  bool compute_row_sums;
};

namespace full {
namespace {

// Named temporary tensors.
enum HybridTemporaryTensor {
  kScratchBuffer = 0,
  kInputQuantized = 1,
  kOutputStateQuantized = 2,
  kCellStateQuantized = 3,
  kInputScalingFactors = 4,
  kOutputStateScalingFactors = 5,
  kProductScalingFactors = 6,
  kRecoveredCellWeights = 7,
  kAccumScratch = 8,
  kInputZeroPoints = 9,
  kOutputStateZeroPoints = 10,
  kRowSums = 11,
  kNumHybridTemporaryTensors = 12,
};

TfLiteStatus PopulateQuantizedLstmParams8x8_16(
    TfLiteContext* context, TfLiteNode* node,
    lstm_eval::IntegerLstmParameter* integer_lstm_param) {
  // Calculate quantized clip for projection and cell.
  const auto* params = static_cast<TfLiteLSTMParams*>(node->builtin_data);
  const float cell_clip = params->cell_clip;
  const float proj_clip = params->proj_clip;

  const TfLiteTensor* cell_state =
      GetVariableInput(context, node, kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);
  const TfLiteTensor* output_tensor = GetOutput(context, node, kOutputTensor);

  auto* cell_state_params =
      static_cast<TfLiteAffineQuantization*>(cell_state->quantization.params);
  auto* proj_params = static_cast<TfLiteAffineQuantization*>(
      output_tensor->quantization.params);
  if (cell_clip > 0.0) {
    integer_lstm_param->quantized_cell_clip = static_cast<int16_t>(std::min(
        std::max(cell_clip / cell_state_params->scale->data[0], -32768.0f),
        32767.0f));
  } else {
    integer_lstm_param->quantized_cell_clip = 0;
  }
  if (proj_clip > 0.0) {
    integer_lstm_param->quantized_proj_clip = static_cast<int8_t>(std::min(
        std::max(proj_clip / proj_params->scale->data[0], -128.0f), 127.0f));
  } else {
    integer_lstm_param->quantized_proj_clip = 0;
  }

  // Calculate effective scales.
  OpData* op_data = static_cast<OpData*>(node->user_data);
  const bool use_layer_norm = op_data->use_layer_norm;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kInputLayerNormCoefficientsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* cell_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kCellLayerNormCoefficientsTensor);
  const TfLiteTensor* output_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kOutputLayerNormCoefficientsTensor);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);
  const bool use_projection = (projection_weights != nullptr);

  // Get intermediate scales and zero points.
  std::vector<float> intermediate_scale;
  std::vector<int32> intermediate_zp;
  for (int i = 0; i < 4; ++i) {
    if (use_layer_norm) {
      const TfLiteTensor* intermediate = GetIntermediates(context, node, i);
      auto* params = static_cast<TfLiteAffineQuantization*>(
          intermediate->quantization.params);
      intermediate_scale.push_back(params->scale->data[0]);
      intermediate_zp.push_back(params->zero_point->data[0]);
    } else {
      // Q3.12 for activation functions.
      intermediate_scale.push_back(std::pow(2, -12));
      intermediate_zp.push_back(0);
    }
  }
  // In the absense of projection, hidden becomes otuput and this intermediate
  // is ignored.
  const TfLiteTensor* hidden = GetIntermediates(context, node, 4);
  auto* hidden_params =
      static_cast<TfLiteAffineQuantization*>(hidden->quantization.params);
  intermediate_scale.push_back(hidden_params->scale->data[0]);
  intermediate_zp.push_back(hidden_params->zero_point->data[0]);

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float cell_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float cell_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float cell_to_output_weight_scale = default_scale;
  float projection_weight_scale = default_scale;
  float layer_norm_input_scale = default_scale;
  float layer_norm_forget_scale = default_scale;
  float layer_norm_cell_scale = default_scale;
  float layer_norm_output_scale = default_scale;
  float output_state_scale = default_scale;
  int cell_scale = 1;

  // Effective scales.
  float effective_input_to_input_scale = default_scale;
  float effective_recurrent_to_input_scale = default_scale;
  float effective_cell_to_input_scale = default_scale;
  float effective_input_to_forget_scale = default_scale;
  float effective_recurrent_to_forget_scale = default_scale;
  float effective_cell_to_forget_scale = default_scale;
  float effective_input_to_cell_scale = default_scale;
  float effective_recurrent_to_cell_scale = default_scale;
  float effective_input_to_output_scale = default_scale;
  float effective_recurrent_to_output_scale = default_scale;
  float effective_cell_to_output_scale = default_scale;
  float effective_proj_scale = default_scale;
  float effective_hidden_scale = default_scale;

  // Populate scales.
  if (!use_cifg) {
    input_to_input_weight_scale = input_to_input_weights->params.scale;
    recurrent_to_input_weight_scale = recurrent_to_input_weights->params.scale;
  }

  if (use_peephole) {
    if (!use_cifg) {
      cell_to_input_weight_scale = cell_to_input_weights->params.scale;
    }
    cell_to_forget_weight_scale = cell_to_forget_weights->params.scale;
    cell_to_output_weight_scale = cell_to_output_weights->params.scale;
  }

  if (use_layer_norm) {
    if (!use_cifg) {
      layer_norm_input_scale = input_layer_norm_coefficients->params.scale;
    }
    layer_norm_forget_scale = forget_layer_norm_coefficients->params.scale;
    layer_norm_cell_scale = cell_layer_norm_coefficients->params.scale;
    layer_norm_output_scale = output_layer_norm_coefficients->params.scale;
  }

  if (use_projection) {
    projection_weight_scale = projection_weights->params.scale;
  }
  output_state_scale = output_state->params.scale;

  input_to_forget_weight_scale = input_to_forget_weights->params.scale;
  input_to_cell_weight_scale = input_to_cell_weights->params.scale;
  input_to_output_weight_scale = input_to_output_weights->params.scale;
  recurrent_to_forget_weight_scale = recurrent_to_forget_weights->params.scale;
  recurrent_to_cell_weight_scale = recurrent_to_cell_weights->params.scale;
  recurrent_to_output_weight_scale = recurrent_to_output_weights->params.scale;

  // Check cell state (already used above)
  TF_LITE_ENSURE(context, CheckedLog2(cell_state->params.scale, &cell_scale));
  TF_LITE_ENSURE(context, cell_scale <= -9);
  integer_lstm_param->cell_scale = cell_scale;
  input_scale = input->params.scale;

  // Calculate effective scales.
  if (!use_cifg) {
    effective_input_to_input_scale =
        input_to_input_weight_scale * input_scale / intermediate_scale[0];
    effective_recurrent_to_input_scale = recurrent_to_input_weight_scale *
                                         output_state_scale /
                                         intermediate_scale[0];
  }
  effective_input_to_forget_scale =
      input_to_forget_weight_scale * input_scale / intermediate_scale[1];
  effective_recurrent_to_forget_scale = recurrent_to_forget_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[1];

  effective_input_to_cell_scale =
      input_to_cell_weight_scale * input_scale / intermediate_scale[2];
  effective_recurrent_to_cell_scale = recurrent_to_cell_weight_scale *
                                      output_state_scale /
                                      intermediate_scale[2];

  effective_input_to_output_scale =
      input_to_output_weight_scale * input_scale / intermediate_scale[3];
  effective_recurrent_to_output_scale = recurrent_to_output_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[3];

  effective_hidden_scale =
      std::pow(2, -15) / intermediate_scale[4] * std::pow(2, -15);

  effective_proj_scale =
      projection_weight_scale * intermediate_scale[4] / output_state_scale;

  if (use_peephole) {
    if (!use_cifg) {
      effective_cell_to_input_scale = std::pow(2, cell_scale) *  // NOLINT
                                      cell_to_input_weight_scale /
                                      intermediate_scale[0];
    }
    effective_cell_to_forget_scale = std::pow(2, cell_scale) *  // NOLINT
                                     cell_to_forget_weight_scale /
                                     intermediate_scale[1];
    effective_cell_to_output_scale = std::pow(2, cell_scale) *  // NOLINT
                                     cell_to_output_weight_scale /
                                     intermediate_scale[3];
  }

  // Decompose scales.
  QuantizeMultiplier(effective_input_to_input_scale,
                     &integer_lstm_param->effective_input_to_input_scale_a,
                     &integer_lstm_param->effective_input_to_input_scale_b);
  QuantizeMultiplier(effective_recurrent_to_input_scale,
                     &integer_lstm_param->effective_recurrent_to_input_scale_a,
                     &integer_lstm_param->effective_recurrent_to_input_scale_b);
  QuantizeMultiplier(effective_cell_to_input_scale,
                     &integer_lstm_param->effective_cell_to_input_scale_a,
                     &integer_lstm_param->effective_cell_to_input_scale_b);
  QuantizeMultiplier(effective_input_to_forget_scale,
                     &integer_lstm_param->effective_input_to_forget_scale_a,
                     &integer_lstm_param->effective_input_to_forget_scale_b);
  QuantizeMultiplier(
      effective_recurrent_to_forget_scale,
      &integer_lstm_param->effective_recurrent_to_forget_scale_a,
      &integer_lstm_param->effective_recurrent_to_forget_scale_b);
  QuantizeMultiplier(effective_cell_to_forget_scale,
                     &integer_lstm_param->effective_cell_to_forget_scale_a,
                     &integer_lstm_param->effective_cell_to_forget_scale_b);
  QuantizeMultiplier(effective_input_to_cell_scale,
                     &integer_lstm_param->effective_input_to_cell_scale_a,
                     &integer_lstm_param->effective_input_to_cell_scale_b);
  QuantizeMultiplier(effective_recurrent_to_cell_scale,
                     &integer_lstm_param->effective_recurrent_to_cell_scale_a,
                     &integer_lstm_param->effective_recurrent_to_cell_scale_b);
  QuantizeMultiplier(effective_input_to_output_scale,
                     &integer_lstm_param->effective_input_to_output_scale_a,
                     &integer_lstm_param->effective_input_to_output_scale_b);
  QuantizeMultiplier(
      effective_recurrent_to_output_scale,
      &integer_lstm_param->effective_recurrent_to_output_scale_a,
      &integer_lstm_param->effective_recurrent_to_output_scale_b);
  QuantizeMultiplier(effective_cell_to_output_scale,
                     &integer_lstm_param->effective_cell_to_output_scale_a,
                     &integer_lstm_param->effective_cell_to_output_scale_b);
  QuantizeMultiplier(effective_proj_scale,
                     &integer_lstm_param->effective_proj_scale_a,
                     &integer_lstm_param->effective_proj_scale_b);
  QuantizeMultiplier(effective_hidden_scale,
                     &integer_lstm_param->effective_hidden_scale_a,
                     &integer_lstm_param->effective_hidden_scale_b);
  QuantizeMultiplier(layer_norm_input_scale,
                     &integer_lstm_param->layer_norm_input_scale_a,
                     &integer_lstm_param->layer_norm_input_scale_b);
  QuantizeMultiplier(layer_norm_forget_scale,
                     &integer_lstm_param->layer_norm_forget_scale_a,
                     &integer_lstm_param->layer_norm_forget_scale_b);
  QuantizeMultiplier(layer_norm_cell_scale,
                     &integer_lstm_param->layer_norm_cell_scale_a,
                     &integer_lstm_param->layer_norm_cell_scale_b);
  QuantizeMultiplier(layer_norm_output_scale,
                     &integer_lstm_param->layer_norm_output_scale_a,
                     &integer_lstm_param->layer_norm_output_scale_b);

  integer_lstm_param->hidden_zp = intermediate_zp[4];

  // 10000 is used to make sure the kernel logic does not overflow.
  if (!use_cifg) {
    integer_lstm_param->input_variance_guard =
        std::max(1, static_cast<int32_t>(10000 * layer_norm_input_scale));
  }
  integer_lstm_param->forget_variance_guard =
      std::max(1, static_cast<int32_t>(10000 * layer_norm_forget_scale));
  integer_lstm_param->cell_variance_guard =
      std::max(1, static_cast<int32_t>(10000 * layer_norm_cell_scale));
  integer_lstm_param->output_variance_guard =
      std::max(1, static_cast<int32_t>(10000 * layer_norm_output_scale));

  return kTfLiteOk;
}

TfLiteStatus PopulateQuantizedLstmParams8x8_8(
    TfLiteContext* context, TfLiteNode* node,
    lstm_eval::IntegerLstmParameter* integer_lstm_param) {
  // Get all tensors.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kInputLayerNormCoefficientsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* cell_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kCellLayerNormCoefficientsTensor);
  const TfLiteTensor* output_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kOutputLayerNormCoefficientsTensor);

  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  const TfLiteTensor* cell_gate_bias =
      GetInput(context, node, kCellGateBiasTensor);
  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state = GetVariableInput(context, node, kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);
  const bool is_layer_norm_lstm = (forget_layer_norm_coefficients != nullptr);
  const bool use_projection = (projection_weights != nullptr);

  // Weights and states.
  int8_t* input_to_input_weight_ptr = nullptr;
  int8_t* recurrent_to_input_weight_ptr = nullptr;
  int8_t* cell_to_input_weight_ptr = nullptr;
  int8_t* input_to_forget_weight_ptr = nullptr;
  int8_t* recurrent_to_forget_weight_ptr = nullptr;
  int8_t* cell_to_forget_weight_ptr = nullptr;
  int8_t* input_to_cell_weight_ptr = nullptr;
  int8_t* recurrent_to_cell_weight_ptr = nullptr;
  int8_t* input_to_output_weight_ptr = nullptr;
  int8_t* recurrent_to_output_weight_ptr = nullptr;
  int8_t* cell_to_output_weight_ptr = nullptr;
  int8_t* projection_weight_ptr = nullptr;
  int16_t* layer_norm_input_weight_ptr = nullptr;
  int16_t* layer_norm_forget_weight_ptr = nullptr;
  int16_t* layer_norm_cell_weight_ptr = nullptr;
  int16_t* layer_norm_output_weight_ptr = nullptr;
  int32_t* input_gate_bias_ptr = nullptr;
  int32_t* forget_gate_bias_ptr = nullptr;
  int32_t* cell_gate_bias_ptr = nullptr;
  int32_t* output_gate_bias_ptr = nullptr;
  int32_t* projection_bias_ptr = nullptr;
  int16_t* cell_ptr = nullptr;
  int8_t* output_state_ptr = nullptr;

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float cell_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float cell_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float cell_to_output_weight_scale = default_scale;
  float projection_weight_scale = default_scale;
  float layer_norm_input_scale = default_scale;
  float layer_norm_forget_scale = default_scale;
  float layer_norm_cell_scale = default_scale;
  float layer_norm_output_scale = default_scale;
  float output_state_scale = default_scale;

  // Effective scales.
  float effective_input_to_input_scale = default_scale;
  float effective_recurrent_to_input_scale = default_scale;
  float effective_cell_to_input_scale = default_scale;
  float effective_input_to_forget_scale = default_scale;
  float effective_recurrent_to_forget_scale = default_scale;
  float effective_cell_to_forget_scale = default_scale;
  float effective_input_to_cell_scale = default_scale;
  float effective_recurrent_to_cell_scale = default_scale;
  float effective_input_to_output_scale = default_scale;
  float effective_recurrent_to_output_scale = default_scale;
  float effective_cell_to_output_scale = default_scale;
  float effective_proj_scale = default_scale;

  // Zero points
  int input_zp = 0;
  int output_state_zp = 0;

  // Populate all the values.
  if (!use_cifg) {
    input_to_input_weight_ptr = input_to_input_weights->data.int8;
    recurrent_to_input_weight_ptr = recurrent_to_input_weights->data.int8;
    input_gate_bias_ptr = input_gate_bias->data.i32;
    input_to_input_weight_scale = input_to_input_weights->params.scale;
    recurrent_to_input_weight_scale = recurrent_to_input_weights->params.scale;
  }

  if (use_peephole) {
    if (!use_cifg) {
      cell_to_input_weight_ptr = cell_to_input_weights->data.int8;
      cell_to_input_weight_scale = cell_to_input_weights->params.scale;
    }
    cell_to_forget_weight_ptr = cell_to_forget_weights->data.int8;
    cell_to_output_weight_ptr = cell_to_output_weights->data.int8;
    cell_to_forget_weight_scale = cell_to_forget_weights->params.scale;
    cell_to_output_weight_scale = cell_to_output_weights->params.scale;
  }

  if (is_layer_norm_lstm) {
    if (!use_cifg) {
      layer_norm_input_weight_ptr = input_layer_norm_coefficients->data.i16;
      layer_norm_input_scale = input_layer_norm_coefficients->params.scale;
    }
    layer_norm_forget_weight_ptr = forget_layer_norm_coefficients->data.i16;
    layer_norm_forget_scale = forget_layer_norm_coefficients->params.scale;
    layer_norm_cell_weight_ptr = cell_layer_norm_coefficients->data.i16;
    layer_norm_cell_scale = cell_layer_norm_coefficients->params.scale;
    layer_norm_output_weight_ptr = output_layer_norm_coefficients->data.i16;
    layer_norm_output_scale = output_layer_norm_coefficients->params.scale;
  }

  if (use_projection) {
    projection_weight_ptr = projection_weights->data.int8;
    projection_weight_scale = projection_weights->params.scale;
    if (projection_bias) {
      projection_bias_ptr = projection_bias->data.i32;
    }
  }
  output_state_scale = output_state->params.scale;

  input_to_forget_weight_ptr = input_to_forget_weights->data.int8;
  input_to_forget_weight_scale = input_to_forget_weights->params.scale;
  input_to_cell_weight_ptr = input_to_cell_weights->data.int8;
  input_to_cell_weight_scale = input_to_cell_weights->params.scale;
  input_to_output_weight_ptr = input_to_output_weights->data.int8;
  input_to_output_weight_scale = input_to_output_weights->params.scale;
  recurrent_to_forget_weight_ptr = recurrent_to_forget_weights->data.int8;
  recurrent_to_forget_weight_scale = recurrent_to_forget_weights->params.scale;
  recurrent_to_cell_weight_ptr = recurrent_to_cell_weights->data.int8;
  recurrent_to_cell_weight_scale = recurrent_to_cell_weights->params.scale;
  recurrent_to_output_weight_ptr = recurrent_to_output_weights->data.int8;
  recurrent_to_output_weight_scale = recurrent_to_output_weights->params.scale;
  forget_gate_bias_ptr = forget_gate_bias->data.i32;
  cell_gate_bias_ptr = cell_gate_bias->data.i32;
  output_gate_bias_ptr = output_gate_bias->data.i32;
  output_state_ptr = output_state->data.int8;
  cell_ptr = cell_state->data.i16;
  input_scale = input->params.scale;
  input_zp = input->params.zero_point;
  output_state_zp = output_state->params.zero_point;

  std::vector<float> intermediate_scale;
  for (int i = 0; i < 12; ++i) {
    TfLiteTensor* intermediate =
        &context->tensors[node->intermediates->data[i]];
    auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
        intermediate->quantization.params);
    intermediate_scale.push_back(params->scale->data[0]);
    integer_lstm_param->intermediate_zp[i] = params->zero_point->data[0];
  }

  // Calculate effective scales.
  if (!use_cifg) {
    effective_input_to_input_scale =
        input_to_input_weight_scale * input_scale / intermediate_scale[1];
    effective_recurrent_to_input_scale = recurrent_to_input_weight_scale *
                                         output_state_scale /
                                         intermediate_scale[2];
  }
  effective_input_to_forget_scale =
      input_to_forget_weight_scale * input_scale / intermediate_scale[4];
  effective_recurrent_to_forget_scale = recurrent_to_forget_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[5];

  effective_input_to_cell_scale =
      input_to_cell_weight_scale * input_scale / intermediate_scale[7];
  effective_recurrent_to_cell_scale = recurrent_to_cell_weight_scale *
                                      output_state_scale /
                                      intermediate_scale[8];

  effective_input_to_output_scale =
      input_to_output_weight_scale * input_scale / intermediate_scale[10];
  effective_recurrent_to_output_scale = recurrent_to_output_weight_scale *
                                        output_state_scale /
                                        intermediate_scale[11];
  effective_proj_scale =
      projection_weight_scale * std::pow(2, -15) / output_state_scale;

  if (use_peephole) {
    if (!use_cifg) {
      effective_cell_to_input_scale =
          std::pow(2, -15) * cell_to_input_weight_scale / intermediate_scale[0];
    }
    effective_cell_to_forget_scale =
        std::pow(2, -15) * cell_to_forget_weight_scale / intermediate_scale[3];
    effective_cell_to_output_scale =
        std::pow(2, -15) * cell_to_output_weight_scale / intermediate_scale[9];
  }

  // Calculate effecgive scales.
  QuantizeMultiplier(effective_input_to_input_scale,
                     &integer_lstm_param->effective_input_to_input_scale_a,
                     &integer_lstm_param->effective_input_to_input_scale_b);
  QuantizeMultiplier(effective_recurrent_to_input_scale,
                     &integer_lstm_param->effective_recurrent_to_input_scale_a,
                     &integer_lstm_param->effective_recurrent_to_input_scale_b);
  QuantizeMultiplier(effective_cell_to_input_scale,
                     &integer_lstm_param->effective_cell_to_input_scale_a,
                     &integer_lstm_param->effective_cell_to_input_scale_b);
  QuantizeMultiplier(effective_input_to_forget_scale,
                     &integer_lstm_param->effective_input_to_forget_scale_a,
                     &integer_lstm_param->effective_input_to_forget_scale_b);
  QuantizeMultiplier(
      effective_recurrent_to_forget_scale,
      &integer_lstm_param->effective_recurrent_to_forget_scale_a,
      &integer_lstm_param->effective_recurrent_to_forget_scale_b);
  QuantizeMultiplier(effective_cell_to_forget_scale,
                     &integer_lstm_param->effective_cell_to_forget_scale_a,
                     &integer_lstm_param->effective_cell_to_forget_scale_b);
  QuantizeMultiplier(effective_input_to_cell_scale,
                     &integer_lstm_param->effective_input_to_cell_scale_a,
                     &integer_lstm_param->effective_input_to_cell_scale_b);
  QuantizeMultiplier(effective_recurrent_to_cell_scale,
                     &integer_lstm_param->effective_recurrent_to_cell_scale_a,
                     &integer_lstm_param->effective_recurrent_to_cell_scale_b);
  QuantizeMultiplier(effective_input_to_output_scale,
                     &integer_lstm_param->effective_input_to_output_scale_a,
                     &integer_lstm_param->effective_input_to_output_scale_b);
  QuantizeMultiplier(
      effective_recurrent_to_output_scale,
      &integer_lstm_param->effective_recurrent_to_output_scale_a,
      &integer_lstm_param->effective_recurrent_to_output_scale_b);
  QuantizeMultiplier(effective_cell_to_output_scale,
                     &integer_lstm_param->effective_cell_to_output_scale_a,
                     &integer_lstm_param->effective_cell_to_output_scale_b);
  QuantizeMultiplier(effective_proj_scale,
                     &integer_lstm_param->effective_proj_scale_a,
                     &integer_lstm_param->effective_proj_scale_b);
  QuantizeMultiplier(layer_norm_input_scale,
                     &integer_lstm_param->layer_norm_input_scale_a,
                     &integer_lstm_param->layer_norm_input_scale_b);
  QuantizeMultiplier(layer_norm_forget_scale,
                     &integer_lstm_param->layer_norm_forget_scale_a,
                     &integer_lstm_param->layer_norm_forget_scale_b);
  QuantizeMultiplier(layer_norm_cell_scale,
                     &integer_lstm_param->layer_norm_cell_scale_a,
                     &integer_lstm_param->layer_norm_cell_scale_b);
  QuantizeMultiplier(layer_norm_output_scale,
                     &integer_lstm_param->layer_norm_output_scale_a,
                     &integer_lstm_param->layer_norm_output_scale_b);

  {
    // Intermdiates in flatbuffer holds Wx, Wh and Wx+Wh.
    // effective Wx, Wh is in effective_input/recurrent_to_<...>_scale
    // So use intermediate_scale to hold scale from Wx and Wh to Wx+Wh
    // 0: [1] -> [0]
    // 1: [2] -> [0]
    // and use intermdiate_zp as is.
    const float s_1_0 = intermediate_scale[1] / intermediate_scale[0];
    const float s_2_0 = intermediate_scale[2] / intermediate_scale[0];
    const float s_4_3 = intermediate_scale[4] / intermediate_scale[3];
    const float s_5_3 = intermediate_scale[5] / intermediate_scale[3];
    const float s_7_6 = intermediate_scale[7] / intermediate_scale[6];
    const float s_8_6 = intermediate_scale[8] / intermediate_scale[6];
    const float s_10_9 = intermediate_scale[10] / intermediate_scale[9];
    const float s_11_9 = intermediate_scale[11] / intermediate_scale[9];
    QuantizeMultiplier(s_1_0, &integer_lstm_param->intermediate_scale_a[0],
                       &integer_lstm_param->intermediate_scale_b[0]);
    QuantizeMultiplier(s_2_0, &integer_lstm_param->intermediate_scale_a[1],
                       &integer_lstm_param->intermediate_scale_b[1]);
    QuantizeMultiplier(s_4_3, &integer_lstm_param->intermediate_scale_a[2],
                       &integer_lstm_param->intermediate_scale_b[2]);
    QuantizeMultiplier(s_5_3, &integer_lstm_param->intermediate_scale_a[3],
                       &integer_lstm_param->intermediate_scale_b[3]);
    QuantizeMultiplier(s_7_6, &integer_lstm_param->intermediate_scale_a[4],
                       &integer_lstm_param->intermediate_scale_b[4]);
    QuantizeMultiplier(s_8_6, &integer_lstm_param->intermediate_scale_a[5],
                       &integer_lstm_param->intermediate_scale_b[5]);
    QuantizeMultiplier(s_10_9, &integer_lstm_param->intermediate_scale_a[6],
                       &integer_lstm_param->intermediate_scale_b[6]);
    QuantizeMultiplier(s_11_9, &integer_lstm_param->intermediate_scale_a[7],
                       &integer_lstm_param->intermediate_scale_b[7]);
  }

  // Calculate quantized clip for projection and cell.
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);
  const float cell_clip = params->cell_clip;
  const float proj_clip = params->proj_clip;

  const TfLiteTensor* output_tensor = GetOutput(context, node, kOutputTensor);

  auto* cell_state_params = reinterpret_cast<TfLiteAffineQuantization*>(
      cell_state->quantization.params);
  auto* proj_params = reinterpret_cast<TfLiteAffineQuantization*>(
      output_tensor->quantization.params);
  TF_LITE_ENSURE_EQ(context, cell_state_params->scale->data[0], 1.0 / 32768);
  if (cell_clip > 0.0 && cell_clip < 1.0) {
    integer_lstm_param->quantized_cell_clip = static_cast<int16_t>(std::min(
        std::max(cell_clip / cell_state_params->scale->data[0], -32768.0f),
        32767.0f));
  } else {
    integer_lstm_param->quantized_cell_clip = 0;
  }
  if (proj_clip > 0.0) {
    integer_lstm_param->quantized_proj_clip = static_cast<int8_t>(std::min(
        std::max(proj_clip / proj_params->scale->data[0], -128.0f), 127.0f));
  } else {
    integer_lstm_param->quantized_proj_clip = 0;
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->kernel_type = kTfLiteLSTMFullKernel;
  // TODO(b/159066113): maybe just add the minimum required temp tensors?
  context->AddTensors(context, kNumHybridTemporaryTensors,
                      &op_data->scratch_tensor_index);
  return op_data;
}

// LINT.IfChange
// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell,
                                        bool use_layer_norm, bool is_integer) {
  const auto* params = static_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);
  TF_LITE_ENSURE(context, (input_to_forget_weights->type == kTfLiteFloat32) ||
                              (input_to_forget_weights->type == kTfLiteUInt8) ||
                              (input_to_forget_weights->type == kTfLiteInt8));

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const bool use_cifg = (input_to_input_weights == nullptr);
  if (!use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
    TF_LITE_ENSURE_TYPES_EQ(context, input_to_input_weights->type,
                            input_to_forget_weights->type);
  }

  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);
  TF_LITE_ENSURE_TYPES_EQ(context, input_to_cell_weights->type,
                          input_to_forget_weights->type);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  if (recurrent_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
    TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_input_weights->type,
                            input_to_forget_weights->type);
  }

  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);
  TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_forget_weights->type,
                          input_to_forget_weights->type);

  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[1],
                    n_output);
  TF_LITE_ENSURE_TYPES_EQ(context, recurrent_to_cell_weights->type,
                          input_to_forget_weights->type);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
      ((input_to_input_weights != nullptr) &&
       (recurrent_to_input_weights != nullptr)) ||
      ((input_to_input_weights == nullptr) &&
       (recurrent_to_input_weights == nullptr));
  TF_LITE_ENSURE(context, cifg_weights_all_or_none == true);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  if (cell_to_input_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_input_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  if (cell_to_forget_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_forget_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);
  if (cell_to_output_weights) {
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_output_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  // Making sure the peephole weights are there all or none.
  const bool peephole_weights_all_or_none =
      ((cell_to_input_weights != nullptr || use_cifg) &&
       (cell_to_forget_weights != nullptr) &&
       (cell_to_output_weights != nullptr)) ||
      ((cell_to_input_weights == nullptr) &&
       (cell_to_forget_weights == nullptr) &&
       (cell_to_output_weights == nullptr));
  TF_LITE_ENSURE(context, peephole_weights_all_or_none == true);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  if (use_cifg) {
    TF_LITE_ENSURE_EQ(context, input_gate_bias, nullptr);
  } else {
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, input_gate_bias->dims->data[0], n_cell);
    if (is_integer) {
      TF_LITE_ENSURE_TYPES_EQ(context, input_gate_bias->type, kTfLiteInt32);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, input_gate_bias->type, kTfLiteFloat32);
    }
  }

  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* cell_gate_bias =
      GetInput(context, node, kCellGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, cell_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, cell_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  if (projection_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(context, projection_weights->type,
                            input_to_forget_weights->type);
  }

  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);
  if (projection_bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, projection_bias->dims->data[0], n_output);
    if (is_integer) {
      TF_LITE_ENSURE_TYPES_EQ(context, projection_bias->type, kTfLiteInt32);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, projection_bias->type, kTfLiteFloat32);
    }
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO(ghodrat): make sure this is correct.
  const bool projection_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projection_tensors_consistent == true);

  if (use_layer_norm) {
    const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kInputLayerNormCoefficientsTensor);
    if (use_cifg) {
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients, nullptr);
    } else {
      TF_LITE_ENSURE(context, input_layer_norm_coefficients != nullptr);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->size, 1);
      TF_LITE_ENSURE_EQ(context, input_layer_norm_coefficients->dims->data[0],
                        n_cell);
      if (is_integer) {
        TF_LITE_ENSURE_TYPES_EQ(context, input_layer_norm_coefficients->type,
                                kTfLiteInt16);
      } else {
        TF_LITE_ENSURE_TYPES_EQ(context, input_layer_norm_coefficients->type,
                                kTfLiteFloat32);
      }
    }

    const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kForgetLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, forget_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, forget_layer_norm_coefficients->dims->data[0],
                      n_cell);
    if (is_integer) {
      TF_LITE_ENSURE_TYPES_EQ(context, forget_layer_norm_coefficients->type,
                              kTfLiteInt16);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, forget_layer_norm_coefficients->type,
                              kTfLiteFloat32);
    }

    const TfLiteTensor* cell_layer_norm_coefficients =
        GetOptionalInputTensor(context, node, kCellLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, cell_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_layer_norm_coefficients->dims->data[0],
                      n_cell);
    if (is_integer) {
      TF_LITE_ENSURE_TYPES_EQ(context, cell_layer_norm_coefficients->type,
                              kTfLiteInt16);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, cell_layer_norm_coefficients->type,
                              kTfLiteFloat32);
    }

    const TfLiteTensor* output_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kOutputLayerNormCoefficientsTensor);
    TF_LITE_ENSURE(context, output_layer_norm_coefficients != nullptr);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, output_layer_norm_coefficients->dims->data[0],
                      n_cell);
    if (is_integer) {
      TF_LITE_ENSURE_TYPES_EQ(context, output_layer_norm_coefficients->type,
                              kTfLiteInt16);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, output_layer_norm_coefficients->type,
                              kTfLiteFloat32);
    }
  }

  return kTfLiteOk;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)

TfLiteStatus PrecomputeZeroPointTimesWeightWithBias(
    TfLiteContext* context, int32_t zero_point,
    const TfLiteTensor* weight_tensor, const TfLiteTensor* bias_tensor,
    std::unique_ptr<int32_t[]>* output) {
  if (weight_tensor == nullptr) {
    return kTfLiteOk;
  }

  const RuntimeShape& weight_shape = GetTensorShape(weight_tensor);
  TF_LITE_ENSURE_EQ(context, weight_shape.DimensionsCount(), 2);
  const int row = weight_shape.Dims(0);
  const int col = weight_shape.Dims(1);
  output->reset(new int32_t[row]);
  if (bias_tensor == nullptr) {
    memset(output->get(), 0, row * sizeof(int32_t));
  } else {
    const int32_t* bias = GetTensorData<int32_t>(bias_tensor);
    memcpy(output->get(), bias, row * sizeof(int32_t));
  }
  if (zero_point != 0) {
    const int8_t* weight = GetTensorData<int8_t>(weight_tensor);
    tensor_utils::MatrixScalarMultiplyAccumulate(weight, zero_point, row, col,
                                                 output->get());
  }
  return kTfLiteOk;
}

TfLiteStatus PopulatePrecomputedZPTimesWeightsWithBias(TfLiteContext* context,
                                                       OpData* op_data,
                                                       TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* output_state =
      GetVariableInput(context, node, kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);

  const int32_t input_zero_point = -input->params.zero_point;
  const int32_t output_state_zero_point = -output_state->params.zero_point;

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);

  lstm_eval::IntegerLstmParameter* integer_lstm_params =
      &op_data->integer_lstm_param;

  const TfLiteTensor* intermediate =
      &context->tensors[node->intermediates->data[4]];
  const auto* params =
      static_cast<TfLiteAffineQuantization*>(intermediate->quantization.params);
  const int32_t hidden_zp = params->zero_point->data[0];

  // Get bias and perform zero point calculation.
  // When there is layer normalization, the gate bias does not apply to matmul
  // directly:
  //      y = ln(w * x + w * r + w * c) + b.
  const bool is_layer_norm = op_data->use_layer_norm;

  // Forget gate.
  const TfLiteTensor* forget_gate_bias =
      is_layer_norm ? nullptr : GetInput(context, node, kForgetGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_forget_weights, forget_gate_bias,
          &(integer_lstm_params->input_to_forget_effective_bias)));

  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_forget_weights,
          nullptr, &(integer_lstm_params->recurrent_to_forget_effective_bias)));

  // Modulation gate.
  const TfLiteTensor* cell_gate_bias =
      is_layer_norm ? nullptr : GetInput(context, node, kCellGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_cell_weights, cell_gate_bias,
          &(integer_lstm_params->input_to_cell_effective_bias)));
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_cell_weights, nullptr,
          &(integer_lstm_params->recurrent_to_cell_effective_bias)));

  // Output gate.
  const TfLiteTensor* output_gate_bias =
      is_layer_norm ? nullptr : GetInput(context, node, kOutputGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_output_weights, output_gate_bias,
          &(integer_lstm_params->input_to_output_effective_bias)));

  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_output_weights,
          nullptr, &(integer_lstm_params->recurrent_to_output_effective_bias)));

  // Input gate. The calculation is only meaningful for non-cifg case.
  const TfLiteTensor* input_gate_bias =
      is_layer_norm ? nullptr : GetInput(context, node, kInputGateBiasTensor);
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, input_zero_point, input_to_input_weights, input_gate_bias,
          &(integer_lstm_params->input_to_input_effective_bias)));
  TF_LITE_ENSURE_OK(
      context,
      PrecomputeZeroPointTimesWeightWithBias(
          context, output_state_zero_point, recurrent_to_input_weights, nullptr,
          &(integer_lstm_params->recurrent_to_input_effective_bias)));

  // Projection bias. The calculation is only meaningful for with projection.
  TF_LITE_ENSURE_OK(context,
                    PrecomputeZeroPointTimesWeightWithBias(
                        context, hidden_zp, projection_weights, projection_bias,
                        &(integer_lstm_params->projection_effective_bias)));
  return kTfLiteOk;
}

// Resize the output, state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
// LINT.IfChange
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = static_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  // Logic for determining regular lstm and layer norm lstm:
  // input_size, forget_gate_layer_norm_tensor (20) null? is_layer_norm?
  // 20,         N/A,                                     No.
  // 24,         null,                                    No.
  // 24,         not null,                                Yes.
  // 20-inputs lstm are deprecated and is only kept here for backward
  // compatibility.
  if (node->inputs->size == 24) {
    const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, kForgetLayerNormCoefficientsTensor);
    if (forget_layer_norm_coefficients == nullptr) {
      op_data->use_layer_norm = false;
    } else {
      op_data->use_layer_norm = true;
    }
  } else if (node->inputs->size == 20) {
    // This is deprecated and is only kept here for backward compatibility.
    op_data->use_layer_norm = false;
  } else {
    context->ReportError(
        context, "The LSTM Full kernel expects 20 or 24 inputs. Got %d inputs",
        node->inputs->size);
    return kTfLiteError;
  }

  const bool use_layer_norm = op_data->use_layer_norm;

  // Inferring batch size, number of outputs and number of cells from the
  // input tensors.
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const bool is_integer = input->type == kTfLiteInt8;
  TF_LITE_ENSURE(context, input->dims->size > 1);
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];

  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);
  const int n_cell = input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->data[0],
                    n_cell);
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_output,
                                          n_cell, use_layer_norm, is_integer));

  // Get the pointer to output, output_state and cell_state tensors.
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state = GetVariableInput(context, node, kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(output_state), n_batch * n_output);
  TF_LITE_ENSURE_EQ(context, NumElements(cell_state), n_batch * n_cell);

  // Resize the output tensors.
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = n_batch;
  output_size->data[1] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = IsHybridOp(input, input_to_output_weights);

  // The type of Integer LSTM.
  const int num_intermediate_tensors = node->intermediates->size;
  if (is_integer) {
    TF_LITE_ENSURE(context, num_intermediate_tensors == 5 ||
                                num_intermediate_tensors == 12);
  }
  // We use number of intermediate tensors to distinguish the 8 bit matmul
  // output and the 16 bit matmul output version.
  const bool is_8x8_16 = num_intermediate_tensors == 5;

  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(kNumHybridTemporaryTensors);
  } else if (is_integer) {
    if (is_8x8_16) {
      node->temporaries = TfLiteIntArrayCreate(6);
    } else {
      node->temporaries = TfLiteIntArrayCreate(8);
    }
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }

  // Create a scratch buffer tensor for float case and hybrid case.
  // TODO(b/152066492): Create a is_float boolean and reorganize the temporary
  // buffer allocation logic.
  if (!is_integer) {
    node->temporaries->data[kScratchBuffer] =
        op_data->scratch_tensor_index + kScratchBuffer;
    TfLiteTensor* scratch_buffer = GetTemporary(context, node, kScratchBuffer);
    scratch_buffer->type = input->type;
    scratch_buffer->allocation_type = kTfLiteArenaRw;

    const TfLiteTensor* input_to_input_weights =
        GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
    const bool use_cifg = (input_to_input_weights == nullptr);
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
    scratch_buffer_size->data[0] = n_batch;
    if (use_cifg) {
      // Reserving space for Cell, Forget, Output gates
      scratch_buffer_size->data[1] = n_cell * 3;
    } else {
      // Reserving space for Input, Cell, Forget, Output gates
      scratch_buffer_size->data[1] = n_cell * 4;
    }
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  if (is_hybrid_op) {
    op_data->compute_row_sums = true;
    // Allocate temporary tensors to store quantized values of input,
    // output_state and cell_state tensors.
    node->temporaries->data[kInputQuantized] =
        op_data->scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized =
        GetTemporary(context, node, kInputQuantized);
    input_quantized->type = input_to_output_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[kOutputStateQuantized] =
        op_data->scratch_tensor_index + kOutputStateQuantized;
    TfLiteTensor* output_state_quantized =
        GetTemporary(context, node, kOutputStateQuantized);
    output_state_quantized->type = input_to_output_weights->type;
    output_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(output_state_quantized->dims,
                             output_state->dims)) {
      TfLiteIntArray* output_state_quantized_size =
          TfLiteIntArrayCopy(output_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output_state_quantized,
                                              output_state_quantized_size));
    }
    node->temporaries->data[kCellStateQuantized] =
        op_data->scratch_tensor_index + kCellStateQuantized;
    TfLiteTensor* cell_state_quantized =
        GetTemporary(context, node, kCellStateQuantized);
    cell_state_quantized->type = input_to_output_weights->type;
    cell_state_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(cell_state_quantized->dims, cell_state->dims)) {
      TfLiteIntArray* cell_state_quantized_size =
          TfLiteIntArrayCopy(cell_state->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, cell_state_quantized,
                                              cell_state_quantized_size));
    }
    // Allocate temporary tensors to store scaling factors and product scaling
    // factors. The latter is a convenience storage which allows to quantize
    // a vector once (which produces the scaling factors) and multiply it with
    // different matrices (which requires multiplying the scaling factors with
    // the scaling factor of the matrix).
    node->temporaries->data[kInputScalingFactors] =
        op_data->scratch_tensor_index + kInputScalingFactors;
    TfLiteTensor* input_sf = GetTemporary(context, node, kInputScalingFactors);
    input_sf->type = kTfLiteFloat32;
    input_sf->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {n_batch};
    if (!TfLiteIntArrayEqualsArray(input_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_sf_size = TfLiteIntArrayCreate(1);
      input_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_sf, input_sf_size));
    }
    node->temporaries->data[kOutputStateScalingFactors] =
        op_data->scratch_tensor_index + kOutputStateScalingFactors;
    TfLiteTensor* output_state_sf =
        GetTemporary(context, node, kOutputStateScalingFactors);
    output_state_sf->type = kTfLiteFloat32;
    output_state_sf->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_sf_size = TfLiteIntArrayCreate(1);
      output_state_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_sf,
                                                       output_state_sf_size));
    }
    node->temporaries->data[kProductScalingFactors] =
        op_data->scratch_tensor_index + kProductScalingFactors;
    TfLiteTensor* prod_scaling_factors =
        GetTemporary(context, node, kProductScalingFactors);
    prod_scaling_factors->type = kTfLiteFloat32;
    prod_scaling_factors->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(prod_scaling_factors->dims, 1,
                                   scaling_dims)) {
      TfLiteIntArray* prod_scaling_factors_size = TfLiteIntArrayCreate(1);
      prod_scaling_factors_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, prod_scaling_factors,
                                              prod_scaling_factors_size));
    }

    // Allocate a temporary tensor to store the recovered cell weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    node->temporaries->data[kRecoveredCellWeights] =
        op_data->scratch_tensor_index + kRecoveredCellWeights;
    TfLiteTensor* recovered_cell_weights =
        GetTemporary(context, node, kRecoveredCellWeights);
    recovered_cell_weights->type = kTfLiteFloat32;
    recovered_cell_weights->allocation_type = kTfLiteArenaRw;
    int recovered_cell_dims[1] = {n_cell};
    if (!TfLiteIntArrayEqualsArray(recovered_cell_weights->dims, 1,
                                   recovered_cell_dims)) {
      TfLiteIntArray* recovered_cell_weights_size = TfLiteIntArrayCreate(1);
      recovered_cell_weights_size->data[0] = n_cell;
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, recovered_cell_weights,
                                              recovered_cell_weights_size));
    }
    // Allocate a temporary tensor to store accumulate values for matrix
    // multiplication before multiplication by scaling factor
    node->temporaries->data[kAccumScratch] =
        op_data->scratch_tensor_index + kAccumScratch;
    TfLiteTensor* accum_scratch = GetTemporary(context, node, kAccumScratch);
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {n_cell, n_batch};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = n_cell;
      accum_size->data[1] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }
    node->temporaries->data[kInputZeroPoints] =
        op_data->scratch_tensor_index + kInputZeroPoints;
    TfLiteTensor* input_zp = GetTemporary(context, node, kInputZeroPoints);
    input_zp->type = kTfLiteFloat32;
    input_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_zp_size = TfLiteIntArrayCreate(1);
      input_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, input_zp, input_zp_size));
    }
    node->temporaries->data[kOutputStateZeroPoints] =
        op_data->scratch_tensor_index + kOutputStateZeroPoints;
    TfLiteTensor* output_state_zp =
        GetTemporary(context, node, kOutputStateZeroPoints);
    output_state_zp->type = kTfLiteFloat32;
    output_state_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_zp_size = TfLiteIntArrayCreate(1);
      output_state_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_zp,
                                                       output_state_zp_size));
    }

    node->temporaries->data[kRowSums] =
        op_data->scratch_tensor_index + kRowSums;
    const TfLiteTensor* input_to_input_weights =
        GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
    const bool use_cifg = (input_to_input_weights == nullptr);
    int row_sums_rows = use_cifg ? 6 : 8;
    const TfLiteTensor* projection_weights =
        GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
    if (projection_weights != nullptr) {
      row_sums_rows += ceil(static_cast<float>(n_output) / n_cell);
    }

    TfLiteTensor* row_sums = GetTemporary(context, node, kRowSums);
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    const int row_sums_dims[2] = {row_sums_rows, n_cell};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 2, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(2);
      row_sums_size->data[0] = row_sums_dims[0];
      row_sums_size->data[1] = row_sums_dims[1];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }
  }

  if (is_integer) {
    if (is_8x8_16) {
      // Integer LSTM prepare function for 8x8->16.
      // This code path needs 5 intermediate tensors per Op.
      // Populate quantization parameters.
      PopulateQuantizedLstmParams8x8_16(context, node,
                                        &op_data->integer_lstm_param);

      // Allocate scratch buffer. Need 6 16bit buffer with size n_batch * n_cell
      // and 1 8bit buffer with size n_batch * n_cell. We also need 1 32 bit
      // buffer with size n_batch * n_cell.
      //
      // Handle cifg case as well, which might save one buffer.
      for (int scratch_index = 0; scratch_index < 6; ++scratch_index) {
        node->temporaries->data[scratch_index] =
            op_data->scratch_tensor_index + scratch_index;
        TfLiteTensor* scratch_tensor =
            GetTemporary(context, node, scratch_index);
        scratch_tensor->type = kTfLiteInt16;
        if (scratch_index == 4) {
          scratch_tensor->type = kTfLiteInt8;
        } else if (scratch_index == 5) {
          scratch_tensor->type = kTfLiteInt32;
        }
        scratch_tensor->allocation_type = kTfLiteArenaRw;
        const int scratch_dimension[2] = {n_batch, n_cell};
        if (!TfLiteIntArrayEqualsArray(scratch_tensor->dims, 2,
                                       scratch_dimension)) {
          TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
          scratch_buffer_size->data[0] = n_batch;
          scratch_buffer_size->data[1] = n_cell;
          TF_LITE_ENSURE_OK(context,
                            context->ResizeTensor(context, scratch_tensor,
                                                  scratch_buffer_size));
        }
      }

      // Populate precomputed zp * weight.
      TF_LITE_ENSURE_OK(context, PopulatePrecomputedZPTimesWeightsWithBias(
                                     context, op_data, node));
    } else {
      // Integer LSTM prepare function for 8x8->8.
      // This code path needs 12 intermediate tensors per Op.
      PopulateQuantizedLstmParams8x8_8(context, node,
                                       &op_data->integer_lstm_param);

      // Allocate scratch buffer. Need 6 16bit buffer with size n_batch * n_cell
      // and 2 8bit buffer with size n_batch * n_cell.
      //
      // Handle cifg case as well, which might save one buffer.
      for (int scratch_index = 0; scratch_index < 8; ++scratch_index) {
        node->temporaries->data[scratch_index] =
            op_data->scratch_tensor_index + scratch_index;
        TfLiteTensor* scratch_tensor =
            GetTemporary(context, node, scratch_index);
        if (scratch_index == 0 || scratch_index == 1) {
          scratch_tensor->type = kTfLiteInt8;
        } else {
          scratch_tensor->type = kTfLiteInt16;
        }
        scratch_tensor->allocation_type = kTfLiteArenaRw;
        const int scratch_dimension[2] = {n_batch, n_cell};
        if (!TfLiteIntArrayEqualsArray(scratch_tensor->dims, 2,
                                       scratch_dimension)) {
          TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
          scratch_buffer_size->data[0] = n_batch;
          scratch_buffer_size->data[1] = n_cell;
          TF_LITE_ENSURE_OK(context,
                            context->ResizeTensor(context, scratch_tensor,
                                                  scratch_buffer_size));
        }
      }
    }
  }
  return kTfLiteOk;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)

// LINT.IfChange
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = static_cast<TfLiteLSTMParams*>(node->builtin_data);
  OpData* op_data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  const TfLiteTensor* input_to_input_weights =
      GetOptionalInputTensor(context, node, kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights =
      GetInput(context, node, kInputToForgetWeightsTensor);
  const TfLiteTensor* input_to_cell_weights =
      GetInput(context, node, kInputToCellWeightsTensor);
  const TfLiteTensor* input_to_output_weights =
      GetInput(context, node, kInputToOutputWeightsTensor);

  const TfLiteTensor* recurrent_to_input_weights =
      GetOptionalInputTensor(context, node, kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights =
      GetInput(context, node, kRecurrentToForgetWeightsTensor);
  const TfLiteTensor* recurrent_to_cell_weights =
      GetInput(context, node, kRecurrentToCellWeightsTensor);
  const TfLiteTensor* recurrent_to_output_weights =
      GetInput(context, node, kRecurrentToOutputWeightsTensor);

  const TfLiteTensor* cell_to_input_weights =
      GetOptionalInputTensor(context, node, kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights =
      GetOptionalInputTensor(context, node, kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights =
      GetOptionalInputTensor(context, node, kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kInputLayerNormCoefficientsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* cell_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kCellLayerNormCoefficientsTensor);
  const TfLiteTensor* output_layer_norm_coefficients =
      GetOptionalInputTensor(context, node, kOutputLayerNormCoefficientsTensor);

  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias =
      GetInput(context, node, kForgetGateBiasTensor);
  const TfLiteTensor* cell_gate_bias =
      GetInput(context, node, kCellGateBiasTensor);
  const TfLiteTensor* output_gate_bias =
      GetInput(context, node, kOutputGateBiasTensor);

  const TfLiteTensor* projection_weights =
      GetOptionalInputTensor(context, node, kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, kProjectionBiasTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state = GetVariableInput(context, node, kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input_to_output_weights->type) {
    case kTfLiteFloat32: {
      // Index the scratch buffers pointers to the global scratch buffer.
      TfLiteTensor* scratch_buffer = GetTemporary(context, node, 0);
      return lstm_eval::EvalFloat(
          input, input_to_input_weights, input_to_forget_weights,
          input_to_cell_weights, input_to_output_weights,
          recurrent_to_input_weights, recurrent_to_forget_weights,
          recurrent_to_cell_weights, recurrent_to_output_weights,
          cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights,
          input_layer_norm_coefficients, forget_layer_norm_coefficients,
          cell_layer_norm_coefficients, output_layer_norm_coefficients,
          /*aux_input=*/nullptr,
          /*aux_input_to_input_weights=*/nullptr,
          /*aux_input_to_forget_weights=*/nullptr,
          /*aux_input_to_cell_weights=*/nullptr,
          /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
          forget_gate_bias, cell_gate_bias, output_gate_bias,
          projection_weights, projection_bias, params,
          /*forward_sequence=*/true,
          /*time_major=*/true,
          /*output_offset=*/0, scratch_buffer, output_state, cell_state,
          output);
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      const bool is_hybrid = (input->type == kTfLiteFloat32);
      if (is_hybrid) {
        TfLiteTensor* row_sums = GetTemporary(context, node, kRowSums);
        const int row_sums_size = row_sums->dims->data[0];
        return lstm_eval::EvalHybrid(
            input, input_to_input_weights, input_to_forget_weights,
            input_to_cell_weights, input_to_output_weights,
            recurrent_to_input_weights, recurrent_to_forget_weights,
            recurrent_to_cell_weights, recurrent_to_output_weights,
            cell_to_input_weights, cell_to_forget_weights,
            cell_to_output_weights, input_layer_norm_coefficients,
            forget_layer_norm_coefficients, cell_layer_norm_coefficients,
            output_layer_norm_coefficients, /*aux_input=*/nullptr,
            /*aux_input_to_input_weights=*/nullptr,
            /*aux_input_to_forget_weights=*/nullptr,
            /*aux_input_to_cell_weights=*/nullptr,
            /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
            forget_gate_bias, cell_gate_bias, output_gate_bias,
            projection_weights, projection_bias, params,
            /*forward_sequence=*/true, /*time_major=*/true, /*output_offset=*/0,
            GetTemporary(context, node, kScratchBuffer),
            GetTemporary(context, node, kInputScalingFactors),
            /*aux_input_sf=*/nullptr,
            GetTemporary(context, node, kOutputStateScalingFactors),
            GetTemporary(context, node, kProductScalingFactors),
            GetTemporary(context, node, kRecoveredCellWeights),
            GetTemporary(context, node, kInputQuantized),
            /*aux_input_quantized=*/nullptr,
            GetTemporary(context, node, kOutputStateQuantized),
            GetTemporary(context, node, kCellStateQuantized), output_state,
            cell_state, GetTemporary(context, node, kAccumScratch), output,
            GetTemporary(context, node, kInputZeroPoints),
            /*aux_input_zp=*/nullptr,
            GetTemporary(context, node, kOutputStateZeroPoints), row_sums,
            row_sums_size, &op_data->compute_row_sums,
            CpuBackendContext::GetFromContext(context));
      } else {
        const int num_intermediate_tensors = node->intermediates->size;
        if (num_intermediate_tensors == 5) {
          TfLiteTensor* scratch0 = GetTemporary(context, node, 0);
          TfLiteTensor* scratch1 = GetTemporary(context, node, 1);
          TfLiteTensor* scratch2 = GetTemporary(context, node, 2);
          TfLiteTensor* scratch3 = GetTemporary(context, node, 3);
          TfLiteTensor* scratch4 = GetTemporary(context, node, 4);
          TfLiteTensor* scratch5 = GetTemporary(context, node, 5);
          return lstm_eval::EvalInteger8x8_16(
              input, input_to_input_weights, input_to_forget_weights,
              input_to_cell_weights, input_to_output_weights,
              recurrent_to_input_weights, recurrent_to_forget_weights,
              recurrent_to_cell_weights, recurrent_to_output_weights,
              cell_to_input_weights, cell_to_forget_weights,
              cell_to_output_weights, input_layer_norm_coefficients,
              forget_layer_norm_coefficients, cell_layer_norm_coefficients,
              output_layer_norm_coefficients, input_gate_bias, forget_gate_bias,
              cell_gate_bias, output_gate_bias, projection_weights,
              projection_bias, params, &op_data->integer_lstm_param,
              output_state, cell_state, output, scratch0, scratch1, scratch2,
              scratch3, scratch4, scratch5,
              CpuBackendContext::GetFromContext(context));
        } else {
          TfLiteTensor* scratch0 = GetTemporary(context, node, 0);
          TfLiteTensor* scratch1 = GetTemporary(context, node, 1);
          TfLiteTensor* scratch2 = GetTemporary(context, node, 2);
          TfLiteTensor* scratch3 = GetTemporary(context, node, 3);
          TfLiteTensor* scratch4 = GetTemporary(context, node, 4);
          TfLiteTensor* scratch5 = GetTemporary(context, node, 5);
          TfLiteTensor* scratch6 = GetTemporary(context, node, 6);
          TfLiteTensor* scratch7 = GetTemporary(context, node, 7);
          return lstm_eval::EvalInteger8x8_8(
              input, input_to_input_weights, input_to_forget_weights,
              input_to_cell_weights, input_to_output_weights,
              recurrent_to_input_weights, recurrent_to_forget_weights,
              recurrent_to_cell_weights, recurrent_to_output_weights,
              cell_to_input_weights, cell_to_forget_weights,
              cell_to_output_weights, input_layer_norm_coefficients,
              forget_layer_norm_coefficients, cell_layer_norm_coefficients,
              output_layer_norm_coefficients, input_gate_bias, forget_gate_bias,
              cell_gate_bias, output_gate_bias, projection_weights,
              projection_bias, params, output_state, cell_state, output,
              &op_data->integer_lstm_param, scratch0, scratch1, scratch2,
              scratch3, scratch4, scratch5, scratch6, scratch7);
          return kTfLiteOk;
        }
      }
    }
    default:
      context->ReportError(context, "Type %d is not currently supported.",
                           input_to_output_weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)

}  // namespace full

// For basic kernel (5-inputs).
namespace basic {

enum InputTensor {
  kInputData = 0,
  kInputPrevActivation = 1,
  kInputWeights = 2,
  kInputBiases = 3,
  kInputPrevState = 4,
  kInputNum = 5,
};

enum OutputTensor {
  kOutputActivation = 0,
  kOutputState = 1,
  kOutputConcatTemp = 2,
  kOutputActivationTemp = 3,
  kOutputNum = 4,
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->kernel_type = kTfLiteLSTMBasicKernel;
  // `scratch_tensor_index` is unused in this kernel.
  op_data->scratch_tensor_index = -1;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size == kInputNum);
  TF_LITE_ENSURE(context, node->outputs->size == kOutputNum);

  const TfLiteTensor* input = GetInput(context, node, kInputData);
  const TfLiteTensor* prev_activation =
      GetInput(context, node, kInputPrevActivation);
  const TfLiteTensor* weights = GetInput(context, node, kInputWeights);
  const TfLiteTensor* bias = GetInput(context, node, kInputBiases);
  const TfLiteTensor* prev_state = GetInput(context, node, kInputPrevState);

  TF_LITE_ENSURE_EQ(context, input->dims->size, 2);
  const int num_batches = input->dims->data[0];
  const int input_depth = input->dims->data[1];

  TF_LITE_ENSURE_EQ(context, prev_activation->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, prev_activation->dims->data[0], num_batches);
  const int activation_depth = prev_activation->dims->data[1];
  const int total_depth = input_depth + activation_depth;

  TF_LITE_ENSURE_EQ(context, weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, weights->dims->data[0], 4 * activation_depth);
  TF_LITE_ENSURE_EQ(context, weights->dims->data[1], total_depth);

  TF_LITE_ENSURE_EQ(context, bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, bias->dims->data[0], 4 * activation_depth);

  TF_LITE_ENSURE_EQ(context, prev_state->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, prev_state->dims->data[0], num_batches);
  TF_LITE_ENSURE_EQ(context, prev_state->dims->data[1], activation_depth);

  TfLiteTensor* activation_out = GetOutput(context, node, kOutputActivation);
  TfLiteTensor* state_out = GetOutput(context, node, kOutputState);
  TfLiteTensor* concat_temp = GetOutput(context, node, kOutputConcatTemp);
  TfLiteTensor* activation_temp =
      GetOutput(context, node, kOutputActivationTemp);

  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, activation_out,
                                 TfLiteIntArrayCopy(prev_activation->dims)));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, state_out,
                                     TfLiteIntArrayCopy(prev_state->dims)));

  TfLiteIntArray* concat_temp_size = TfLiteIntArrayCreate(2);
  concat_temp_size->data[0] = num_batches;
  concat_temp_size->data[1] = total_depth;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, concat_temp, concat_temp_size));
  TfLiteIntArray* activation_temp_size = TfLiteIntArrayCreate(2);
  activation_temp_size->data[0] = num_batches;
  activation_temp_size->data[1] = 4 * activation_depth;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, activation_temp,
                                                   activation_temp_size));

  // Set the state tensors as persistent.
  for (auto index : {kInputPrevActivation, kInputPrevState}) {
    TfLiteTensor* tensor = &context->tensors[node->inputs->data[index]];
    tensor->allocation_type = kTfLiteArenaRwPersistent;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputData);
  const TfLiteTensor* prev_activation =
      GetInput(context, node, kInputPrevActivation);
  const TfLiteTensor* weights = GetInput(context, node, kInputWeights);
  const TfLiteTensor* bias = GetInput(context, node, kInputBiases);
  const TfLiteTensor* prev_state = GetInput(context, node, kInputPrevState);

  TfLiteTensor* activation_out = GetOutput(context, node, kOutputActivation);
  TfLiteTensor* state_out = GetOutput(context, node, kOutputState);
  TfLiteTensor* concat_temp = GetOutput(context, node, kOutputConcatTemp);
  TfLiteTensor* activation_temp =
      GetOutput(context, node, kOutputActivationTemp);

  if (input->type == kTfLiteFloat32 &&
      prev_activation->type == kTfLiteFloat32 &&
      weights->type == kTfLiteFloat32 && bias->type == kTfLiteFloat32 &&
      prev_state->type == kTfLiteFloat32 && state_out->type == kTfLiteFloat32 &&
      activation_out->type == kTfLiteFloat32 &&
      concat_temp->type == kTfLiteFloat32 &&
      activation_temp->type == kTfLiteFloat32) {
    tflite::LstmCellParams op_params;
    // Float LSTM cell does not need parameters to be set: leave untouched.
    optimized_ops::LstmCell(
        op_params,
        // Inputs.
        GetTensorShape(input), GetTensorData<float>(input),
        GetTensorShape(prev_activation), GetTensorData<float>(prev_activation),
        GetTensorShape(weights), GetTensorData<float>(weights),
        GetTensorShape(bias), GetTensorData<float>(bias),
        GetTensorShape(prev_state), GetTensorData<float>(prev_state),
        // Outputs.
        GetTensorShape(state_out), GetTensorData<float>(state_out),
        GetTensorShape(activation_out), GetTensorData<float>(activation_out),
        GetTensorShape(concat_temp), GetTensorData<float>(concat_temp),
        GetTensorShape(activation_temp), GetTensorData<float>(activation_temp),
        CpuBackendContext::GetFromContext(context));
  } else if (input->type == kTfLiteUInt8 &&
             prev_activation->type == kTfLiteUInt8 &&
             weights->type == kTfLiteUInt8 && bias->type == kTfLiteInt32 &&
             prev_state->type == kTfLiteInt16 &&
             state_out->type == kTfLiteInt16 &&
             activation_out->type == kTfLiteUInt8 &&
             concat_temp->type == kTfLiteUInt8 &&
             activation_temp->type == kTfLiteInt16) {
    int state_scale_log2_rounded;
    if (!CheckedLog2(state_out->params.scale, &state_scale_log2_rounded)) {
      context->ReportError(
          context,
          "The internal state of a LSTM cell must have a power-of-two scale.");
      return kTfLiteError;
    }
    const int state_integer_bits = 15 + state_scale_log2_rounded;
    if (state_integer_bits != 4) {
      context->ReportError(context,
                           "The only case of quantized LstmCell currently "
                           "supported is with StateIntegerBits==4");
      return kTfLiteError;
    }

    double real_accum_multiplier = 4096 * bias->params.scale;
    int32 accum_multiplier;
    int accum_shift;
    tflite::QuantizeMultiplier(real_accum_multiplier, &accum_multiplier,
                               &accum_shift);
    tflite::LstmCellParams op_params;
    op_params.weights_zero_point = weights->params.zero_point;
    op_params.accum_multiplier = accum_multiplier;
    op_params.accum_shift = accum_shift;
    optimized_ops::LstmCell<4>(
        op_params,
        // Inputs.
        GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(prev_activation),
        GetTensorData<uint8_t>(prev_activation), GetTensorShape(weights),
        GetTensorData<uint8_t>(weights), GetTensorShape(bias),
        GetTensorData<int32_t>(bias), GetTensorShape(prev_state),
        GetTensorData<int16_t>(prev_state),
        // Outputs.
        GetTensorShape(state_out), GetTensorData<int16_t>(state_out),
        GetTensorShape(activation_out), GetTensorData<uint8_t>(activation_out),
        GetTensorShape(concat_temp), GetTensorData<uint8_t>(concat_temp),
        GetTensorShape(activation_temp),
        GetTensorData<int16_t>(activation_temp),
        CpuBackendContext::GetFromContext(context));
  } else {
    context->ReportError(context,
                         "Unsupported combination of data types for LstmCell");
    return kTfLiteError;
  }

  // TODO(ycling): Investigate if this copy can be avoided with the 5-inputs
  // LSTM kernel.
  memcpy(prev_activation->data.raw, activation_out->data.raw,
         activation_out->bytes);
  memcpy(prev_state->data.raw, state_out->data.raw, state_out->bytes);

  return kTfLiteOk;
}

}  // namespace basic

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const auto* params = reinterpret_cast<const TfLiteLSTMParams*>(buffer);
  switch (params->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Init(context, buffer, length);
    case kTfLiteLSTMBasicKernel:
      return basic::Init(context, buffer, length);
    default:
      return nullptr;
  }
  return nullptr;
}
void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  switch (op_data->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Prepare(context, node);
    case kTfLiteLSTMBasicKernel:
      return basic::Prepare(context, node);
    default:
      return kTfLiteError;
  }
  return kTfLiteError;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  switch (op_data->kernel_type) {
    case kTfLiteLSTMFullKernel:
      return full::Eval(context, node);
    case kTfLiteLSTMBasicKernel:
      return basic::Eval(context, node);
    default:
      return kTfLiteError;
  }
  return kTfLiteError;
}

}  // namespace lstm

TfLiteRegistration* Register_LSTM() {
  static TfLiteRegistration r = {lstm::Init, lstm::Free, lstm::Prepare,
                                 lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
