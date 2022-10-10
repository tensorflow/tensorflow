/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>

#include <algorithm>
#include <cstddef>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_eval.h"
#include "tensorflow/lite/kernels/lstm_shared.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unidirectional_sequence_lstm {
namespace {

struct OpData {
  // If the lstm is layer norm.
  bool use_layer_norm;
  // The scratch tensor index.
  int scratch_tensor_index;
  bool compute_row_sums = false;

  lstm_eval::IntegerLstmParameter integer_lstm_param;
};

TfLiteStatus PopulateQuantizedLstmParams8x8_16(
    TfLiteContext* context, TfLiteNode* node,
    lstm_eval::IntegerLstmParameter* integer_lstm_param) {
  // Calculate quantized clip for projection and cell.
  const auto* params =
      static_cast<TfLiteUnidirectionalSequenceLSTMParams*>(node->builtin_data);
  const float cell_clip = params->cell_clip;
  const float proj_clip = params->proj_clip;

  const TfLiteTensor* cell_state =
      GetVariableInput(context, node, lstm::full::kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, lstm::full::kOutputTensor, &output_tensor));

  TF_LITE_ENSURE(context,
                 cell_state->quantization.type != kTfLiteNoQuantization);
  auto* cell_state_params =
      static_cast<TfLiteAffineQuantization*>(cell_state->quantization.params);
  TF_LITE_ENSURE(context,
                 output_tensor->quantization.type != kTfLiteNoQuantization);
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

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kInputTensor, &input));

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToForgetWeightsTensor,
                   &input_to_forget_weights));
  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node,
                                          lstm::full::kInputToCellWeightsTensor,
                                          &input_to_cell_weights));
  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToOutputWeightsTensor,
                   &input_to_output_weights));

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToForgetWeightsTensor,
                   &recurrent_to_forget_weights));
  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToCellWeightsTensor,
                   &recurrent_to_cell_weights));
  const TfLiteTensor* recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToOutputWeightsTensor,
                   &recurrent_to_output_weights));

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
      context, node, lstm::full::kInputLayerNormCoefficientsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
      context, node, lstm::full::kForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* cell_layer_norm_coefficients = GetOptionalInputTensor(
      context, node, lstm::full::kCellLayerNormCoefficientsTensor);
  const TfLiteTensor* output_layer_norm_coefficients = GetOptionalInputTensor(
      context, node, lstm::full::kOutputLayerNormCoefficientsTensor);

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
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
      TfLiteTensor* intermediate;
      TF_LITE_ENSURE_OK(context,
                        GetIntermediatesSafe(context, node, i, &intermediate));
      TF_LITE_ENSURE(context,
                     intermediate->quantization.type != kTfLiteNoQuantization);
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
  // In the absence of projection, hidden becomes otuput and this intermediate
  // is ignored.
  TfLiteTensor* hidden;
  TF_LITE_ENSURE_OK(context, GetIntermediatesSafe(context, node, 4, &hidden));
  TF_LITE_ENSURE(context, hidden->quantization.type != kTfLiteNoQuantization);
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
  // TF_LITE_ENSURE(context, cell_scale <= -9);
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

}  // namespace

// Temporary tensors
enum TemporaryTensor {
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
  kNumTemporaryTensors = 12,
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  context->AddTensors(context, kNumTemporaryTensors,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

// Check that input tensor dimensions matches with each other.
TfLiteStatus CheckInputTensorDimensions(TfLiteContext* context,
                                        TfLiteNode* node, int n_input,
                                        int n_output, int n_cell,
                                        bool use_layer_norm, bool is_integer) {
  const auto* params = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  TF_LITE_ENSURE(context, params->cell_clip >= 0);
  TF_LITE_ENSURE(context, params->proj_clip >= 0);

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  if (input_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_EQ(context, input_to_input_weights->dims->data[1], n_input);
  }

  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToForgetWeightsTensor,
                   &input_to_forget_weights));
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_forget_weights->dims->data[1], n_input);

  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node,
                                          lstm::full::kInputToCellWeightsTensor,
                                          &input_to_cell_weights));
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, input_to_cell_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  if (recurrent_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[0],
                      n_cell);
    TF_LITE_ENSURE_EQ(context, recurrent_to_input_weights->dims->data[1],
                      n_output);
  }

  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToForgetWeightsTensor,
                   &recurrent_to_forget_weights));
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[0],
                    n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_forget_weights->dims->data[1],
                    n_output);

  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToCellWeightsTensor,
                   &recurrent_to_cell_weights));
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[0], n_cell);
  TF_LITE_ENSURE_EQ(context, recurrent_to_cell_weights->dims->data[1],
                    n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
      ((input_to_input_weights != nullptr) &&
       (recurrent_to_input_weights != nullptr)) ||
      ((input_to_input_weights == nullptr) &&
       (recurrent_to_input_weights == nullptr));
  TF_LITE_ENSURE(context, cifg_weights_all_or_none == true);

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToInputWeightsTensor);
  if (cell_to_input_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_input_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_input_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToForgetWeightsTensor);
  if (cell_to_forget_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_forget_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_forget_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToOutputWeightsTensor);
  if (cell_to_output_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, cell_to_output_weights->dims->data[0], n_cell);
    TF_LITE_ENSURE_TYPES_EQ(
        context, cell_to_output_weights->type,
        is_integer ? kTfLiteInt16 : input_to_forget_weights->type);
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = (input_to_input_weights == nullptr);
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
      GetOptionalInputTensor(context, node, lstm::full::kInputGateBiasTensor);
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

  const TfLiteTensor* forget_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kForgetGateBiasTensor,
                            &forget_gate_bias));
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, forget_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, forget_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* cell_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, lstm::full::kCellGateBiasTensor,
                                 &cell_gate_bias));
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cell_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, cell_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, cell_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* output_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kOutputGateBiasTensor,
                            &output_gate_bias));
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, output_gate_bias->dims->data[0], n_cell);
  if (is_integer) {
    TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteInt32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output_gate_bias->type, kTfLiteFloat32);
  }

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);
  if (projection_weights != nullptr) {
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[0], n_output);
    TF_LITE_ENSURE_EQ(context, projection_weights->dims->data[1], n_cell);
  }

  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, lstm::full::kProjectionBiasTensor);
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
  const bool projecton_tensors_consistent =
      ((projection_weights != nullptr) || (projection_bias == nullptr));
  TF_LITE_ENSURE(context, projecton_tensors_consistent == true);

  if (use_layer_norm) {
    const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, lstm::full::kInputLayerNormCoefficientsTensor);
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

    const TfLiteTensor* forget_layer_norm_coefficients;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node,
                              lstm::full::kForgetLayerNormCoefficientsTensor,
                              &forget_layer_norm_coefficients));
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

    const TfLiteTensor* cell_layer_norm_coefficients;
    TF_LITE_ENSURE_OK(context,
                      GetInputSafe(context, node,
                                   lstm::full::kCellLayerNormCoefficientsTensor,
                                   &cell_layer_norm_coefficients));
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

    const TfLiteTensor* output_layer_norm_coefficients;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node,
                              lstm::full::kOutputLayerNormCoefficientsTensor,
                              &output_layer_norm_coefficients));
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
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kInputTensor, &input));
  const TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);

  const int32_t input_zero_point = -input->params.zero_point;
  const int32_t output_state_zero_point = -output_state->params.zero_point;

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToForgetWeightsTensor,
                   &input_to_forget_weights));
  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node,
                                          lstm::full::kInputToCellWeightsTensor,
                                          &input_to_cell_weights));
  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToOutputWeightsTensor,
                   &input_to_output_weights));

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToForgetWeightsTensor,
                   &recurrent_to_forget_weights));
  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToCellWeightsTensor,
                   &recurrent_to_cell_weights));
  const TfLiteTensor* recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToOutputWeightsTensor,
                   &recurrent_to_output_weights));

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, lstm::full::kProjectionBiasTensor);

  lstm_eval::IntegerLstmParameter* integer_lstm_params =
      &op_data->integer_lstm_param;

  const TfLiteTensor* intermediate =
      &context->tensors[node->intermediates->data[4]];
  TF_LITE_ENSURE(context,
                 intermediate->quantization.type != kTfLiteNoQuantization);
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
      is_layer_norm
          ? nullptr
          : GetInput(context, node, lstm::full::kForgetGateBiasTensor);
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
      is_layer_norm ? nullptr
                    : GetInput(context, node, lstm::full::kCellGateBiasTensor);
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
      is_layer_norm
          ? nullptr
          : GetInput(context, node, lstm::full::kOutputGateBiasTensor);
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
      is_layer_norm ? nullptr
                    : GetInput(context, node, lstm::full::kInputGateBiasTensor);
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

// Resize the output and  state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const int scratch_tensor_index = op_data->scratch_tensor_index;

  // Check we have all the inputs and outputs we need.
  bool use_layer_norm = false;
  if (node->inputs->size == 24) {
    const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
        context, node, lstm::full::kForgetLayerNormCoefficientsTensor);
    if (forget_layer_norm_coefficients == nullptr) {
      use_layer_norm = false;
    } else {
      use_layer_norm = true;
    }
  } else if (node->inputs->size == 20) {
    // This is deprecated and is only kept here for backward compatibility.
    use_layer_norm = false;
  } else {
    TF_LITE_KERNEL_LOG(
        context, "The LSTM Full kernel expects 20 or 24 inputs. Got %d inputs",
        node->inputs->size);
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  op_data->use_layer_norm = use_layer_norm;

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kInputTensor, &input));
  const bool is_integer = input->type == kTfLiteInt8;
  TF_LITE_ENSURE(context, input->dims->size > 1);
  const auto* params =
      reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          node->builtin_data);
  const bool time_major = params->time_major;
  const int n_batch = time_major ? input->dims->data[1] : input->dims->data[0];
  const int n_input = input->dims->data[2];

  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToOutputWeightsTensor,
                   &input_to_output_weights));
  const int n_cell = input_to_output_weights->dims->data[0];
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_to_output_weights->dims->data[1], n_input);

  const TfLiteTensor* recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToOutputWeightsTensor,
                   &recurrent_to_output_weights));
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, recurrent_to_output_weights->dims->data[0],
                    n_cell);
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Check that input tensor dimensions matches with each other.
  TF_LITE_ENSURE_OK(
      context, CheckInputTensorDimensions(context, node, n_input, n_output,
                                          n_cell, use_layer_norm, is_integer));

  // Get the pointer to output, output_state and cell_state buffer tensors.
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                           lstm::full::kOutputTensor, &output));

  TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state =
      GetVariableInput(context, node, lstm::full::kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  TF_LITE_ENSURE_EQ(context, NumElements(output_state), n_batch * n_output);
  TF_LITE_ENSURE_EQ(context, NumElements(cell_state), n_batch * n_cell);

  // Resize the output tensors.
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  output_size->data[input->dims->size - 1] = n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  if (is_integer) {
    const int num_intermediate_tensors = node->intermediates->size;
    TF_LITE_ENSURE(context, num_intermediate_tensors == 5);
  }

  TfLiteIntArrayFree(node->temporaries);
  if (IsHybridOp(input, input_to_output_weights)) {
    node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors);
  } else if (is_integer) {
    node->temporaries = TfLiteIntArrayCreate(6);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[kScratchBuffer] =
      scratch_tensor_index + kScratchBuffer;

  // Create a scratch buffer tensor.
  TfLiteTensor* scratch_buffer;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kScratchBuffer,
                                              &scratch_buffer));
  scratch_buffer->type = input->type;
  scratch_buffer->allocation_type = kTfLiteArenaRw;

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  const bool use_cifg = (input_to_input_weights == nullptr);
  TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(2);
  scratch_buffer_size->data[0] = n_batch;
  if (use_cifg) {
    // Reserving space for Cell, Forget, Output gates and scratch accumulation
    // buffer and an extra 16 bytes to avoid internal ruy copies.
    scratch_buffer_size->data[1] = n_cell * 4 + 16;
  } else {
    // Reserving space for Input, Cell, Forget, Output gates and scratch
    // accumulation buffer and an extra 16 bytes to avoid internal ruy copies.
    scratch_buffer_size->data[1] = n_cell * 5 + 16;
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                   scratch_buffer_size));

  if (IsHybridOp(input, input_to_output_weights)) {
    op_data->compute_row_sums = true;
    // Allocate temporary tensors to store quantized values of input,
    // output_state and cell_state tensors.
    node->temporaries->data[kInputQuantized] =
        scratch_tensor_index + kInputQuantized;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kInputQuantized,
                                                &input_quantized));
    input_quantized->type = input_to_output_weights->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    node->temporaries->data[kOutputStateQuantized] =
        scratch_tensor_index + kOutputStateQuantized;
    TfLiteTensor* output_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kOutputStateQuantized,
                                       &output_state_quantized));
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
        scratch_tensor_index + kCellStateQuantized;
    TfLiteTensor* cell_state_quantized;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kCellStateQuantized,
                                       &cell_state_quantized));
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
    TfLiteTensor* input_sf;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, kInputScalingFactors, &input_sf));
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
    TfLiteTensor* output_state_sf;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kOutputStateScalingFactors,
                                  &output_state_sf));
    output_state_sf->type = kTfLiteFloat32;
    output_state_sf->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_sf->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_sf_size = TfLiteIntArrayCreate(1);
      output_state_sf_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_sf,
                                                       output_state_sf_size));
    }
    node->temporaries->data[kProductScalingFactors] =
        scratch_tensor_index + kProductScalingFactors;
    TfLiteTensor* prod_scaling_factors;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kProductScalingFactors,
                                       &prod_scaling_factors));
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
        scratch_tensor_index + kRecoveredCellWeights;
    TfLiteTensor* recovered_cell_weights;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kRecoveredCellWeights,
                                       &recovered_cell_weights));
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

    // Allocate a temporary tensor to store the accumulated int32 values.
    node->temporaries->data[kAccumScratch] =
        scratch_tensor_index + kAccumScratch;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kAccumScratch,
                                                &accum_scratch));
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
    TfLiteTensor* input_zp;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, kInputZeroPoints, &input_zp));
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
    TfLiteTensor* output_state_zp;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kOutputStateZeroPoints,
                                       &output_state_zp));
    output_state_zp->type = kTfLiteFloat32;
    output_state_zp->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(output_state_zp->dims, 1, scaling_dims)) {
      TfLiteIntArray* output_state_zp_size = TfLiteIntArrayCreate(1);
      output_state_zp_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_state_zp,
                                                       output_state_zp_size));
    }
    node->temporaries->data[kRowSums] = scratch_tensor_index + kRowSums;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, kRowSums, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->name = "Lstm_row_sums";
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_rows = use_cifg ? 6 : 8;
    const TfLiteTensor* projection_weights = GetOptionalInputTensor(
        context, node, lstm::full::kProjectionWeightsTensor);
    if (projection_weights != nullptr) {
      row_sums_rows += ceil(static_cast<float>(n_output) / n_cell);
    }
    int row_sums_dims[2] = {row_sums_rows, n_cell};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 2, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(2);
      row_sums_size->data[0] = row_sums_dims[0];
      row_sums_size->data[1] = row_sums_dims[1];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }
  }

  if (is_integer) {
    // Integer UnidirectionalSequenceLSTM prepare function for 8x8->16.
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
      TfLiteTensor* scratch_tensor;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, scratch_index,
                                                  &scratch_tensor));

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
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          node->builtin_data);
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const bool use_layer_norm = op_data->use_layer_norm;
  const bool time_major = params->time_major;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kInputTensor, &input));

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToForgetWeightsTensor,
                   &input_to_forget_weights));
  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node,
                                          lstm::full::kInputToCellWeightsTensor,
                                          &input_to_cell_weights));
  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kInputToOutputWeightsTensor,
                   &input_to_output_weights));

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToForgetWeightsTensor,
                   &recurrent_to_forget_weights));
  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToCellWeightsTensor,
                   &recurrent_to_cell_weights));
  const TfLiteTensor* recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, lstm::full::kRecurrentToOutputWeightsTensor,
                   &recurrent_to_output_weights));

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, lstm::full::kCellToOutputWeightsTensor);

  const TfLiteTensor* input_gate_bias =
      GetOptionalInputTensor(context, node, lstm::full::kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kForgetGateBiasTensor,
                            &forget_gate_bias));
  const TfLiteTensor* cell_gate_bias;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, lstm::full::kCellGateBiasTensor,
                                 &cell_gate_bias));
  const TfLiteTensor* output_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, lstm::full::kOutputGateBiasTensor,
                            &output_gate_bias));

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, lstm::full::kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias =
      GetOptionalInputTensor(context, node, lstm::full::kProjectionBiasTensor);

  TfLiteTensor* output_state =
      GetVariableInput(context, node, lstm::full::kOutputStateTensor);
  TFLITE_DCHECK(output_state != nullptr);
  TfLiteTensor* cell_state =
      GetVariableInput(context, node, lstm::full::kCellStateTensor);
  TFLITE_DCHECK(cell_state != nullptr);

  const TfLiteTensor* input_layer_norm_coefficients =
      use_layer_norm
          ? GetOptionalInputTensor(
                context, node, lstm::full::kInputLayerNormCoefficientsTensor)
          : nullptr;
  const TfLiteTensor* forget_layer_norm_coefficients =
      use_layer_norm ? GetInput(context, node,
                                lstm::full::kForgetLayerNormCoefficientsTensor)
                     : nullptr;
  const TfLiteTensor* cell_layer_norm_coefficients =
      use_layer_norm ? GetInput(context, node,
                                lstm::full::kCellLayerNormCoefficientsTensor)
                     : nullptr;
  const TfLiteTensor* output_layer_norm_coefficients =
      use_layer_norm ? GetInput(context, node,
                                lstm::full::kOutputLayerNormCoefficientsTensor)
                     : nullptr;

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                           lstm::full::kOutputTensor, &output));

  // Copy out the LSTM specific params so they can be passed in the function.
  TfLiteLSTMParams lstm_params;
  lstm_params.activation = params->activation;
  lstm_params.cell_clip = params->cell_clip;
  lstm_params.proj_clip = params->proj_clip;
  lstm_params.asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;

  switch (input_to_output_weights->type) {
    case kTfLiteFloat32: {
      // Index the scratch buffers pointers to the global scratch buffer.
      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, kScratchBuffer,
                                                  &scratch_buffer));
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
          projection_weights, projection_bias, &lstm_params,
          /*forward_sequence=*/true, time_major,
          /*output_offset=*/0, scratch_buffer, output_state, cell_state, output,
          CpuBackendContext::GetFromContext(context));
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      const bool is_hybrid = input->type == kTfLiteFloat32;
      if (is_hybrid) {
        // Index the scratch buffers pointers to the global scratch buffer.
        TfLiteTensor* scratch_buffer;
        TF_LITE_ENSURE_OK(
            context,
            GetTemporarySafe(context, node, kScratchBuffer, &scratch_buffer));

        OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
        TfLiteTensor* row_sums;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, kRowSums, &row_sums));
        const int row_sums_size = row_sums->dims->data[0];
        return lstm_eval::EvalHybrid(
            input, input_to_input_weights,
            /*input_to_input_weights_ledger*/ nullptr, input_to_forget_weights,
            /*input_to_forget_weights_ledger*/ nullptr, input_to_cell_weights,
            /*input_to_cell_weights_ledger*/ nullptr, input_to_output_weights,
            /*input_to_output_weights_ledger*/ nullptr,
            recurrent_to_input_weights,
            /*recurrent_to_input_weights_ledger*/ nullptr,
            recurrent_to_forget_weights,
            /*recurrent_to_forget_weights_ledger*/ nullptr,
            recurrent_to_cell_weights,
            /*recurrent_to_cell_weights_ledger*/ nullptr,
            recurrent_to_output_weights,
            /*recurrent_to_output_weights_ledger*/ nullptr,
            cell_to_input_weights, cell_to_forget_weights,
            cell_to_output_weights, input_layer_norm_coefficients,
            forget_layer_norm_coefficients, cell_layer_norm_coefficients,
            output_layer_norm_coefficients,
            /*aux_input=*/nullptr,
            /*aux_input_to_input_weights=*/nullptr,
            /*aux_input_to_forget_weights=*/nullptr,
            /*aux_input_to_cell_weights=*/nullptr,
            /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
            forget_gate_bias, cell_gate_bias, output_gate_bias,
            projection_weights, /*projection_weights_ledger*/ nullptr,
            projection_bias, &lstm_params,
            /*forward_sequence=*/true, time_major,
            /*output_offset=*/0, scratch_buffer,
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
        TfLiteTensor* scratch0;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 0, &scratch0));
        TfLiteTensor* scratch1;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 1, &scratch1));
        TfLiteTensor* scratch2;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 2, &scratch2));
        TfLiteTensor* scratch3;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 3, &scratch3));
        TfLiteTensor* scratch4;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 4, &scratch4));
        TfLiteTensor* scratch5;
        TF_LITE_ENSURE_OK(context,
                          GetTemporarySafe(context, node, 5, &scratch5));
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
            projection_bias, &lstm_params, /*forward_sequence=*/true,
            time_major, &op_data->integer_lstm_param, output_state, cell_state,
            output, scratch0, scratch1, scratch2, scratch3, scratch4, scratch5,
            CpuBackendContext::GetFromContext(context));
      }
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is not currently supported.",
                         TfLiteTypeGetName(input_to_output_weights->type));
      return kTfLiteError;
  }
}
}  // namespace unidirectional_sequence_lstm

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  static TfLiteRegistration r = {unidirectional_sequence_lstm::Init,
                                 unidirectional_sequence_lstm::Free,
                                 unidirectional_sequence_lstm::Prepare,
                                 unidirectional_sequence_lstm::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
