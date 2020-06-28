/*******************************************************************************
 * Copyright (c) 2020 Cadence Design Systems, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use this Software with Cadence processor cores only and
 * not with any other processors and platforms, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

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

#include <math.h>
#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini/xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {
namespace {

struct OpData {
  int32 effective_scale_1_a;
  int32 effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
  int scratch_tensor_index;
  int scratch_output_tensor_index;
};

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

/**
 * This version of SVDF is specific to TFLite Micro. It contains only a full
 * integer receipe with optimizations for the Xtensa HiFiMini platform.
 *
 * Note: passing OpData by value might seem like an oversight but it helps
 * reduce the latency. See b/155656675 for more details.
 */
TfLiteStatus EvalIntegerSVDF(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* input_tensor,
                             const TfLiteTensor* weights_feature_tensor,
                             const TfLiteTensor* weights_time_tensor,
                             const TfLiteTensor* bias_tensor,
                             const TfLiteSVDFParams* params,
                             TfLiteTensor* activation_state_tensor,
                             TfLiteTensor* output_tensor, OpData data,
                             int32_t input_zp, int32_t output_zp) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  int32_t* scratch_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));
  TFLITE_DCHECK(scratch_tensor != nullptr);
  int32_t* scratch_output_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_output_tensor_index));
  TFLITE_DCHECK(scratch_output_tensor != nullptr);

  // Shift states.
  int16_t* const state_ptr = GetTensorData<int16_t>(activation_state_tensor);

  // Left shift the activation_state.

  // 4-byte alignment check for state_ptr
  if (((reinterpret_cast<int>(state_ptr)) & 0x3) == 0) {
    // 4-bytes aligned processing
    ae_p16x2s* new_state_start = (ae_p16x2s*)(state_ptr - 2);
    const ae_p16x2s* old_state_start = (ae_p16x2s*)(state_ptr - 2);
    int loopcnt = (n_batch * n_filter * n_memory) - 1;
    ae_p24x2s dstate, dtmp, dout;

    AE_LP16X2F_IU(dtmp, old_state_start, 4);
    AE_LP16X2F_IU(dstate, old_state_start, 4);
    for (int i = 0; i < (loopcnt >> 1); i++) {
      dout = AE_SELP24_LH(dtmp, dstate);
      dtmp = dstate;
      AE_LP16X2F_IU(dstate, old_state_start, 4);
      AE_SP16X2F_IU(dout, new_state_start, 4);
    }
    if (loopcnt & 0x1) {
      AE_SP16F_L_I(dtmp, (ae_p16s*)new_state_start, 4);
    }
  } else {
    // 2-bytes aligned processing
    ae_p16s* new_state_start = (ae_p16s*)(state_ptr - 1);
    const ae_p16s* old_state_start = (ae_p16s*)(state_ptr);
    int loopcnt = (n_batch * n_filter * n_memory) - 1;
    ae_p24x2s dstate;
    for (int i = 0; i < loopcnt; i++) {
      AE_LP16F_IU(dstate, old_state_start, 2);
      AE_SP16F_L_IU(dstate, new_state_start, 2);
    }
  }
  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Feature matmul.
  {
    int16_t* state = GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    int16_t* result_in_batch = state + (n_memory - 1);
    int err = 0;

    for (int b = 0; b < n_batch; b++) {
      err = xa_nn_matXvec_out_stride_sym8sxasym8s_16(
          &result_in_batch[b * n_filter * n_memory], weight_feature,
          &input[b * n_input], NULL, n_filter, n_input, n_input, n_memory,
          -input_zp, (data.effective_scale_1_a << 8), data.effective_scale_1_b);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_matXvec_sym8sxasym8s_16 failed");
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      int8_t* output_ptr = GetTensorData<int8_t>(output_tensor) + b * n_unit;

      const int16_t* vector1_ptr = GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      int err = 0;
      const int32_t* bias_ptr = GetTensorData<int32_t>(bias_tensor);
      err = xa_nn_dot_prod_16x16_asym8s(
          output_ptr, vector1_ptr, vector2_ptr, bias_ptr, n_memory * n_rank,
          (data.effective_scale_2_a << 8), data.effective_scale_2_b, output_zp,
          n_unit);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_dot_prod_16x16_asym8s failed");
    }
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params = static_cast<const TfLiteSVDFParams*>(node->builtin_data);

  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  const TfLiteTensor* activation_state =
      GetInput(context, node, kInputActivationStateTensor);

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  // Ensure the input size is a multiple of two.  This is necessary since
  // optimized kernels access the memory in chunks of two, and all accesses
  // must be aligned to 16 bits.
  // TODO(b/153202598): Remove when padding is allowed in TFLite tensors.
  TF_LITE_ENSURE_EQ(context, input_size % 2, 0);

  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  if (input->type != kTfLiteInt8) {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  // Validate Input Tensor:
  TF_LITE_ENSURE(context, input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  // Validate Tensor Output:
  // [0] = float/int8, {2, batch_size, num_units}
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 2);
  TF_LITE_ENSURE_EQ(context, output->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, output->dims->data[1], num_units);

  // Validate Weights Feature Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_feature), 2);
  TF_LITE_ENSURE_EQ(context, weights_feature->dims->data[1], input_size);

  // Validate Weights Time Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_time), 2);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[1], memory_size);

  // Validate Optional Bias Input Tensor:
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
    TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteInt16);
  TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteInt16);

  // Validate output tensor:
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);

  // Calculate effective scales.
  auto* input_params =
      static_cast<TfLiteAffineQuantization*>(input->quantization.params);
  auto* weights_feature_params = static_cast<TfLiteAffineQuantization*>(
      weights_feature->quantization.params);
  auto* state_params = static_cast<TfLiteAffineQuantization*>(
      activation_state->quantization.params);
  auto* weight_time_params =
      static_cast<TfLiteAffineQuantization*>(weights_time->quantization.params);
  auto* output_params =
      static_cast<TfLiteAffineQuantization*>(output->quantization.params);
  const float effective_scale_1 = input_params->scale->data[0] *
                                  weights_feature_params->scale->data[0] /
                                  state_params->scale->data[0];
  const float effective_scale_2 = state_params->scale->data[0] *
                                  weight_time_params->scale->data[0] /
                                  output_params->scale->data[0];

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  xtensa::hifimini::QuantizeMultiplier(effective_scale_1,
                                       &data->effective_scale_1_a,
                                       &data->effective_scale_1_b);
  xtensa::hifimini::QuantizeMultiplier(effective_scale_2,
                                       &data->effective_scale_2_a,
                                       &data->effective_scale_2_b);

  const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
      context, batch_size * num_filters * sizeof(int32_t),
      &(data->scratch_tensor_index));
  TF_LITE_ENSURE_OK(context, scratch_status);
  const TfLiteStatus scratch_output_status =
      context->RequestScratchBufferInArena(
          context, batch_size * num_units * sizeof(int32_t),
          &(data->scratch_output_tensor_index));
  TF_LITE_ENSURE_OK(context, scratch_output_status);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* activation_state =
      GetVariableInput(context, node, kInputActivationStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalIntegerSVDF(context, node, input, weights_feature, weights_time,
                         bias, params, activation_state, output, data,
                         input->params.zero_point, output->params.zero_point);
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {/*init=*/svdf::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/svdf::Prepare,
                                 /*invoke=*/svdf::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
