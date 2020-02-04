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
#include "tensorflow/lite/micro/kernels/xtensa-hifimini/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa-hifimini/utils.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {
namespace {

// These constants represent constants specific to the hotword "OK G" model.
// They exist until (b/132070898) is fixed.
constexpr int kScratchTensorMaxSize = 64;
constexpr int kMaxOpDataSize = 7;

struct OpData {
  int32 effective_scale_1_a;
  int32 effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
};

static int kStaticOpDataCounter = 0;
static OpData kStaticOpData[kMaxOpDataSize];

/**
 * This version of SVDF is specific to TFLite Micro. It contains only a full
 * integer receipe with optimizations for the Xtensa HiFiMini platform.
 */

void EvalIntegerSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input_tensor,
    const TfLiteTensor* weights_feature_tensor,
    const TfLiteTensor* weights_time_tensor, const TfLiteTensor* bias_tensor,
    const TfLiteSVDFParams* params, TfLiteTensor* activation_state_tensor,
    TfLiteTensor* output_tensor, int32_t scale_1_a, int scale_1_b,
    int32_t scale_2_a, int scale_2_b, int32_t input_zp, int32_t output_zp) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  // TODO(b/132070898): Move these temp variables to the new scratch buffer API
  // when ready.
  int32_t scratch_tensor[kScratchTensorMaxSize];
  int32_t scratch_output_tensor[kScratchTensorMaxSize];

  // Rewrite last bit of state.
  {
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      for (int c = 0; c < n_filter; ++c) {
        int16_t* state_ptr = state_ptr_batch + c * n_memory;
        state_ptr[n_memory - 1] = 0;
      }
    }
  }

  // Feature matmul.
  {
    int16_t* state = GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    int16_t* result_in_batch = state + (n_memory - 1);

    ae_q56s output_int16_max_56 = AE_CVTQ48A32S(INT16_MAX);
    ae_q56s output_int16_min_56 = AE_CVTQ48A32S(INT16_MIN);
    ae_p24x2s input_zp_24x2 = AE_CONVERT_INT32_24x2(input_zp);

    for (int b = 0; b < n_batch; b++) {
      const int8_t* weight_feature_ptr = weight_feature - 2;

      for (int r = 0; r < n_filter; r++) {
        ae_q56s dot_prod_56 = AE_ZEROQ56();

        const int8_t* input_batch_ptr = input + b * n_input;
        const int8_t* offset_input_batch_ptr = input_batch_ptr - 2;

        int num_iters = n_input / 2;
        for (int c = 0; c < num_iters; c++) {
          // Load 2 sets of values:
          ae_p24x2s weight_feature_ptr_24x2;
          ae_p24x2s input_batch_ptr_24x2;
          AE_LP8X2F_IU(weight_feature_ptr_24x2, weight_feature_ptr, 2);
          AE_LP8X2F_IU(input_batch_ptr_24x2, offset_input_batch_ptr, 2);

          // Right shift the signed 8bit values to expand to signed 24bit
          // values:
          weight_feature_ptr_24x2 = AE_P24X2S_SRAI(weight_feature_ptr_24x2, 16);
          input_batch_ptr_24x2 = AE_P24X2S_SRAI(input_batch_ptr_24x2, 16);

          // First subtract input_zp from input_batch_ptr_24x2:
          input_batch_ptr_24x2 =
              AE_SUBSP24S(input_batch_ptr_24x2, input_zp_24x2);

          // Multiply accum:
          AE_MULAAP24S_HH_LL(dot_prod_56, weight_feature_ptr_24x2,
                             input_batch_ptr_24x2);
        }

        // Left shift 48bit value into 24bit space and place on the PR register:
        dot_prod_56 = AE_Q56S_SLAI(dot_prod_56, 24);
        ae_p24x2s dot_prod_24x2 = AE_TRUNCP24Q48(dot_prod_56);

        dot_prod_56 =
            tflite::ops::micro::xtensa::hifimini::MultiplyByQuantizedMultiplier(
                dot_prod_24x2, scale_1_a, scale_1_b);

        // Align from 48bit to 32bit on the QR register
        dot_prod_56 = AE_Q56S_SLAI(dot_prod_56, 16);
        // Cap min/max and convert to int32:
        dot_prod_56 = AE_MAXQ56S(dot_prod_56, output_int16_min_56);
        dot_prod_56 = AE_MINQ56S(dot_prod_56, output_int16_max_56);
        // Truncate immediately since the QR register is already 32 bit aligned:
        *result_in_batch = AE_TRUNCA32Q48(dot_prod_56);
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Perform batched vector dot product:
      const int16_t* vector1_ptr = GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;

      int num_iters = n_filter / 2;
      const ae_p16x2s* offset_vector1 = (const ae_p16x2s*)(vector1_ptr - 2);
      const ae_p16x2s* offset_vector2 = (const ae_p16x2s*)(vector2_ptr - 2);

      for (int i = 0; i < n_filter; i++) {
        *scratch_ptr_batch = 0;

        ae_q56s sum_56 = AE_ZEROQ56();
        int num_iters = n_memory / 2;
        for (int j = 0; j < num_iters; j++) {
          ae_p24x2s vector1_24x2;
          ae_p24x2s vector2_24x2;
          AE_LP16X2F_IU(vector1_24x2, offset_vector1, 4);
          AE_LP16X2F_IU(vector2_24x2, offset_vector2, 4);
          AE_MULAAP24S_HH_LL(sum_56, vector1_24x2, vector2_24x2);
        }
        // Truncate directly since values are already 32bit aligned:
        *scratch_ptr_batch = AE_TRUNCA32Q48(sum_56);
        scratch_ptr_batch++;
      }
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Add bias.
    if (bias_tensor) {
      // Vector batch assign:
      const int32_t* bias_data = GetTensorData<int32_t>(bias_tensor);
      for (int i = 0; i < n_batch; ++i) {
        int32_t* output_ptr = scratch_output_tensor + i * n_unit;
        const int32_t* bias_ptr = bias_data;
        for (int j = 0; j < n_unit; ++j) {
          *output_ptr++ = *bias_ptr++;
        }
      }
    } else {
      int32_t* output_ptr = scratch_output_tensor;
      for (int i = 0; i < n_batch * n_unit; ++i) {
        *output_ptr++ = 0;
      }
    }

    // Reduce.
    for (int b = 0; b < n_batch; ++b) {
      int32_t* output_temp_ptr = scratch_output_tensor + b * n_unit;
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Reduction sum vector
      for (int i = 0; i < n_unit; ++i) {
        for (int j = 0; j < n_rank; ++j) {
          output_temp_ptr[i] += *scratch_ptr_batch++;
        }
      }
    }

    // Rescale.
    ae_q56s output_int8_max_56 = AE_CVTQ48A32S(INT8_MAX);
    ae_q56s output_int8_min_56 = AE_CVTQ48A32S(INT8_MIN);
    ae_q56s output_zp_56 = AE_CVTQ48A32S(output_zp);
    for (int i = 0; i < n_batch * n_unit; ++i) {
      ae_q56s x_56 =
          tflite::ops::micro::xtensa::hifimini::MultiplyByQuantizedMultiplier(
              scratch_output_tensor[i], scale_2_a, scale_2_b);
      // Align from 48bit to 32bit on the QR register:
      x_56 = AE_Q56S_SLAI(x_56, 16);
      // Add output adjustment:
      x_56 = AE_ADDQ56(x_56, output_zp_56);
      // Cap min/max and convert to int32 (already aligned to 32bit):
      x_56 = AE_MAXQ56S(x_56, output_int8_min_56);
      x_56 = AE_MINQ56S(x_56, output_int8_max_56);
      int32_t x_32 = AE_TRUNCA32Q48(x_56);
      GetTensorData<int8_t>(output_tensor)[i] =
          static_cast<int8_t>(AE_TRUNCA32Q48(x_56));
    }
  }

  // Shift state.
  {
    for (int b = 0; b < n_batch; ++b) {
      int16_t* state_ptr_batch =
          GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      for (int f = 0; f < n_filter; ++f) {
        // Shift the vector left:
        int16_t* batch_ptr = state_ptr_batch;
        int16_t* batch_start = state_ptr_batch + 1;
        int16_t* batch_end = state_ptr_batch + n_memory;
        while (batch_start != batch_end) {
          *batch_ptr++ = *batch_start++;
        }
        state_ptr_batch[n_memory - 1] = 0;
        state_ptr_batch += n_memory;
      }
    }
  }
}

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

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
  TfLiteTensor* activation_state =
      &context->tensors[node->inputs->data[kInputActivationStateTensor]];

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  if (input->type != kTfLiteInt8) {
    context->ReportError(context,
                         "HiFi Mini kernel SVDF only supports full integer.");
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
  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteInt16);

  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
  }

  TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteInt16);

  // Validate Scratch Tensors:
  // [0] = (shared - see float block below for usage)
  // [1] = Output Temp, int8_t, {2, num_units, batch_size}
  // TODO(b/132070898): Scratch values are used as stack variables in
  // EvalIntegerSVDF().

  // Validate output tensor:
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);

  // TODO(b/132070898): Use statically slotted OpData structures until a
  // scratch memory API is ready.
  OpData* op_data = &kStaticOpData[kStaticOpDataCounter++];
  node->user_data = op_data;

  // Calculate effective scales.
  auto* input_params =
      reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
  auto* weights_feature_params = reinterpret_cast<TfLiteAffineQuantization*>(
      weights_feature->quantization.params);
  auto* state_params = reinterpret_cast<TfLiteAffineQuantization*>(
      activation_state->quantization.params);
  auto* weight_time_params = reinterpret_cast<TfLiteAffineQuantization*>(
      weights_time->quantization.params);
  auto* output_params =
      reinterpret_cast<TfLiteAffineQuantization*>(output->quantization.params);
  const double effective_scale_1 = input_params->scale->data[0] *
                                   weights_feature_params->scale->data[0] /
                                   state_params->scale->data[0];
  const double effective_scale_2 = state_params->scale->data[0] *
                                   weight_time_params->scale->data[0] /
                                   output_params->scale->data[0];
  xtensa::hifimini::QuantizeMultiplier(effective_scale_1,
                                       &op_data->effective_scale_1_a,
                                       &op_data->effective_scale_1_b);
  xtensa::hifimini::QuantizeMultiplier(effective_scale_2,
                                       &op_data->effective_scale_2_a,
                                       &op_data->effective_scale_2_b);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* activation_state =
      &context->tensors[node->inputs->data[kInputActivationStateTensor]];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);

  EvalIntegerSVDF(context, node, input, weights_feature, weights_time, bias,
                  params, activation_state, output,
                  op_data->effective_scale_1_a, op_data->effective_scale_1_b,
                  op_data->effective_scale_2_a, op_data->effective_scale_2_b,
                  input->params.zero_point, output->params.zero_point);
  return kTfLiteOk;
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {svdf::Init, svdf::Free, svdf::Prepare,
                                 svdf::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
