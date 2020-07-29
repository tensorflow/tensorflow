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
#include "tensorflow/lite/micro/kernels/xtensa_hifimini_legacy/fixedpoint_utils.h"

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
void EvalIntegerSVDF(TfLiteContext* context, TfLiteNode* node,
                     const TfLiteTensor* input_tensor,
                     const TfLiteTensor* weights_feature_tensor,
                     const TfLiteTensor* weights_time_tensor,
                     const TfLiteTensor* bias_tensor,
                     const TfLiteSVDFParams* params,
                     TfLiteTensor* activation_state_tensor,
                     TfLiteTensor* output_tensor, OpData data, int32_t input_zp,
                     int32_t output_zp) {
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
  {
    int16_t* new_state_start = state_ptr;
    const int16_t* old_state_start = state_ptr + 1;
    const int16_t* old_state_end = state_ptr + n_batch * n_filter * n_memory;
    while (old_state_start != old_state_end) {
      *new_state_start++ = *old_state_start++;
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Feature matmul.
  {
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    int16_t* result_in_batch = state_ptr + (n_memory - 1);

    ae_q56s output_int16_max_56 = AE_CVTQ48A32S(INT16_MAX);
    ae_q56s output_int16_min_56 = AE_CVTQ48A32S(INT16_MIN);
    ae_p24x2s input_zp_24x2 = AE_MOVPA24(input_zp);

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
                dot_prod_24x2, data.effective_scale_1_a,
                data.effective_scale_1_b);

        // Cap min/max and convert to int32:
        dot_prod_56 = AE_MAXQ56S(dot_prod_56, output_int16_min_56);
        dot_prod_56 = AE_MINQ56S(dot_prod_56, output_int16_max_56);
        // Truncate immediately since the QR register is already 32 bit aligned:
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod_56.
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
      const int16_t* vector2_ptr = state_ptr + b * n_memory * n_filter;

      const ae_p16x2s* offset_vector1 =
          reinterpret_cast<const ae_p16x2s*>(vector1_ptr - 2);
      const ae_p16x2s* offset_vector2 =
          reinterpret_cast<const ae_p16x2s*>(vector2_ptr - 2);

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
              scratch_output_tensor[i], data.effective_scale_2_a,
              data.effective_scale_2_b);
      // Add output adjustment:
      x_56 = AE_ADDQ56(x_56, output_zp_56);
      // Cap min/max and convert to int32 (already aligned to 32bit):
      x_56 = AE_MAXQ56S(x_56, output_int8_min_56);
      x_56 = AE_MINQ56S(x_56, output_int8_max_56);
      GetTensorData<int8_t>(output_tensor)[i] =
          static_cast<int8_t>(AE_TRUNCA32Q48(x_56));
    }
  }
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
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

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

  EvalIntegerSVDF(context, node, input, weights_feature, weights_time, bias,
                  params, activation_state, output, data,
                  input->params.zero_point, output->params.zero_point);
  return kTfLiteOk;
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
