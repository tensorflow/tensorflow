/******************************************************************************
 * Copyright (C) 2019 Cadence Design Systems, Inc.
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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {
namespace {

// These constants represent constants specific to the hotword "OK G" model.
// They exist until (b/132070898) is fixed.
constexpr int kScratchTensorMaxSize = 64;

struct OpData {
  int32 effective_scale_1_a;
  int32 effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
};

/**
 * This version of SVDF is specific to TFLite Micro. It contains the following
 * differences between the TFLite version:
 *
 * 1.) Scratch tensor allocation - scratch tensors must be known ahead of time
 * for the Micro interpreter.
 * 2.) Output dimensions - the TFLite version determines output size and runtime
 * and resizes the output tensor. Micro runtime does not support tensor
 * resizing.
 */

static inline TfLiteStatus ApplyTimeWeightsBiasAndActivation(
    TfLiteContext* context, int batch_size, int memory_size, int num_filters,
    int num_units, int rank, const float* const __restrict__ weights_time_ptr,
    const float* const __restrict__ bias_ptr, TfLiteFusedActivation activation,
    float* const __restrict__ state_ptr, float* const __restrict__ scratch_ptr,
    float* const __restrict__ output_ptr) {
  // Compute matmul(activation_state, weights_time).
  float* scratch_bias = scratch_ptr;
  if (bias_ptr) {
    const float* bias_data = bias_ptr;
    for (int j = 0; j < num_units; ++j) {
      scratch_bias[j] = *bias_data++;
    }
  } else {
    for (int j = 0; j < num_units; ++j) {
      scratch_bias[j] = 0.0f;
    }
  }
  int err = 0;
  for (int b = 0; b < batch_size; ++b) {
    const float* weights_time_vec = weights_time_ptr;
    const float* mat_ptr = state_ptr + b * memory_size * num_filters;
    float* output_ptr_batch = output_ptr + b * num_units;
    for (int j = 0; j < num_units; j++) {
      err = xa_nn_matXvec_f32xf32_f32(
          output_ptr_batch, mat_ptr, NULL, weights_time_vec, NULL, scratch_bias,
          1, memory_size * rank, 0, memory_size * rank, 0);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_matXvec_f32xf32_f32 failed");

      output_ptr_batch++;
      mat_ptr += memory_size * rank;
      weights_time_vec += memory_size * rank;
    }
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output_ptr + b * num_units;
    for (int i = 0; i < num_units; ++i) {
      *output_ptr_batch = ActivationValFloat(activation, *output_ptr_batch);
      ++output_ptr_batch;
    }
  }
  return kTfLiteOk;
}

inline TfLiteStatus EvalFloatSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* input,
    const TfLiteTensor* weights_feature, const TfLiteTensor* weights_time,
    const TfLiteTensor* bias, const TfLiteSVDFParams* params,
    TfLiteTensor* activation_state, TfLiteTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  const float* weights_feature_ptr = GetTensorData<float>(weights_feature);
  const float* weights_time_ptr = GetTensorData<float>(weights_time);
  const float* bias_ptr = GetTensorData<float>(bias);
  const float* input_ptr = GetTensorData<float>(input);

  float* state_ptr = GetTensorData<float>(activation_state);

  // TODO(b/132070898): Move this temp variable to the new scratch buffer API
  // when ready.
  float scratch_tensor[kScratchTensorMaxSize];
  float* scratch_ptr = scratch_tensor;

  float* output_ptr = GetTensorData<float>(output);

  // Left shift the activation_state.
  {
    float* new_state_start = state_ptr;
    const float* old_state_start = state_ptr + 1;
    const float* old_state_end =
        state_ptr + batch_size * num_filters * memory_size;
    while (old_state_start != old_state_end) {
      *new_state_start++ = *old_state_start++;
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Compute conv1d(inputs, weights_feature).
  // The activation_state's rightmost column is used to save current cycle
  // activation. This is achieved by starting at state_ptr[memory_size - 1] and
  // having the stride equal to memory_size.

  // Perform batched matrix vector multiply operation:
  {
    const float* matrix = weights_feature_ptr;
    const float* vector = input_ptr;
    float* result = &state_ptr[memory_size - 1];
    float* result_in_batch = result;

    float* out_scratch = scratch_ptr;
    float* bias_scratch = output_ptr;
    for (int i = 0; i < num_units; i++) bias_scratch[i] = 0.0f;

    int err = 0;
    for (int i = 0; i < batch_size; i++) {
      /* We are using output buffer for bias (it is needed by NNLib kernel,
      so only num_units size is guaranteed, so introduced rank loop and
      calling matXvec for num_units rows */
      for (int j = 0; j < rank; j++) {
        err = xa_nn_matXvec_f32xf32_f32(
            &out_scratch[j * num_units], &matrix[j * input_size * num_units],
            NULL, &vector[i * input_size], NULL, bias_scratch, num_units,
            input_size, 0, input_size, 0);
        CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_matXvec_f32xf32_f32 failed");
      }
      for (int j = 0; j < num_filters; ++j) {
        *result_in_batch = out_scratch[j];
        result_in_batch += memory_size;
      }
    }
  }

  return ApplyTimeWeightsBiasAndActivation(
      context, batch_size, memory_size, num_filters, num_units, rank,
      weights_time_ptr, bias_ptr, params->activation, state_ptr, scratch_ptr,
      output_ptr);
}

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
    int16_t* state = GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        GetTensorData<int8_t>(weights_feature_tensor);
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_ptr = weight_feature;
      for (int r = 0; r < n_filter; r++) {
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod += *matrix_ptr++ * (*vector_in_batch++ - input_zp);
        }
        dot_prod =
            MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod_56.
        *result_in_batch = dot_prod;
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

      for (int i = 0; i < n_filter; i++) {
        *scratch_ptr_batch = 0;
        for (int j = 0; j < n_memory; j++) {
          *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
        }
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
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = scratch_output_tensor[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      GetTensorData<int8_t>(output_tensor)[i] = static_cast<int8_t>(x4);
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
  const TfLiteTensor* activation_state =
      GetInput(context, node, kInputActivationStateTensor);

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  const bool is_full_integer = input->type == kTfLiteInt8;

  // Validate Input Tensor:
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
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

  if (is_full_integer) {
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
  } else {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

    // Validate Input Tensor dtypes:
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteFloat32);

    if (bias) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteFloat32);
    }

    // Validate shared Scratch Tensor:
    // [0] = Holds dot-product of time-forward calculations in
    //       ApplyTimeWeightsBiasAndActivation():
    //         float/int32, {2, batch_size, num_filters}
    // TODO(b/132070898): Scratch values are used as stack variables in
    // EvalIntegerSVDF().

    // Full-float SVDF only uses the one shared scratch tensor (see above for
    // usage).
    // TODO(b/132070898): Use input tensor as variable until scratch tensor
    // allocation has been implemented.
    // TF_LITE_ENSURE_EQ(context, node->temporaries->size, 1);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* activation_state =
      GetVariableInput(context, node, kInputActivationStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const bool is_full_integer = input->type == kTfLiteInt8;

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      // TODO(b/132070898): Use input tensor as variable until scratch tensor
      // allocation has been implemented.
      // TfLiteTensor* scratch = GetTemporary(context, node, /*index=*/0);
      return EvalFloatSVDF(context, node, input, weights_feature, weights_time,
                           bias, params, activation_state, output);
      break;
    }

    case kTfLiteInt8: {
      if (is_full_integer) {
        // TODO(b/132070898): Store these values in ::Prepare() instead of
        // ::Eval():
        // Calculate effective scales.
        OpData op_data;
        auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
            input->quantization.params);
        auto* weights_feature_params =
            reinterpret_cast<TfLiteAffineQuantization*>(
                weights_feature->quantization.params);
        auto* state_params = reinterpret_cast<TfLiteAffineQuantization*>(
            activation_state->quantization.params);
        auto* weight_time_params = reinterpret_cast<TfLiteAffineQuantization*>(
            weights_time->quantization.params);
        auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
            output->quantization.params);
        const double effective_scale_1 =
            static_cast<double>(input_params->scale->data[0] *
                                weights_feature_params->scale->data[0] /
                                state_params->scale->data[0]);
        const double effective_scale_2 = static_cast<double>(
            state_params->scale->data[0] * weight_time_params->scale->data[0] /
            output_params->scale->data[0]);
        QuantizeMultiplier(effective_scale_1, &op_data.effective_scale_1_a,
                           &op_data.effective_scale_1_b);
        QuantizeMultiplier(effective_scale_2, &op_data.effective_scale_2_a,
                           &op_data.effective_scale_2_b);

        TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);
        EvalIntegerSVDF(
            context, node, input, weights_feature, weights_time, bias, params,
            activation_state, output, op_data.effective_scale_1_a,
            op_data.effective_scale_1_b, op_data.effective_scale_2_a,
            op_data.effective_scale_2_b, input->params.zero_point,
            output->params.zero_point);
        return kTfLiteOk;
      }
      break;
    }

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {/*init=*/nullptr,
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
