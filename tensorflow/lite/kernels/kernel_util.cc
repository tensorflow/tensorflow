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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/round.h"

namespace tflite {

void GuardedQuantizeMultiplier(double effective_output_scale,
                               int32_t* significand, int* shift) {
  QuantizeMultiplier(effective_output_scale, significand, shift);
  // Additional guard to make sure RoundingDivideByPOT does not fail.
  if (*shift < -31) {
    // If shift is less than -31, RoundingDivideByPOT fails. This happens when
    // min and max are close and small. For this particular case, both
    // significand and shift are set to zero.
    *significand = 0;
    *shift = 0;
  }
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift) {
  TF_LITE_ENSURE_EQ(context, input->quantization.type,
                    kTfLiteAffineQuantization);
  TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                    kTfLiteAffineQuantization);
  // TODO(jianlijianli): Enable bias type check and bias scale == input scale
  // * filter scale for each channel in affine quantization once bias
  // quantization is properly populated.
  // TF_LITE_ENSURE_EQ(context, bias->quantization.type,
  // kTfLiteAffineQuantization);

  // Check data type.
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  const bool is_per_channel = affine_quantization->scale->size > 1;
  if (is_per_channel) {
    //  Currently only Int8 is supported for per channel quantization.
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(
        context, affine_quantization->scale->size,
        filter->dims->data[affine_quantization->quantized_dimension]);
  }

  // Populate multiplier and shift using affine quantization.
  const int num_channels = affine_quantization->scale->size;
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; ++i) {
    const double filter_scale = static_cast<double>(filter_scales[i]);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int shift;
    GuardedQuantizeMultiplier(effective_output_scale, &significand, &shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = shift;
  }

  // Populate scalar quantization parameters.
  // This check on legacy quantization parameters is kept only for backward
  // compatibility.
  if (input->type == kTfLiteUInt8) {
    // Check bias scale == input scale * filter scale.
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;

    // Populate quantization parameteters with multiplier and shift.
    QuantizeMultiplier(real_multiplier, multiplier, &exponent);
    *shift = -exponent;
    CalculateActivationRangeUint8(activation, output, output_activation_min,
                                  output_activation_max);
  }
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = input->params.scale * filter->params.scale;
  // TODO(ahentz): The following conditions must be guaranteed by the training
  // pipeline.
  if (bias) {
    const double bias_scale = bias->params.scale;
    TF_LITE_ENSURE(context,
                   std::abs(input_product_scale - bias_scale) <=
                       1e-6 * std::min(input_product_scale, bias_scale));
  }
  return GetQuantizedConvolutionMultipler(context, input, filter, output,
                                          multiplier);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = input->params.scale * filter->params.scale;
  TF_LITE_ENSURE(context, input_product_scale >= 0);
  *multiplier = input_product_scale / output->params.scale;

  return kTfLiteOk;
}

namespace {
void CalculateActivationRangeQuantizedImpl(TfLiteFusedActivation activation,
                                           int32_t qmin, int32_t qmax,
                                           TfLiteTensor* output,
                                           int32_t* act_min, int32_t* act_max) {
  const auto scale = output->params.scale;
  const auto zero_point = output->params.zero_point;

  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(TfLiteRound(f / scale));
  };

  if (activation == kTfLiteActRelu) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  } else if (activation == kTfLiteActRelu6) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  } else if (activation == kTfLiteActRelu1) {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
}
}  // namespace

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  if (output->type == kTfLiteUInt8) {
    qmin = std::numeric_limits<uint8_t>::min();
    qmax = std::numeric_limits<uint8_t>::max();
  } else if (output->type == kTfLiteInt8) {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
  } else if (output->type == kTfLiteInt16) {
    qmin = std::numeric_limits<int16_t>::min();
    qmax = std::numeric_limits<int16_t>::max();
  } else {
    TF_LITE_ENSURE(context, false);
  }

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
  return kTfLiteOk;
}

void CalculateActivationRangeUint8(TfLiteFusedActivation activation,
                                   TfLiteTensor* output, int32_t* act_min,
                                   int32_t* act_max) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
}

void CalculateActivationRangeInt8(TfLiteFusedActivation activation,
                                  TfLiteTensor* output, int32_t* act_min,
                                  int32_t* act_max) {
  const int32_t qmin = std::numeric_limits<int8_t>::min();
  const int32_t qmax = std::numeric_limits<int8_t>::max();

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
}

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

// TODO(petewarden): Having macros around this is ugly, look at other strategies
// before replicating this approach elsewhere.
#ifndef TF_LITE_STATIC_MEMORY
TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape) {
  int64_t dims1 = NumDimensions(input1);
  int64_t dims2 = NumDimensions(input2);
  int64_t out_dims = std::max(dims1, dims2);
  if (NumElements(input1) == 0) {
    *output_shape = TfLiteIntArrayCopy(input1->dims);
    return kTfLiteOk;
  }
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(out_dims), TfLiteIntArrayFree);
  for (int i = 0; i < out_dims; ++i) {
    int64_t d1 = i >= dims1 ? 1 : SizeOfDimension(input1, dims1 - i - 1);
    int64_t d2 = i >= dims2 ? 1 : SizeOfDimension(input2, dims2 - i - 1);
    TF_LITE_ENSURE(context, d1 == d2 || d1 == 1 || d2 == 1);
    shape->data[out_dims - i - 1] = std::max(d1, d2);
  }
  *output_shape = shape.release();
  return kTfLiteOk;
}
#endif  // TF_LITE_STATIC_MEMORY

}  // namespace tflite
