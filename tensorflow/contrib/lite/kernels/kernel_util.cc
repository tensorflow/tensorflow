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
#include "tensorflow/contrib/lite/kernels/kernel_util.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/contrib/lite/kernels/internal/round.h"

namespace tflite {

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = input->params.scale * filter->params.scale;
  const double bias_scale = bias->params.scale;
  const double output_scale = output->params.scale;

  // TODO(ahentz): The following conditions must be guaranteed by the training
  // pipeline.
  TF_LITE_ENSURE(context, std::abs(input_product_scale - bias_scale) <=
                              1e-6 * std::min(input_product_scale, bias_scale));
  TF_LITE_ENSURE(context, input_product_scale >= 0);

  *multiplier = input_product_scale / output_scale;

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

void CalculateActivationRangeFloat(TfLiteFusedActivation activation,
                                   float* activation_min,
                                   float* activation_max) {
  if (activation == kTfLiteActRelu) {
    *activation_min = 0.f;
    *activation_max = std::numeric_limits<float>::max();
  } else if (activation == kTfLiteActRelu6) {
    *activation_min = 0.f;
    *activation_max = 6.f;
  } else if (activation == kTfLiteActRelu1) {
    *activation_min = -1.f;
    *activation_max = 1.f;
  } else {
    *activation_min = std::numeric_limits<float>::lowest();
    *activation_max = std::numeric_limits<float>::max();
  }
}

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape) {
  int64_t dims1 = NumDimensions(input1);
  int64_t dims2 = NumDimensions(input2);
  int64_t out_dims = std::max(dims1, dims2);
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

}  // namespace tflite
