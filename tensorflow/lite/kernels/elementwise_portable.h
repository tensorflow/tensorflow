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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace elementwise {
namespace {

constexpr char kAbsName[] = "Abs";
constexpr char kRsqrtName[] = "Rsqrt";

struct OpData {
  int32_t multiplier;
  int shift;
  int input_offset;
  int output_offset;
  bool needs_rescale;
};

bool IsNumericSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
  return type == kTfLiteBool;
}

bool IsAbsSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8 || type == kTfLiteInt16;
}

inline void SetAbsOutputMultiplier(const float input_scale,
                                   const float output_scale,
                                   int32_t* multiplier, int* shift) {
  QuantizeMultiplier(static_cast<double>(input_scale / output_scale),
                     multiplier, shift);
}

inline void SetRsqrtOutputMultiplier(const float input_scale,
                                     const float output_scale,
                                     int32_t* multiplier, int32_t* shift) {
  const double scale =
      1. / static_cast<double>((std::sqrt(input_scale) * output_scale));
  QuantizeMultiplier(scale, multiplier, shift);
}

typedef bool (*IsSupportedType)(TfLiteType);
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node,
                            IsSupportedType is_supported_type,
                            const char* op_name, bool resize = false) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!is_supported_type(input->type)) {
    TF_LITE_UNSUPPORTED_TYPE(context, input->type, op_name);
  }
  // For int16 type input, we support both quantized and non-quantized
  // evaluation.
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       input->quantization.type != kTfLiteNoQuantization)) {
    // TfLiteTensor* output = GetOutput(context, node, 0);
    auto* op_data = static_cast<OpData*>(node->user_data);
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);
    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);
    op_data->input_offset = input_params->zero_point->data[0];
    op_data->output_offset = output_params->zero_point->data[0];
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, op_data->input_offset, 0);
      TF_LITE_ENSURE_EQ(context, op_data->output_offset, 0);
    }
    const float input_scale = input_params->scale->data[0];
    const float output_scale = output_params->scale->data[0];
    op_data->needs_rescale = input_scale != output_scale;
    if (op_name == kAbsName && op_data->needs_rescale) {
      SetAbsOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                             &op_data->shift);
    } else if (op_name == kRsqrtName) {
      SetRsqrtOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                               &op_data->shift);
    }
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace elementwise
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
