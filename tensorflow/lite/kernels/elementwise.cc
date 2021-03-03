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

#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace elementwise {
namespace {

constexpr char kAbsName[] = "Abs";
constexpr char kSinName[] = "Sin";
constexpr char kCosName[] = "Cos";
constexpr char kLogName[] = "Log";
constexpr char kSqrtName[] = "Sqrt";
constexpr char kRsqrtName[] = "Rsqrt";
constexpr char kSquareName[] = "Square";
constexpr char kNotName[] = "Not";

struct OpData {
  int32_t multiplier;
  int32_t shift;
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

bool IsRsqrtSupportedType(const TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

inline void SetAbsOutputMultiplier(const float input_scale,
                                   const float output_scale,
                                   int32_t* multiplier, int32_t* shift) {
  QuantizeMultiplier(input_scale / output_scale, multiplier, shift);
}

inline void SetRsqrtOutputMultiplier(const float input_scale,
                                     const float output_scale,
                                     int32_t* multiplier, int32_t* shift) {
  const double scale = 1. / (std::sqrt(input_scale) * output_scale);
  QuantizeMultiplier(scale, multiplier, shift);
}

typedef bool (*IsSupportedType)(TfLiteType);
template <IsSupportedType is_supported_type, const char* op_name>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
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
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TfLiteTensor* output = GetOutput(context, node, 0);
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
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             std::function<T(T)> func,
                             std::function<TfLiteStatus(T)> validate_input_func,
                             TfLiteType expected_type) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const int64_t num_elements = NumElements(input);
  const T* in_data = GetTensorData<T>(input);
  T* out_data = GetTensorData<T>(output);
  for (int64_t i = 0; i < num_elements; ++i) {
    if (validate_input_func) {
      TF_LITE_ENSURE_OK(context, validate_input_func(in_data[i]));
    }
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             std::function<T(T)> func,
                             TfLiteType expected_type) {
  return EvalImpl<T>(context, node, func, /*validate_input_func=*/nullptr,
                     expected_type);
}

inline TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}

inline TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}

void* ElementWiseQuantizedInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  return new OpData();
}

void ElementWiseQuantizedFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

template <typename T>
TfLiteStatus AbsEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteType type) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int kMin = std::numeric_limits<T>::min();
  const int kMax = std::numeric_limits<T>::max();

  std::function<T(T)> func = [&](T i) {
    const int32_t value = std::abs(i - op_data->input_offset);
    if (!op_data->needs_rescale) {
      return static_cast<T>(
          std::min(std::max(value + op_data->output_offset, kMin), kMax));
    }
    const int32_t output = MultiplyByQuantizedMultiplier(
                               value, op_data->multiplier, op_data->shift) +
                           op_data->output_offset;
    return static_cast<T>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<T>(context, node, func, type);
}

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteType type = GetInput(context, node, 0)->type;
  switch (type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(context, node, std::abs<float>, type);
    case kTfLiteInt8:
      return AbsEvalQuantized<int8_t>(context, node, type);
    case kTfLiteInt16:
      return AbsEvalQuantized<int16_t>(context, node, type);
    default:
      TF_LITE_KERNEL_LOG(context, "Current data type %s is not supported.",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sin);
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::cos);
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::log);
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, std::sqrt);
}

TfLiteStatus RsqrtEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                                TfLiteType type) {
  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int kMin = std::numeric_limits<int8_t>::min();
  const int kMax = std::numeric_limits<int8_t>::max();
  std::function<TfLiteStatus(int8_t)> validate_input_func = [&](int8_t i) {
    TF_LITE_ENSURE_MSG(context, i >= op_data->input_offset,
                       "Rsqrt is only defined for positive values");
    return kTfLiteOk;
  };

  std::function<int8_t(int8_t)> func = [&](int8_t i) {
    const int32_t value = (i - op_data->input_offset);
    const int32_t kShift = 20;  // Shift to keep value integer.
    if (value == 0) {
      // Assume that any value close to 0 represents the max output value.
      return static_cast<int8_t>(kMax);
    }
    int32_t inv_sqrt_multiplier;
    int inv_sqrt_shift;
    GetInvSqrtQuantizedMultiplierExp(value, kReverseShift, &inv_sqrt_multiplier,
                                     &inv_sqrt_shift);
    const int32_t data = MultiplyByQuantizedMultiplier(1, inv_sqrt_multiplier,
                                                       inv_sqrt_shift + kShift);
    const int32_t output =
        MultiplyByQuantizedMultiplier(data, op_data->multiplier,
                                      op_data->shift - kShift) +
        op_data->output_offset;
    return static_cast<int8_t>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<int8_t>(context, node, func, validate_input_func, type);
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteType type = GetInput(context, node, 0)->type;
  switch (type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(
          context, node, [](float f) { return 1.f / std::sqrt(f); }, type);
    case kTfLiteInt8:
      return RsqrtEvalQuantized(context, node, type);
    default:
      TF_LITE_KERNEL_LOG(context, "Current data type %s is not supported.",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalNumeric(context, node, [](float f) { return f * f; });
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
  return EvalLogical(context, node, [](bool v) { return !v; });
}

}  // namespace
}  // namespace elementwise

TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {
      elementwise::ElementWiseQuantizedInit,
      elementwise::ElementWiseQuantizedFree,
      elementwise::GenericPrepare<elementwise::IsAbsSupportedType,
                                  elementwise::kAbsName>,
      elementwise::AbsEval};
  return &r;
}

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType,
                                  elementwise::kSinName>,
      elementwise::SinEval};
  return &r;
}

TfLiteRegistration* Register_COS() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType,
                                  elementwise::kCosName>,
      elementwise::CosEval};
  return &r;
}

TfLiteRegistration* Register_LOG() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType,
                                  elementwise::kLogName>,
      elementwise::LogEval};
  return &r;
}

TfLiteRegistration* Register_SQRT() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType,
                                  elementwise::kSqrtName>,
      elementwise::SqrtEval};
  return &r;
}

TfLiteRegistration* Register_RSQRT() {
  static TfLiteRegistration r = {
      elementwise::ElementWiseQuantizedInit,
      elementwise::ElementWiseQuantizedFree,
      elementwise::GenericPrepare<elementwise::IsRsqrtSupportedType,
                                  elementwise::kRsqrtName>,
      elementwise::RsqrtEval};
  return &r;
}

TfLiteRegistration* Register_SQUARE() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsNumericSupportedType,
                                  elementwise::kSquareName>,
      elementwise::SquareEval};
  return &r;
}

TfLiteRegistration* Register_LOGICAL_NOT() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr,
      elementwise::GenericPrepare<elementwise::IsLogicalSupportedType,
                                  elementwise::kNotName>,
      elementwise::LogicalNotEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
