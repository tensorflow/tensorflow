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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace comparisons {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ComparisonPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Don't support string and bool.
  TF_LITE_ENSURE(context,
                 input1->type != kTfLiteString || input1->type != kTfLiteBool);
  // Currently only support tensors have the same type.
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = kTfLiteBool;

  bool requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

// TODO(ruic): optimize macros below to using template functions.
#define TF_LITE_QUANTIZE_COMPARISON(opname)                                    \
  template <typename input_dtype>                                              \
  void EvalQuantized##opname(TfLiteContext* context, TfLiteNode* node,         \
                             const TfLiteTensor* input1,                       \
                             const TfLiteTensor* input2, TfLiteTensor* output, \
                             bool requires_broadcast) {                        \
    if (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8) {         \
      auto input1_offset = -input1->params.zero_point;                         \
      auto input2_offset = -input2->params.zero_point;                         \
      const int left_shift = 8;                                                \
                                                                               \
      int32 input1_multiplier;                                                 \
      int input1_shift;                                                        \
      QuantizeMultiplierSmallerThanOneExp(input1->params.scale,                \
                                          &input1_multiplier, &input1_shift);  \
      int32 input2_multiplier;                                                 \
      int input2_shift;                                                        \
      QuantizeMultiplierSmallerThanOneExp(input2->params.scale,                \
                                          &input2_multiplier, &input2_shift);  \
                                                                               \
      ComparisonParams op_params;                                              \
      op_params.left_shift = left_shift;                                       \
      op_params.input1_offset = input1_offset;                                 \
      op_params.input1_multiplier = input1_multiplier;                         \
      op_params.input1_shift = input1_shift;                                   \
      op_params.input2_offset = input2_offset;                                 \
      op_params.input2_multiplier = input2_multiplier;                         \
      op_params.input2_shift = input2_shift;                                   \
      if (requires_broadcast) {                                                \
        reference_ops::Broadcast4DSlow##opname##WithScaling(                   \
            op_params, GetTensorShape(input1),                                 \
            GetTensorData<input_dtype>(input1), GetTensorShape(input2),        \
            GetTensorData<input_dtype>(input2), GetTensorShape(output),        \
            GetTensorData<bool>(output));                                      \
      } else {                                                                 \
        reference_ops::opname##WithScaling(                                    \
            op_params, GetTensorShape(input1),                                 \
            GetTensorData<input_dtype>(input1), GetTensorShape(input2),        \
            GetTensorData<input_dtype>(input2), GetTensorShape(output),        \
            GetTensorData<bool>(output));                                      \
      }                                                                        \
    }                                                                          \
  }
TF_LITE_QUANTIZE_COMPARISON(Equal);
TF_LITE_QUANTIZE_COMPARISON(NotEqual);
TF_LITE_QUANTIZE_COMPARISON(Greater);
TF_LITE_QUANTIZE_COMPARISON(GreaterEqual);
TF_LITE_QUANTIZE_COMPARISON(Less);
TF_LITE_QUANTIZE_COMPARISON(LessEqual);
#undef TF_LITE_QUANTIZE_COMPARISON

#define TF_LITE_COMPARISON(type, opname, requires_broadcast)                  \
  {                                                                           \
    ComparisonParams op_params;                                               \
    requires_broadcast                                                        \
        ? reference_ops::Broadcast4DSlow##opname##NoScaling(                  \
              op_params, GetTensorShape(input1), GetTensorData<type>(input1), \
              GetTensorShape(input2), GetTensorData<type>(input2),            \
              GetTensorShape(output), GetTensorData<bool>(output))            \
        : reference_ops::opname##NoScaling(                                   \
              op_params, GetTensorShape(input1), GetTensorData<type>(input1), \
              GetTensorShape(input2), GetTensorData<type>(input2),            \
              GetTensorShape(output), GetTensorData<bool>(output));           \
  }

#define TF_LITE_COMPARISON_INVOKE_EQUAL(op_name)                              \
  TfLiteStatus op_name##Eval(TfLiteContext* context, TfLiteNode* node) {      \
    const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);      \
    const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);      \
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);           \
    bool requires_broadcast = !HaveSameShapes(input1, input2);                \
    switch (input1->type) {                                                   \
      case kTfLiteBool:                                                       \
        TF_LITE_COMPARISON(bool, op_name, requires_broadcast);                \
        break;                                                                \
      case kTfLiteFloat32:                                                    \
        TF_LITE_COMPARISON(float, op_name, requires_broadcast);               \
        break;                                                                \
      case kTfLiteInt32:                                                      \
        TF_LITE_COMPARISON(int32_t, op_name, requires_broadcast);             \
        break;                                                                \
      case kTfLiteInt64:                                                      \
        TF_LITE_COMPARISON(int64_t, op_name, requires_broadcast);             \
        break;                                                                \
      case kTfLiteUInt8:                                                      \
        EvalQuantized##op_name<uint8_t>(context, node, input1, input2, output,\
                                        requires_broadcast);                  \
        break;                                                                \
      case kTfLiteInt8:                                                       \
        EvalQuantized##op_name<int8_t>(context, node, input1, input2, output, \
                                       requires_broadcast);                   \
        break;                                                                \
      default:                                                                \
        context->ReportError(context,                                         \
                             "Does not support type %d,"                      \
                             " requires bool|float|int|uint8",                \
                             input1->type);                                   \
        return kTfLiteError;                                                  \
    }                                                                         \
    return kTfLiteOk;                                                         \
  }                                                                           \

#define TF_LITE_COMPARISON_INVOKE_GL(op_name)                                 \
  TfLiteStatus op_name##Eval(TfLiteContext* context, TfLiteNode* node) {      \
    const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);      \
    const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);      \
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);           \
    bool requires_broadcast = !HaveSameShapes(input1, input2);                \
    switch (input1->type) {                                                   \
      case kTfLiteFloat32:                                                    \
        TF_LITE_COMPARISON(float, op_name, requires_broadcast);               \
        break;                                                                \
      case kTfLiteInt32:                                                      \
        TF_LITE_COMPARISON(int32_t, op_name, requires_broadcast);             \
        break;                                                                \
      case kTfLiteInt64:                                                      \
        TF_LITE_COMPARISON(int64_t, op_name, requires_broadcast);             \
        break;                                                                \
      case kTfLiteUInt8:                                                      \
        EvalQuantized##op_name<uint8_t>(context, node, input1, input2, output,\
                                        requires_broadcast);                  \
        break;                                                                \
      case kTfLiteInt8:                                                       \
        EvalQuantized##op_name<int8_t>(context, node, input1, input2, output, \
                                       requires_broadcast);                   \
        break;                                                                \
      default:                                                                \
        context->ReportError(context,                                         \
                             "Does not support type %d,"                      \
                             " requires float|int|uint8",                     \
                             input1->type);                                   \
        return kTfLiteError;                                                  \
    }                                                                         \
    return kTfLiteOk;                                                         \
  }                                                                           \

TF_LITE_COMPARISON_INVOKE_EQUAL(Equal)
TF_LITE_COMPARISON_INVOKE_EQUAL(NotEqual)
TF_LITE_COMPARISON_INVOKE_GL(Greater)
TF_LITE_COMPARISON_INVOKE_GL(GreaterEqual)
TF_LITE_COMPARISON_INVOKE_GL(Less)
TF_LITE_COMPARISON_INVOKE_GL(LessEqual)
#undef TF_LITE_COMPARISON
#undef TF_LITE_COMPARISON_INVOKE_EQUAL
#undef TF_LITE_COMPARISON_INVOKE_GL

}  // namespace
}  // namespace comparisons

TfLiteRegistration* Register_EQUAL() {
  static TfLiteRegistration r = {
      nullptr, nullptr, comparisons::ComparisonPrepare, comparisons::EqualEval};
  return &r;
}

TfLiteRegistration* Register_NOT_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::NotEqualEval};
  return &r;
}

TfLiteRegistration* Register_GREATER() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::GreaterEval};
  return &r;
}

TfLiteRegistration* Register_GREATER_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::GreaterEqualEval};
  return &r;
}

TfLiteRegistration* Register_LESS() {
  static TfLiteRegistration r = {
      nullptr, nullptr, comparisons::ComparisonPrepare, comparisons::LessEval};
  return &r;
}

TfLiteRegistration* Register_LESS_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::LessEqualEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
