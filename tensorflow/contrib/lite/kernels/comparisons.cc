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
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/string_util.h"

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
  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);
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
  void EvalQuantized##opname(TfLiteContext* context, TfLiteNode* node,         \
                             const TfLiteTensor* input1,                       \
                             const TfLiteTensor* input2, TfLiteTensor* output, \
                             bool requires_broadcast) {                        \
    if (input1->type == kTfLiteUInt8) {                                        \
      auto input1_offset = -input1->params.zero_point;                         \
      auto input2_offset = -input2->params.zero_point;                         \
      const int left_shift = 20;                                               \
      const double twice_max_input_scale =                                     \
          2 * std::max(input1->params.scale, input2->params.scale);            \
      const double real_input1_multiplier =                                    \
          input1->params.scale / twice_max_input_scale;                        \
      const double real_input2_multiplier =                                    \
          input2->params.scale / twice_max_input_scale;                        \
                                                                               \
      int32 input1_multiplier;                                                 \
      int input1_shift;                                                        \
      QuantizeMultiplierSmallerThanOneExp(real_input1_multiplier,              \
                                          &input1_multiplier, &input1_shift);  \
      int32 input2_multiplier;                                                 \
      int input2_shift;                                                        \
      QuantizeMultiplierSmallerThanOneExp(real_input2_multiplier,              \
                                          &input2_multiplier, &input2_shift);  \
                                                                               \
      if (requires_broadcast) {                                                \
        reference_ops::Broadcast##opname(                                      \
            left_shift, GetTensorData<uint8_t>(input1), GetTensorDims(input1), \
            input1_offset, input1_multiplier, input1_shift,                    \
            GetTensorData<uint8_t>(input2), GetTensorDims(input2),             \
            input2_offset, input2_multiplier, input2_shift,                    \
            GetTensorData<bool>(output), GetTensorDims(output));               \
      } else {                                                                 \
        reference_ops::opname(                                                 \
            left_shift, GetTensorData<uint8_t>(input1), GetTensorDims(input1), \
            input1_offset, input1_multiplier, input1_shift,                    \
            GetTensorData<uint8_t>(input2), GetTensorDims(input2),             \
            input2_offset, input2_multiplier, input2_shift,                    \
            GetTensorData<bool>(output), GetTensorDims(output));               \
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

#define TF_LITE_COMPARISON(type, opname, requires_broadcast)    \
  requires_broadcast                                            \
      ? reference_ops::Broadcast##opname(                       \
            GetTensorData<type>(input1), GetTensorDims(input1), \
            GetTensorData<type>(input2), GetTensorDims(input2), \
            GetTensorData<bool>(output), GetTensorDims(output)) \
      : reference_ops::opname(                                  \
            GetTensorData<type>(input1), GetTensorDims(input1), \
            GetTensorData<type>(input2), GetTensorDims(input2), \
            GetTensorData<bool>(output), GetTensorDims(output));

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, Equal, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, Equal, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, Equal, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedEqual(context, node, input1, input2, output,
                         requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// TODO(renjieliu): Refactor the logic to avoid duplications.
TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, NotEqual, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, NotEqual, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, NotEqual, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedNotEqual(context, node, input1, input2, output,
                            requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, Greater, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, Greater, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, Greater, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedGreater(context, node, input1, input2, output,
                           requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, GreaterEqual, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, GreaterEqual, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, GreaterEqual, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedGreaterEqual(context, node, input1, input2, output,
                                requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, Less, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, Less, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, Less, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedLess(context, node, input1, input2, output,
                        requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      TF_LITE_COMPARISON(float, LessEqual, requires_broadcast);
      break;
    case kTfLiteInt32:
      TF_LITE_COMPARISON(int32_t, LessEqual, requires_broadcast);
      break;
    case kTfLiteInt64:
      TF_LITE_COMPARISON(int64_t, LessEqual, requires_broadcast);
      break;
    case kTfLiteUInt8:
      EvalQuantizedLessEqual(context, node, input1, input2, output,
                             requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

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
