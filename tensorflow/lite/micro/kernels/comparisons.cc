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
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace comparisons {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

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
      QuantizeMultiplierSmallerThanOneExp(                                     \
          static_cast<double>(input1->params.scale), &input1_multiplier,       \
          &input1_shift);                                                      \
      int32 input2_multiplier;                                                 \
      int input2_shift;                                                        \
      QuantizeMultiplierSmallerThanOneExp(                                     \
          static_cast<double>(input2->params.scale), &input2_multiplier,       \
          &input2_shift);                                                      \
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

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      TF_LITE_COMPARISON(bool, Equal, requires_broadcast);
      break;
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
      EvalQuantizedEqual<uint8_t>(context, node, input1, input2, output,
                                  requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedEqual<int8_t>(context, node, input1, input2, output,
                                 requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Does not support type %d, requires bool|float|int|uint8",
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
    case kTfLiteBool:
      TF_LITE_COMPARISON(bool, NotEqual, requires_broadcast);
      break;
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
      EvalQuantizedNotEqual<uint8_t>(context, node, input1, input2, output,
                                     requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedNotEqual<int8_t>(context, node, input1, input2, output,
                                    requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Does not support type %d, requires bool|float|int|uint8",
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
      EvalQuantizedGreater<uint8_t>(context, node, input1, input2, output,
                                    requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedGreater<int8_t>(context, node, input1, input2, output,
                                   requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
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
      EvalQuantizedGreaterEqual<uint8_t>(context, node, input1, input2, output,
                                         requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedGreaterEqual<int8_t>(context, node, input1, input2, output,
                                        requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
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
      EvalQuantizedLess<uint8_t>(context, node, input1, input2, output,
                                 requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedLess<int8_t>(context, node, input1, input2, output,
                                requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
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
      EvalQuantizedLessEqual<uint8_t>(context, node, input1, input2, output,
                                      requires_broadcast);
      break;
    case kTfLiteInt8:
      EvalQuantizedLessEqual<int8_t>(context, node, input1, input2, output,
                                     requires_broadcast);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Does not support type %d, requires float|int|uint8",
                         input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace comparisons

TfLiteRegistration* Register_EQUAL() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::EqualEval;
  return &r;
}

TfLiteRegistration* Register_NOT_EQUAL() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::NotEqualEval;
  return &r;
}

TfLiteRegistration* Register_GREATER() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::GreaterEval;
  return &r;
}

TfLiteRegistration* Register_GREATER_EQUAL() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::GreaterEqualEval;
  return &r;
}

TfLiteRegistration* Register_LESS() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::LessEval;
  return &r;
}

TfLiteRegistration* Register_LESS_EQUAL() {
  static TfLiteRegistration r = {};
  r.invoke = comparisons::LessEqualEval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
