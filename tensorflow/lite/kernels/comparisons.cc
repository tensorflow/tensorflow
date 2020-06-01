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
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace comparisons {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ComparisonPrepareCommon(TfLiteContext* context, TfLiteNode* node,
                                     bool is_string_allowed) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Don't support string.
  if (!is_string_allowed) {
    TF_LITE_ENSURE(context, input1->type != kTfLiteString);
  }
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

TfLiteStatus ComparisonPrepare(TfLiteContext* context, TfLiteNode* node) {
  return ComparisonPrepareCommon(context, node, false);
}

TfLiteStatus ComparisonPrepareStringAllowed(TfLiteContext* context,
                                            TfLiteNode* node) {
  return ComparisonPrepareCommon(context, node, true);
}

template <typename input_dtype, reference_ops::ComparisonFn<int32> opname>
void ComparisonQuantized(const TfLiteTensor* input1, const TfLiteTensor* input2,
                         TfLiteTensor* output, bool requires_broadcast) {
  if (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8) {
    auto input1_offset = -input1->params.zero_point;
    auto input2_offset = -input2->params.zero_point;
    const int left_shift = 8;

    int32 input1_multiplier;
    int input1_shift;
    QuantizeMultiplierSmallerThanOneExp(input1->params.scale,
                                        &input1_multiplier, &input1_shift);
    int32 input2_multiplier;
    int input2_shift;
    QuantizeMultiplierSmallerThanOneExp(input2->params.scale,
                                        &input2_multiplier, &input2_shift);

    ComparisonParams op_params;
    op_params.left_shift = left_shift;
    op_params.input1_offset = input1_offset;
    op_params.input1_multiplier = input1_multiplier;
    op_params.input1_shift = input1_shift;
    op_params.input2_offset = input2_offset;
    op_params.input2_multiplier = input2_multiplier;
    op_params.input2_shift = input2_shift;
    if (requires_broadcast) {
      reference_ops::BroadcastComparison4DSlowWithScaling<input_dtype, opname>(
          op_params, GetTensorShape(input1), GetTensorData<input_dtype>(input1),
          GetTensorShape(input2), GetTensorData<input_dtype>(input2),
          GetTensorShape(output), GetTensorData<bool>(output));
    } else {
      reference_ops::ComparisonWithScaling<input_dtype, opname>(
          op_params, GetTensorShape(input1), GetTensorData<input_dtype>(input1),
          GetTensorShape(input2), GetTensorData<input_dtype>(input2),
          GetTensorShape(output), GetTensorData<bool>(output));
    }
  }
}

template <typename T, reference_ops::ComparisonFn<T> opname>
void Comparison(const TfLiteTensor* input1, const TfLiteTensor* input2,
                TfLiteTensor* output, bool requires_broadcast) {
  ComparisonParams op_params;
  requires_broadcast
      ? reference_ops::BroadcastComparison4DSlowImpl<T, opname>(
            op_params, GetTensorShape(input1), GetTensorData<T>(input1),
            GetTensorShape(input2), GetTensorData<T>(input2),
            GetTensorShape(output), GetTensorData<bool>(output))
      : reference_ops::ComparisonImpl<T, opname>(
            op_params, GetTensorShape(input1), GetTensorData<T>(input1),
            GetTensorShape(input2), GetTensorData<T>(input2),
            GetTensorShape(output), GetTensorData<bool>(output));
}

void ComparisonString(bool (*opname)(const StringRef&, const StringRef&),
                      const TfLiteTensor* input1, const TfLiteTensor* input2,
                      TfLiteTensor* output, bool requires_broadcast) {
  bool* output_data = GetTensorData<bool>(output);
  if (requires_broadcast) {
    reference_ops::BroadcastComparison4DSlowStringImpl(
        opname, GetTensorShape(input1), input1, GetTensorShape(input2), input2,
        GetTensorShape(output), output_data);
  } else {
    reference_ops::ComparisonStringImpl(opname, GetTensorShape(input1), input1,
                                        GetTensorShape(input2), input2,
                                        GetTensorShape(output), output_data);
  }
}

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      Comparison<bool, reference_ops::EqualFn>(input1, input2, output,
                                               requires_broadcast);
      break;
    case kTfLiteFloat32:
      Comparison<float, reference_ops::EqualFn>(input1, input2, output,
                                                requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::EqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::EqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::EqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::EqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteString:
      ComparisonString(reference_ops::StringRefEqualFn, input1, input2, output,
                       requires_broadcast);
      break;
    default:
      context->ReportError(
          context,
          "Does not support type %d, requires bool|float|int|uint8|string",
          input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      Comparison<bool, reference_ops::NotEqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteFloat32:
      Comparison<float, reference_ops::NotEqualFn>(input1, input2, output,
                                                   requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::NotEqualFn>(input1, input2, output,
                                                     requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::NotEqualFn>(input1, input2, output,
                                                     requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::NotEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::NotEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteString:
      ComparisonString(reference_ops::StringRefNotEqualFn, input1, input2,
                       output, requires_broadcast);
      break;
    default:
      context->ReportError(
          context,
          "Does not support type %d, requires bool|float|int|uint8|string",
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
      Comparison<float, reference_ops::GreaterFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::GreaterFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::GreaterFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::GreaterFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::GreaterFn>(
          input1, input2, output, requires_broadcast);
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
      Comparison<float, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                       requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                         requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                         requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::GreaterEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::GreaterEqualFn>(
          input1, input2, output, requires_broadcast);
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
      Comparison<float, reference_ops::LessFn>(input1, input2, output,
                                               requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::LessFn>(input1, input2, output,
                                                 requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::LessFn>(input1, input2, output,
                                                 requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::LessFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::LessFn>(input1, input2, output,
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
      Comparison<float, reference_ops::LessEqualFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::LessEqualFn>(input1, input2, output,
                                                      requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::LessEqualFn>(input1, input2, output,
                                                      requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::LessEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::LessEqualFn>(
          input1, input2, output, requires_broadcast);
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
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepareStringAllowed,
                                 comparisons::EqualEval};
  return &r;
}

TfLiteRegistration* Register_NOT_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepareStringAllowed,
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
