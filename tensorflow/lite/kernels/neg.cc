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

#include "tensorflow/lite/kernels/internal/reference/neg.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/neg.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace neg {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  // sum of input and output zero points, accumulated on int16
  // highest possible value: 127 + 127 = 254
  // lowest possible value: -128 + - 128 = -256
  // thus, accumulate on int16
  int16_t zero_points_sum;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // allocate a new object to carry information from Prepare() to
  // Eval().
  auto* op_data = new OpData;
  return static_cast<void*>(op_data);
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  constexpr auto kI8Min =
      static_cast<int16_t>(std::numeric_limits<int8_t>::min());
  constexpr auto kI8Max =
      static_cast<int16_t>(std::numeric_limits<int8_t>::max());

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  output->type = input->type;

  if (input->type == kTfLiteInt8) {
    auto op_data = static_cast<OpData*>(node->user_data);

    // using EQ attempts to cast to int via the %d format specifier and gives
    // incorrect value and since some hardware platforms do not support float
    // formatting by default, I did a direct equality check instead
    TF_LITE_ENSURE(context, input->params.scale == output->params.scale);

    // within: [-128, 127]
    TF_LITE_ENSURE(context, input->params.zero_point <= kI8Max);
    TF_LITE_ENSURE(context, input->params.zero_point >= kI8Min);

    // within: [-128, 127]
    TF_LITE_ENSURE(context, output->params.zero_point <= kI8Max);
    TF_LITE_ENSURE(context, output->params.zero_point >= kI8Min);

    op_data->zero_points_sum = static_cast<int16_t>(input->params.zero_point) +
                               static_cast<int16_t>(output->params.zero_point);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteInt64:
      reference_ops::Negate(
          GetTensorShape(input), GetTensorData<int64_t>(input),
          GetTensorShape(output), GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::Negate(
          GetTensorShape(input), GetTensorData<int32_t>(input),
          GetTensorShape(output), GetTensorData<int32_t>(output));
      break;
    case kTfLiteFloat32:
      reference_ops::Negate(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      break;
    case kTfLiteInt8: {
      auto op_data = static_cast<OpData*>(node->user_data);
      reference_integer_ops::Negate(
          GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output),
          op_data->zero_points_sum);
    } break;

    default:
      context->ReportError(
          context,
          "Neg only currently supports int64, int32, float32, and int8 got %d.",
          input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace neg

TfLiteRegistration* Register_NEG() {
  static TfLiteRegistration r = {/*init=*/neg::Init, /*free=*/neg::Free,
                                 /*prepare=*/neg::Prepare,
                                 /*invoke=*/neg::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
