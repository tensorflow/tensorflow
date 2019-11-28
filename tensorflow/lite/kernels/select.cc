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
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace select {

constexpr int kInputTensorCondition = 0;
constexpr int kInputTensorX = 1;
constexpr int kInputTensorY = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus SelectPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_condition =
      GetInput(context, node, kInputTensorCondition);
  const TfLiteTensor* input_x = GetInput(context, node, kInputTensorX);
  const TfLiteTensor* input_y = GetInput(context, node, kInputTensorY);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Input must be bool.
  TF_LITE_ENSURE(context, input_condition->type == kTfLiteBool);

  // Input tensors must have the same type and size
  TF_LITE_ENSURE_EQ(context, input_x->type, input_y->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input_x, input_y));
  output->type = input_x->type;

  // Either the same shape, or input_condition must be Rank 1 and match over the
  // first dimension.
  bool same_shape = HaveSameShapes(input_condition, input_x);
  if (!same_shape && NumDimensions(input_condition) == 1) {
    same_shape =
        SizeOfDimension(input_condition, 0) == SizeOfDimension(input_x, 0);
  }

  TF_LITE_ENSURE(context, same_shape);

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_x->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus SelectEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_condition =
      GetInput(context, node, kInputTensorCondition);
  const TfLiteTensor* input_x = GetInput(context, node, kInputTensorX);
  const TfLiteTensor* input_y = GetInput(context, node, kInputTensorY);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  bool is_rank_one = !HaveSameShapes(input_condition, input_x);

#define TF_LITE_SELECT(type, op)                                           \
  reference_ops::op(GetTensorShape(input_condition),                       \
                    GetTensorData<bool>(input_condition),                  \
                    GetTensorShape(input_x), GetTensorData<type>(input_x), \
                    GetTensorShape(input_y), GetTensorData<type>(input_y), \
                    GetTensorShape(output), GetTensorData<type>(output));

#define TF_LITE_SWITCH(type, op)                                               \
  switch (type) {                                                              \
    break;                                                                     \
    case kTfLiteBool:                                                          \
      TF_LITE_SELECT(bool, op);                                                \
      break;                                                                   \
    case kTfLiteFloat32:                                                       \
      TF_LITE_SELECT(float, op);                                               \
      break;                                                                   \
    case kTfLiteUInt8:                                                         \
      TF_LITE_SELECT(uint8_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt8:                                                          \
      TF_LITE_SELECT(int8_t, op);                                              \
      break;                                                                   \
    case kTfLiteInt16:                                                         \
      TF_LITE_SELECT(int16_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt32:                                                         \
      TF_LITE_SELECT(int32_t, op);                                             \
      break;                                                                   \
    case kTfLiteInt64:                                                         \
      TF_LITE_SELECT(int64_t, op);                                             \
      break;                                                                   \
    default:                                                                   \
      context->ReportError(context,                                            \
                           "Does not support type other than bool|float|int, " \
                           "got %d",                                           \
                           type);                                              \
      return kTfLiteError;                                                     \
  }

  if (is_rank_one) {
    TF_LITE_SWITCH(input_x->type, RankOneSelect);
  } else {
    TF_LITE_SWITCH(input_x->type, Select);
  }

#undef TF_LITE_SELECT
#undef TF_LITE_SWITCH
  return kTfLiteOk;
}

}  // namespace select

TfLiteRegistration* Register_SELECT() {
  static TfLiteRegistration r = {nullptr, nullptr, select::SelectPrepare,
                                 select::SelectEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
