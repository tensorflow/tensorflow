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
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace comparisons {
namespace {

struct OpData {
  ComparisonParams params;
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data);
      break;
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::EqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// TODO(renjieliu): Refactor the logic to avoid duplications.
TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<bool>(input1), input2_shape,
                tflite::micro::GetTensorData<bool>(input2), output_shape,
                output_data);
      break;
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowNotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::NotEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowGreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::GreaterEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::LessNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  RuntimeShape input1_shape = tflite::micro::GetTensorShape(input1);
  RuntimeShape input2_shape = tflite::micro::GetTensorShape(input2);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  bool* output_data = tflite::micro::GetTensorData<bool>(output);

  bool requires_broadcast = !tflite::micro::HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<float>(input1), input2_shape,
                tflite::micro::GetTensorData<float>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt32:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int32_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int32_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt64:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualNoScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int64_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int64_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteUInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<uint8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<uint8_t>(input2), output_shape,
                output_data);
      break;
    case kTfLiteInt8:
      requires_broadcast
          ? reference_ops::Broadcast4DSlowLessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data)
          : reference_ops::LessEqualWithScaling(
                data->params, input1_shape,
                tflite::micro::GetTensorData<int8_t>(input1), input2_shape,
                tflite::micro::GetTensorData<int8_t>(input2), output_shape,
                output_data);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input1->type), input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);

  if (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8) {
    auto input1_offset = -input1->params.zero_point;
    auto input2_offset = -input2->params.zero_point;
    const int kLeftShift = 8;

    int32_t input1_multiplier;
    int input1_shift;
    QuantizeMultiplierSmallerThanOneExp(
        static_cast<double>(input1->params.scale), &input1_multiplier,
        &input1_shift);
    int32_t input2_multiplier;
    int input2_shift;
    QuantizeMultiplierSmallerThanOneExp(
        static_cast<double>(input2->params.scale), &input2_multiplier,
        &input2_shift);

    data->params.left_shift = kLeftShift;
    data->params.input1_offset = input1_offset;
    data->params.input1_multiplier = input1_multiplier;
    data->params.input1_shift = input1_shift;
    data->params.input2_offset = input2_offset;
    data->params.input2_multiplier = input2_multiplier;
    data->params.input2_shift = input2_shift;
  }

  return kTfLiteOk;
}

}  // namespace comparisons

TfLiteRegistration Register_EQUAL() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::EqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_NOT_EQUAL() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::NotEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_GREATER() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::GreaterEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_GREATER_EQUAL() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::GreaterEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LESS() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::LessEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_LESS_EQUAL() {
  return {/*init=*/comparisons::Init,
          /*free=*/nullptr,
          /*prepare=*/comparisons::Prepare,
          /*invoke=*/comparisons::LessEqualEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
