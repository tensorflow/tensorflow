/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

namespace {
TfLiteStatus ExpandTensorDim(TfLiteContext* context, const TfLiteTensor* input,
                             int axis, TfLiteTensor* output) {
  const TfLiteIntArray *input_dims = input->dims;
  if (axis < 0) {
    axis = input_dims.size + 1 + axis;
  }
  TF_LITE_ENSURE(context, axis <= input_dims.size);

  TfLiteIntArray* output_dims = output->dims;
  for (int i = 0; i < output_dims->size; ++i) {
    if (i < axis) {
      output_dims->data[i] = input_dims.data[i];
    } else if (i == axis) {
      output_dims->data[i] = 1;
    } else {
      output_dims->data[i] = input_dims.data[i - 1];
    }
  }
  return kTfLiteOk;
}

TfLiteStatus GetAxisValueFromTensor(TfLiteContext* context,
                                    const TfLiteTensor* axis, int* axis_value) {
  TF_LITE_ENSURE_EQ(context, NumElements(&axis), 1);
  switch (axis->type) {
    case kTfLiteInt32:
      *axis_value = tflite::micro::GetTensorData<int32_t>(axis);
      return kTfLiteOk;
    case kTfLiteInt64:
      *axis_value = tflite::micro::GetTensorData<int64_t>(axis);
      return kTfLiteOk;
    default:
      TF_LITE_KERNEL_LOG(context, "Axis type %s (%d) not supported by Expand_Dims.",
                         TfLiteTypeGetName(axis->type), axis->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = input->type;
  if (IsConstantTensor(axis)) {
    int axis_value;
    TF_LITE_ENSURE_OK(context,
                      GetAxisValueFromTensor(context, axis, &axis_value));
    return ExpandTensorDim(context, input, axis_value, output);
  }
  SetTensorToDynamic(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Just copy input to output.
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      tflite::micro::GetEvalInput(context, node, kAxisTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (IsDynamicTensor(output)) {
    int axis_value;
    TF_LITE_ENSURE_OK(context,
                      GetAxisValueFromTensor(context, axis, &axis_value));
    TF_LITE_ENSURE_OK(context,
                      ExpandTensorDim(context, *input, axis_value, output));
  }
  // For loop for memcpy(output->data.raw, input->data.raw, input->bytes);
  for (size_t i = 0; i < input->bytes; ++i) {
    output->data.raw[i] = input->data.raw[i]
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_EXPAND_DIMS() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
