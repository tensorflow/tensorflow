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

#include "tensorflow/lite/kernels/internal/reference/cumsum.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

static const int kInputTensor = 0;
static const int kAxisTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis_tensor =
      tflite::micro::GetEvalInput(context, node, kAxisTensor);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  auto* params = static_cast<TfLiteCumsumParams*>(node->builtin_data);
  auto input_shape = tflite::micro::GetTensorShape(input);

  int32_t axis = *tflite::micro::GetTensorData<int32_t>(axis_tensor);
  if (axis < 0) axis += input_shape.DimensionsCount();

  if (axis < 0 || axis >= input_shape.DimensionsCount()) {
    TF_LITE_KERNEL_LOG(context, "CUMSUM Invalid axis: %d", axis);
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::CumSum(tflite::micro::GetTensorData<float>(input),
                            input_shape, axis, params->exclusive,
                            params->reverse,
                            tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Unsupported input type, CUMSUM only supports FLOAT32.");
      return kTfLiteError;
    }
  }

  return kTfLiteError;
}

}  // namespace

TfLiteRegistration Register_CUMSUM() {
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
