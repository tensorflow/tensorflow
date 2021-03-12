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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

static const int kInputTensor = 0;
static const int kAxisTensor = 1;
static const int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteInt32 ||
                              input->type == kTfLiteFloat32 ||
                              input->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // TODO: ensure output shape matches input shape
  return kTfLiteError;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis_tensor = GetInput(context, node, kAxisTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  auto* params = reinterpret_cast<TfLiteCumsumParams*>(node->builtin_data);

  int axis = *GetTensorData<int>(axis_tensor);
  if (axis < 0) axis += NumDimensions(input);

  if (axis < 0 || axis >= NumDimensions(input)) {
    TF_LITE_KERNEL_LOG(context, "Invalid axis: ", axis);
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::CumSum(GetTensorData<float>(input), GetTensorShape(input),
                            axis, params->exclusive, params->reverse,
                            GetTensorData<float>(output));
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

TfLiteRegistration* Register_CUMSUM() { return nullptr; }

}  // namespace tflite
