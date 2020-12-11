/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/fill.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

namespace {

template <typename T>
TfLiteStatus EnsureEqImpl(TfLiteContext* context, const TfLiteIntArray* array,
                          const TfLiteTensor* tensor) {
  for (int i = 0; i < array->size; ++i) {
    TF_LITE_ENSURE_EQ(context, array->data[i], GetTensorData<T>(tensor)[i]);
  }
  return kTfLiteOk;
}

// Ensure the equality of an int array and tensor, which should be
// one-dimensional and of integer type.
TfLiteStatus EnsureEq(TfLiteContext* context, const TfLiteIntArray* array,
                      const TfLiteTensor* tensor) {
  TF_LITE_ENSURE_EQ(context, NumDimensions(tensor), 1);
  const auto tensor_len = tensor->dims->data[0];
  TF_LITE_ENSURE_EQ(context, array->size, tensor_len);

  switch (tensor->type) {
    case kTfLiteInt32:
      return EnsureEqImpl<int32_t>(context, array, tensor);
    case kTfLiteUInt8:
      return EnsureEqImpl<uint8_t>(context, array, tensor);
    case kTfLiteInt64:
      return EnsureEqImpl<int64_t>(context, array, tensor);
    case kTfLiteInt16:
      return EnsureEqImpl<int16_t>(context, array, tensor);
    case kTfLiteInt8:
      return EnsureEqImpl<int8_t>(context, array, tensor);
    default:
      context->ReportError(
          context,
          "cannot compare int array to tensor of type %d.",
          tensor->type);
      return kTfLiteError;
  }
}

constexpr int kDimsTensor = 0;
constexpr int kValueTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* dims;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDimsTensor, &dims));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // The value tensor must be a scalar.
  TF_LITE_ENSURE_EQ(context, NumDimensions(value), 0);

  // The value type and output type must match.
  TF_LITE_ENSURE_EQ(context, output->type, value->type);

  // The dims tensor must match the output tensor shape.
  TF_LITE_ENSURE_OK(context, EnsureEq(context, output->dims, dims));

  return kTfLiteOk;
}

template <typename T>
void FillImpl(const TfLiteEvalTensor* value, TfLiteEvalTensor* output) {
  using namespace tflite::micro;
  reference_ops::Fill(GetTensorShape(value), GetTensorData<T>(value),
                      GetTensorShape(output), GetTensorData<T>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite::micro;
  const TfLiteEvalTensor* value = GetEvalInput(context, node, kValueTensor);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, kOutputTensor);

  switch (output->type) {
    case kTfLiteInt32:
      FillImpl<int32_t>(value, output);
      break;
    case kTfLiteInt64:
      FillImpl<int64_t>(value, output);
      break;
    case kTfLiteFloat32:
      FillImpl<float>(value, output);
      break;
    default:
      context->ReportError(
          context,
          "Fill only currently supports int8, int16, int32, int64, float32 "
          "for input 1, got %d.",
          value->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace anonymous

TfLiteRegistration Register_FILL() {
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
