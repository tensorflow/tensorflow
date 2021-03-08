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
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = input->type;

  return kTfLiteOk;
}

template <typename T>
void resetZeros(T* out, const int num_elements) {
  for (int i = 0; i < num_elements; ++i) {
    out[i] = static_cast<T>(0);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  int flat_size = MatchingFlatSize(tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorShape(output));
  switch (input->type) {
    case kTfLiteInt64:
      resetZeros(tflite::micro::GetTensorData<int64_t>(output), flat_size);
      break;
    case kTfLiteInt32:
      resetZeros(tflite::micro::GetTensorData<int32_t>(output), flat_size);
      break;
    case kTfLiteInt8:
      resetZeros(tflite::micro::GetTensorData<int8_t>(output), flat_size);
      break;
    case kTfLiteFloat32:
      resetZeros(tflite::micro::GetTensorData<float>(output), flat_size);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "ZerosLike only currently supports int64, int32, "
                         "and float32, got %d.",
                         input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_ZEROS_LIKE() {
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
