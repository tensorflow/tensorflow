/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace zeros_like {

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

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const int num_elements = NumElements(input);
  switch (input->type) {
    case kTfLiteInt64:
      memset(GetTensorData<int64_t>(output), 0, num_elements * sizeof(int64_t));
      break;
    case kTfLiteInt32:
      memset(GetTensorData<int32_t>(output), 0, num_elements * sizeof(int32_t));
      break;
    case kTfLiteFloat32:
      memset(GetTensorData<float>(output), 0, num_elements * sizeof(float));
      break;
    default:
      context->ReportError(context,
                           "ZerosLike only currently supports int64, int32, "
                           "and float32, got %d.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace zeros_like

TfLiteRegistration* Register_ZEROS_LIKE() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 zeros_like::Prepare, zeros_like::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
