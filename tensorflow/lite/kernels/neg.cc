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

#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace neg {

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
    case kTfLiteInt8:
      reference_ops::Negate(GetTensorShape(input), GetTensorData<int8_t>(input),
                            GetTensorShape(output),
                            GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Negate(
          GetTensorShape(input), GetTensorData<int16_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
      break;
    case kTfLiteFloat16:
      reference_ops::Negate(
          GetTensorShape(input), GetTensorData<Eigen::half>(input),
          GetTensorShape(output), GetTensorData<Eigen::half>(output));
      break;
    case kTfLiteBFloat16:
      reference_ops::Negate(
          GetTensorShape(input), GetTensorData<Eigen::half>(input),
          GetTensorShape(output), GetTensorData<Eigen::half>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Neg only currently supports int64, int32, and float32, got %d.",
          input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace neg

TfLiteRegistration* Register_NEG() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 neg::Prepare, neg::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
