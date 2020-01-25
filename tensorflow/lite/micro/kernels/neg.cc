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

#include "tensorflow/lite/kernels/internal/reference/neg.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace neg {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    // TODO(wangtz): handle for kTfLiteInt8
    case kTfLiteFloat32:
      reference_ops::Negate(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      break;
    default:
      context->ReportError(
          context, "Neg only currently supports float32, got %d.", input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace neg

TfLiteRegistration* Register_NEG() {
  static TfLiteRegistration r = {};
  r.invoke = neg::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
