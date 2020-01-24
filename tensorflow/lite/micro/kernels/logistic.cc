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

#include "tensorflow/lite/kernels/internal/reference/logistic.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Logistic(
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    }
    default: {
      // TODO(b/141211002): Also support other data types once we have supported
      // temporary tensors in TFLM.
      context->ReportError(context,
                           "Only float32 is supported currently, got %s",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace activations

TfLiteRegistration* Register_LOGISTIC() {
  static TfLiteRegistration r = {};
  r.prepare = activations::Prepare;
  r.invoke = activations::Eval;
  return &r;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite
