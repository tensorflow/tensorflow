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

#include "tensorflow/lite/kernels/internal/reference/dequantize.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace dequantize {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // TODO(b/140515557): Add cached dequant to improve hybrid model performance.
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  tflite::DequantizationParams op_params;
  op_params.zero_point = input->params.zero_point;
  op_params.scale = static_cast<double>(input->params.scale);
  switch (input->type) {
    case kTfLiteUInt8:
      reference_ops::Dequantize(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::Dequantize(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    default:
      context->ReportError(context, "Type %s (%d) not supported.",
                           TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace dequantize

TfLiteRegistration* Register_DEQUANTIZE() {
  static TfLiteRegistration r = {nullptr, nullptr, dequantize::Prepare,
                                 dequantize::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
