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
#include <string.h>
#include <vector>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace softplus {

const float  threshold = std::log(std::numeric_limits<float>::epsilon()) + 2.0f;

struct SoftPlusContext {
  SoftPlusContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  SoftPlusContext op_context(context, node);
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(op_context.input->dims);
  op_context.output->type = op_context.input->type;
  return context->ResizeTensor(context, op_context.output, output_dims);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  SoftPlusContext op_context(context, node);

  if (op_context.input->type == kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);
    reference_ops::softplus(threshold, GetTensorShape(input),
                            GetTensorData<float>(input), GetTensorShape(output),
                            GetTensorData<float>(output));
  } else {
    context->ReportError(context,
                         "Type %d is currently not supported by SoftPlus.",
                         op_context.input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace softplus

TfLiteRegistration* Register_SOFTPLUS_REF() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 softplus::Prepare, softplus::Eval};
  return &r;
}

TfLiteRegistration* Register_SOFTPLUS() { return Register_SOFTPLUS_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
