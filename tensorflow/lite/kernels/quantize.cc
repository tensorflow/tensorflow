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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace quantize {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);

  TF_LITE_ENSURE(context, op_context.input->type == kTfLiteFloat32);
  TF_LITE_ENSURE(context, op_context.output->type == kTfLiteUInt8 ||
                              op_context.output->type == kTfLiteInt8);

  // TODO(b/128934713): Add support for fixed-point per-channel quantization.
  // Currently this only support affine per-layer quantization.
  TF_LITE_ENSURE_EQ(context, op_context.output->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      op_context.output->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, affine_quantization->scale->size == 1);

  return context->ResizeTensor(context, op_context.output,
                               TfLiteIntArrayCopy(op_context.input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  tflite::QuantizationParams op_params;
  op_params.zero_point = op_context.output->params.zero_point;
  op_params.scale = op_context.output->params.scale;
  switch (op_context.output->type) {
    case kTfLiteUInt8:
      optimized_ops::AffineQuantize(op_params, GetTensorShape(op_context.input),
                                    GetTensorData<float>(op_context.input),
                                    GetTensorShape(op_context.output),
                                    GetTensorData<uint8_t>(op_context.output));
      break;
    case kTfLiteInt8:
      optimized_ops::AffineQuantize(op_params, GetTensorShape(op_context.input),
                                    GetTensorData<float>(op_context.input),
                                    GetTensorShape(op_context.output),
                                    GetTensorData<int8_t>(op_context.output));
      break;
    default:
      context->ReportError(context, "Type %d not supported.",
                           op_context.input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace quantize

TfLiteRegistration* Register_QUANTIZE_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, quantize::Prepare,
                                 quantize::Eval};
  return &r;
}

TfLiteRegistration* Register_QUANTIZE() { return Register_QUANTIZE_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
