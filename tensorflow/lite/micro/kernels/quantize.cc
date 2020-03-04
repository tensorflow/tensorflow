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
#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace quantize {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  // TODO(b/128934713): Add support for fixed-point per-channel quantization.
  // Currently this only support affine per-layer quantization.
  TF_LITE_ENSURE_EQ(context, output->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(output->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, affine_quantization->scale->size == 1);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt16);
  TF_LITE_ENSURE(context,
                 output->type == kTfLiteUInt8 || output->type == kTfLiteInt8);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  tflite::QuantizationParams op_params;
  op_params.zero_point = output->params.zero_point;
  op_params.scale = static_cast<double>(output->params.scale);

  if (input->type == kTfLiteFloat32) {
    switch (output->type) {
      case kTfLiteInt8:
        reference_ops::AffineQuantize(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
        break;
      case kTfLiteUInt8:
        reference_ops::AffineQuantize(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                           TfLiteTypeGetName(input->type),
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt16) {
    switch (output->type) {
      case kTfLiteInt8:
        reference_ops::AffineQuantize(
            op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
        break;

      default:
        TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                           TfLiteTypeGetName(input->type),
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                       TfLiteTypeGetName(input->type),
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace quantize

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// AffineQuantize takes scale and zero point and quantizes the float value to
// quantized output, in int8 or uint8 format.
TfLiteRegistration* Register_QUANTIZE() {
  static TfLiteRegistration r = {};
  r.init = quantize::Init;
  r.free = quantize::Free;
  r.prepare = quantize::Prepare;
  r.invoke = quantize::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
