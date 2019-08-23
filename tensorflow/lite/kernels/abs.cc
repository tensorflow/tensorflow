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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/abs.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/abs.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace abs {

enum KernelType {
  kReference,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  // Parameters used in quantized paths
  int32_t output_multiplier;
  int output_shift;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    double real_multiplier = static_cast<double>(input->params.scale) /
                             static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalAbs(TfLiteContext* context, TfLiteNode* node,
                     const OpData* data, const TfLiteTensor* input,
                     TfLiteTensor* output) {
#define TF_LITE_ABS(type, opname, data_type)                           \
  type::opname(GetTensorShape(input), GetTensorData<data_type>(input), \
               GetTensorShape(output), GetTensorData<data_type>(output))

  if (output->type == kTfLiteInt32) {
    TF_LITE_ABS(reference_ops, Abs, int32_t);
  } else if (output->type == kTfLiteFloat32) {
    TF_LITE_ABS(reference_ops, Abs, float);
#undef TF_LITE_ABS
  } else {
    context->ReportError(
        context, "Unsupported combination of input and output types in Abs.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           const OpData* data, const TfLiteTensor* input,
                           TfLiteTensor* output) {
  if (input->type == output->type &&
      (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8)) {
    tflite::AbsParams op_params;
    op_params.input_offset = -input->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
#define TF_LITE_ABS(type, opname, dtype)                                      \
  type::opname(op_params, GetTensorShape(input), GetTensorData<dtype>(input), \
               GetTensorShape(output), GetTensorData<dtype>(output))
    if (input->type == kTfLiteInt8) {
      TF_LITE_ABS(reference_integer_ops, Abs, int8_t);
    } else if (input->type == kTfLiteUInt8) {
      TF_LITE_ABS(reference_integer_ops, Abs, uint8_t);
    }
#undef TF_LITE_ABS
  } else {
    context->ReportError(
        context, "Unsupported combination of input and output types in Abs.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt32) {
    TF_LITE_ENSURE_OK(context,
                      EvalAbs<kernel_type>(context, node, data, input, output));
  } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, EvalQuantized<kernel_type>(context, node, data,
                                                          input, output));
  } else {
    context->ReportError(context,
                         "Abs only supports FLOAT32, INT32 and quantized UINT8,"
                         " INT8 now, got %d.",
                         output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace abs

TfLiteRegistration* Register_ABS_REF() {
  static TfLiteRegistration r = {abs::Init, abs::Free, abs::Prepare,
                                 abs::Eval<abs::kReference>};
  return &r;
}

TfLiteRegistration* Register_ABS() { return Register_ABS_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
