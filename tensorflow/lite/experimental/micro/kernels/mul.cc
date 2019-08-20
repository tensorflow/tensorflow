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

#include "tensorflow/lite/kernels/internal/reference/mul.h"

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace mul {

constexpr int kInput1Tensor = 0;
constexpr int kInput2Tensor = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t output_multiplier;
  int output_shift;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteMulParams* params, OpData* data) {
  const TfLiteTensor* input1 = GetInput(context, node, kInput1Tensor);
  const TfLiteTensor* input2 = GetInput(context, node, kInput2Tensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);

  if (output->type == kTfLiteUInt8) {
    CalculateActivationRangeUint8(params->activation, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }

  double real_multiplier =
      input1->params.scale * input2->params.scale / output->params.scale;
  QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                     &data->output_shift);

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteMulParams* params, OpData* data,
                   const TfLiteTensor* input1, const TfLiteTensor* input2,
                   TfLiteTensor* output) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);
  op_params.input1_offset = -input1->params.zero_point;
  op_params.input2_offset = -input2->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;

  reference_ops::Mul(op_params, GetTensorShape(input1),
                     GetTensorData<uint8_t>(input1), GetTensorShape(input2),
                     GetTensorData<uint8_t>(input2), GetTensorShape(output),
                     GetTensorData<uint8_t>(output));
}

void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteMulParams* params, OpData* data,
               const TfLiteTensor* input1, const TfLiteTensor* input2,
               TfLiteTensor* output) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);

  reference_ops::Mul(op_params, GetTensorShape(input1),
                     GetTensorData<float>(input1), GetTensorShape(input2),
                     GetTensorData<float>(input2), GetTensorShape(output),
                     GetTensorData<float>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input1 = GetInput(context, node, kInput1Tensor);
  const TfLiteTensor* input2 = GetInput(context, node, kInput2Tensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  CalculateOpData(context, node, params, &data);

  switch (input1->type) {
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, &data, input1, input2, output);
      break;
    case kTfLiteFloat32:
      EvalFloat(context, node, params, &data, input1, input2, output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input1->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace mul

TfLiteRegistration* Register_MUL() {
  static TfLiteRegistration r = {mul::Init, mul::Free, mul::Prepare, mul::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
