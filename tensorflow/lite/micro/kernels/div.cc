/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/div.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace div {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  // Parameters used in the quantized paths where the output is 8bit
  int32_t input1_zero_point;
  int32_t input2_zero_point;
  int32_t output_zero_point;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // Parameters used in all quantized paths
  int32_t output_multiplier;
  int output_shift;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDivParams* params, OpData* data) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
    const double real_multiplier =
        input1->params.scale / (input2->params.scale * output->params.scale);
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
    data->input1_zero_point = input1->params.zero_point;
    data->input2_zero_point = input2->params.zero_point;
    data->output_zero_point = output->params.zero_point;
  }

  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteDivParams*>(node->builtin_data);
  auto* data = static_cast<OpData*>(node->user_data);
  return CalculateOpData(context, node, params, data);
}

void EvalDiv(TfLiteContext* context, TfLiteNode* node, TfLiteDivParams* params,
             const OpData* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params = {};

#define TF_LITE_DIV(type, opname, data_type)                           \
  data_type output_activation_min, output_activation_max;              \
  CalculateActivationRange(params->activation, &output_activation_min, \
                           &output_activation_max);                    \
  SetActivationParams(output_activation_min, output_activation_max,    \
                      &op_params);                                     \
  type::opname(op_params, tflite::micro::GetTensorShape(input1),       \
               tflite::micro::GetTensorData<data_type>(input1),        \
               tflite::micro::GetTensorShape(input2),                  \
               tflite::micro::GetTensorData<data_type>(input2),        \
               tflite::micro::GetTensorShape(output),                  \
               tflite::micro::GetTensorData<data_type>(output))

  bool requires_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (output->type == kTfLiteInt32) {
    if (requires_broadcast) {
      TF_LITE_DIV(reference_ops, BroadcastDivSlow, int32_t);
    } else {
      TF_LITE_DIV(reference_ops, Div, int32_t);
    }
  } else if (output->type == kTfLiteFloat32) {
    if (requires_broadcast) {
      TF_LITE_DIV(reference_ops, BroadcastDivSlow, float);
    } else {
      TF_LITE_DIV(reference_ops, Div, float);
    }
  }
#undef TF_LITE_DIV
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDivParams* params, const OpData* data,
                           const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params = {};

#define TF_LITE_DIV(type, opname, dtype)                         \
  type::opname(op_params, tflite::micro::GetTensorShape(input1), \
               tflite::micro::GetTensorData<dtype>(input1),      \
               tflite::micro::GetTensorShape(input2),            \
               tflite::micro::GetTensorData<dtype>(input2),      \
               tflite::micro::GetTensorShape(output),            \
               tflite::micro::GetTensorData<dtype>(output))

  if (input1->type == kTfLiteUInt8 && input2->type == kTfLiteUInt8 &&
      output->type == kTfLiteUInt8) {
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    op_params.input1_offset = -data->input1_zero_point;
    op_params.input2_offset = -data->input2_zero_point;
    op_params.output_offset = data->output_zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;

    bool requires_broadcast = reference_ops::ProcessBroadcastShapes(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorShape(input2), &op_params);

    if (requires_broadcast) {
      TF_LITE_DIV(reference_ops, BroadcastDivSlow, uint8_t);
    } else {
      TF_LITE_DIV(reference_ops, Div, uint8_t);
    }
#undef TF_LITE_DIV
  } else {
    TF_LITE_KERNEL_LOG(
        context, "Unsupported combination of input and output types in DIV.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = static_cast<TfLiteDivParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt32) {
    EvalDiv(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_OK(context, EvalQuantized(context, node, params, data,
                                             input1, input2, output));
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "DIV only supports FLOAT32, INT32 and quantized UINT8 "
                       "now, got type %s (%d).",
                       TfLiteTypeGetName(output->type), output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace div

TfLiteRegistration Register_DIV() {
  return {/*init=*/div::Init,
          /*free=*/nullptr,
          /*prepare=*/div::Prepare,
          /*invoke=*/div::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
