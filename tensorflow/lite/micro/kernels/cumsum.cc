/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/cumsum.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

constexpr int kCumSumIntegerShift = 20;

// only used with INT8 tensors
struct OpData {
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t input_offset;
  int32_t output_offset;
  int32_t input_multiplier;
  int32_t output_multiplier;
  int input_shift;
  int output_shift;
  int left_shift;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  if (output->type == kTfLiteInt8) {
    node->user_data =
        context->AllocatePersistentBuffer(context, sizeof(OpData));
    OpData* data = static_cast<OpData*>(node->user_data);

    // 8bit -> 8bit general quantized path, with general rescalings
    data->input_offset = -input->params.zero_point;
    data->output_offset = output->params.zero_point;
    data->left_shift = kCumSumIntegerShift;
    const double twice_max_input_scale =
        2 * static_cast<double>(input->params.scale);
    const double real_input_multiplier =
        static_cast<double>(input->params.scale) / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << data->left_shift) * static_cast<double>(output->params.scale));

    QuantizeMultiplierSmallerThanOneExp(
        real_input_multiplier, &data->input_multiplier, &data->input_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, kTfLiteActNone, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis_tensor =
      tflite::micro::GetEvalInput(context, node, kAxisTensor);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  auto* cs_params = static_cast<TfLiteCumsumParams*>(node->builtin_data);
  auto input_shape = tflite::micro::GetTensorShape(input);

  int32_t axis = *tflite::micro::GetTensorData<int32_t>(axis_tensor);
  if (axis < 0) axis += input_shape.DimensionsCount();

  if (axis < 0 || axis >= input_shape.DimensionsCount()) {
    TF_LITE_KERNEL_LOG(context, "CUMSUM Invalid axis: %d", axis);
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::CumSum(tflite::micro::GetTensorData<float>(input),
                            input_shape, axis, cs_params->exclusive,
                            cs_params->reverse,
                            tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;

    case kTfLiteInt8: {
      auto* data = static_cast<OpData*>(node->user_data);
      ArithmeticParams params;
      params.left_shift = data->left_shift;
      params.input1_offset = data->input_offset;
      params.input1_multiplier = data->input_multiplier;
      params.input1_shift = data->input_shift;
      params.output_offset = data->output_offset;
      params.output_multiplier = data->output_multiplier;
      params.output_shift = data->output_shift;
      SetActivationParams(data->output_activation_min,
                          data->output_activation_max, &params);
      reference_ops::CumSum(params, tflite::micro::GetTensorData<int8_t>(input),
                            input_shape, axis, cs_params->exclusive,
                            cs_params->reverse,
                            tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;

    default: {
      TF_LITE_KERNEL_LOG(context,
                         "CUMSUM only supports FLOAT32 and INT8, got %s.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteError;
}

}  // namespace

TfLiteRegistration Register_CUMSUM() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
