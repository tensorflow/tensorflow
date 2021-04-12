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
#include "tensorflow/lite/kernels/internal/reference/log_softmax.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

// used only with quantized data
struct LogSoftmaxOpData {
  int32_t input_multiplier;
  int32_t input_left_shift;
  int32_t reverse_scaling_divisor;
  int32_t reverse_scaling_right_shift;
  int diff_min;
  size_t outer_size;  // number of tensor elements skipping computation axis
  size_t depth;       // number of tensor elements on computation axis
};

// input/output tensor index
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  if (input->type == kTfLiteInt8) {
    node->user_data =
        context->AllocatePersistentBuffer(context, sizeof(LogSoftmaxOpData));
    auto data = static_cast<LogSoftmaxOpData*>(node->user_data);

    // quantization datum
    constexpr int32_t kOutputZeroPoint = 127;
    constexpr float kOutputScale = 16.0 / 256;
    constexpr double kBeta = 1.0;
    constexpr int kScaledDiffIntegerBits = 5;

    TF_LITE_ENSURE(context, output->params.scale == kOutputScale);
    TF_LITE_ENSURE(context, output->params.zero_point == kOutputZeroPoint);

    int input_left_shift;
    int reverse_scaling_right_shift;
    tflite::PreprocessLogSoftmaxScalingExp(
        kBeta, static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
        &data->input_multiplier, &input_left_shift,
        &data->reverse_scaling_divisor, &reverse_scaling_right_shift);
    data->input_left_shift = static_cast<int32_t>(input_left_shift);
    data->reverse_scaling_right_shift =
        static_cast<int32_t>(-reverse_scaling_right_shift);
    // diff_min has a negative value, and is used to limit the maximum magnitude
    // of the diffs, which are <= 0.
    data->diff_min =
        -tflite::CalculateInputRadius(kScaledDiffIntegerBits, input_left_shift);

    RuntimeShape input_shape = GetTensorShape(input);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    data->outer_size =
        static_cast<size_t>(FlatSizeSkipDim(input_shape, trailing_dim));
    data->depth = static_cast<size_t>(input_shape.Dims(trailing_dim));
  }

  return kTfLiteOk;
}

TfLiteStatus LogSoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

TfLiteStatus LogSoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const LogSoftmaxOpData* data =
      static_cast<LogSoftmaxOpData*>(node->user_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteFloat32: {
      SoftmaxParams op_params = {};
      reference_ops::LogSoftmax(op_params, tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      SoftmaxParams op_params = {};
      op_params.input_multiplier = data->input_multiplier;
      op_params.input_left_shift = data->input_left_shift;
      op_params.reverse_scaling_divisor = data->reverse_scaling_divisor;
      op_params.reverse_scaling_right_shift = data->reverse_scaling_right_shift;
      op_params.diff_min = data->diff_min;
      reference_ops::LogSoftmax(op_params, data->outer_size, data->depth,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<int8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "LOG_SOFTMAX only supports float32, int8, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace

TfLiteRegistration Register_LOG_SOFTMAX() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/LogSoftmaxPrepare,
          /*invoke=*/LogSoftmaxEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
