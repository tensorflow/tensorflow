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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {
#if !defined(XTENSA)
FullyConnectedParams ToFloatParams(TfLiteFusedActivation activation) {
  FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}
#endif  // !defined(XTENSA)
}  // namespace

const int kFullyConnectedInputTensor = 0;
const int kFullyConnectedWeightsTensor = 1;
const int kFullyConnectedBiasTensor = 2;
const int kFullyConnectedOutputTensor = 0;

void* InitFullyConnectedReference(TfLiteContext* context, const char* buffer,
                                  size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(
      context, sizeof(OpDataFullyConnectedReference));
}

#if !defined(XTENSA)
TfLiteStatus CalculateOpDataFullyConnectedReference(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnectedReference* data) {
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);

    data->input_zero_point = input->params.zero_point;
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    return CalculateActivationRangeQuantized(context, activation, output,
                                             &data->output_activation_min,
                                             &data->output_activation_max);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloatFullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node, TfLiteFusedActivation activation,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  tflite::reference_ops::FullyConnected(
      ToFloatParams(activation), tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8FullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataFullyConnectedReference& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  reference_integer_ops::FullyConnected(
      data.ToQuantizedParams(), tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedFullyConnectedReference(
    TfLiteContext* context, TfLiteNode* node,
    const OpDataFullyConnectedReference& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
#define TF_LITE_FULLY_CONNECTED(output_data_type)                     \
  reference_ops::FullyConnected(                                      \
      data.ToQuantizedParams(), tflite::micro::GetTensorShape(input), \
      tflite::micro::GetTensorData<uint8_t>(input),                   \
      tflite::micro::GetTensorShape(filter),                          \
      tflite::micro::GetTensorData<uint8_t>(filter),                  \
      tflite::micro::GetTensorShape(bias),                            \
      tflite::micro::GetTensorData<int32_t>(bias),                    \
      tflite::micro::GetTensorShape(output),                          \
      tflite::micro::GetTensorData<output_data_type>(output))
  switch (output->type) {
    case kTfLiteUInt8:
      TF_LITE_FULLY_CONNECTED(uint8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_FULLY_CONNECTED(int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

#endif  // !defined(XTENSA)

}  // namespace tflite
