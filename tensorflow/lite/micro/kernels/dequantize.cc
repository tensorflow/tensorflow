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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace dequantize {

struct OpData {
  tflite::DequantizationParams quantization_params;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  int32_t output_zero_point;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // TODO(b/140515557): Add cached dequant to improve hybrid model performance.
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, input->type == kTfLiteUInt8 ||
                              input->type == kTfLiteInt8 ||
                              input->type == kTfLiteInt16);
  TF_LITE_ENSURE(
      context, output->type == kTfLiteFloat32 || output->type == kTfLiteInt32);

  if (output->type == kTfLiteInt32) {
    const double effective_output_scale =
        static_cast<double>(input->params.scale) /
        static_cast<double>(output->params.scale);
    QuantizeMultiplier(effective_output_scale, &data->output_multiplier,
                       &data->output_shift);
  }

  data->quantization_params.zero_point = input->params.zero_point;
  data->quantization_params.scale = static_cast<double>(input->params.scale);
  data->output_zero_point = output->params.zero_point;
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (output->type == kTfLiteFloat32) {
    switch (input->type) {
      case kTfLiteUInt8:
        reference_ops::Dequantize(data->quantization_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<uint8_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
        break;
      case kTfLiteInt8:
        reference_ops::Dequantize(data->quantization_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<int8_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
        break;
      case kTfLiteInt16:
        reference_ops::Dequantize(data->quantization_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<int16_t>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                           TfLiteTypeGetName(input->type),
                           TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (output->type == kTfLiteInt32) {
    int flat_size = MatchingFlatSize(tflite::micro::GetTensorShape(input),
                                     tflite::micro::GetTensorShape(output));
    switch (input->type) {
      case kTfLiteInt16: {
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int16_t>(input), flat_size,
            data->output_multiplier, data->output_shift,
            data->quantization_params.zero_point, data->output_zero_point,
            tflite::micro::GetTensorData<int32_t>(output));
        break;
      }
      case kTfLiteInt8: {
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int8_t>(input), flat_size,
            data->output_multiplier, data->output_shift,
            data->quantization_params.zero_point, data->output_zero_point,
            tflite::micro::GetTensorData<int32_t>(output));
        break;
      }
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

}  // namespace dequantize

TfLiteRegistration Register_DEQUANTIZE() {
  return {/*init=*/dequantize::Init,
          /*free=*/nullptr,
          /*prepare=*/dequantize::Prepare,
          /*invoke=*/dequantize::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
