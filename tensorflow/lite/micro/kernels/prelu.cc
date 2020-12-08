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

#include "tensorflow/lite/kernels/internal/reference/prelu.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

TfLiteStatus CalculatePreluParams(const TfLiteTensor* input,
                                  const TfLiteTensor* alpha,
                                  TfLiteTensor* output, PreluParams* params) {
  if (output->type == kTfLiteInt8 || output->type == kTfLiteUInt8 ||
      output->type == kTfLiteInt16) {
    double real_multiplier_1 = static_cast<double>(input->params.scale) /
                               static_cast<double>(output->params.scale);
    double real_multiplier_2 = static_cast<double>(input->params.scale) *
                               static_cast<double>(alpha->params.scale) /
                               static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier_1, &params->output_multiplier_1,
                       &params->output_shift_1);
    QuantizeMultiplier(real_multiplier_2, &params->output_multiplier_2,
                       &params->output_shift_2);

    params->input_offset = -input->params.zero_point;
    params->alpha_offset = -alpha->params.zero_point;
    params->output_offset = output->params.zero_point;
  }

  return kTfLiteOk;
}

}  // namespace

inline void BroadcastPrelu4DSlowFloat(
    const RuntimeShape& unextended_input1_shape, const float* input1_data,
    const RuntimeShape& unextended_input2_shape, const float* input2_data,
    const RuntimeShape& unextended_output_shape, float* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = in1_val >= 0.0f ? in1_val : in1_val * in2_val;
        }
      }
    }
  }
}

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(PreluParams));
}

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  PreluParams* params = static_cast<PreluParams*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* alpha = GetInput(context, node, 1);
  TF_LITE_ENSURE(context, alpha != nullptr);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  return CalculatePreluParams(input, alpha, output, params);
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const PreluParams& params =
      *(static_cast<const PreluParams*>(node->user_data));

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* alpha = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteFloat32: {
      BroadcastPrelu4DSlowFloat(tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(alpha),
                                tflite::micro::GetTensorData<float>(alpha),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<uint8_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<uint8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Only float32 and uint8_t are supported currently, got %d.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration Register_PRELU() {
  return {/*init=*/activations::PreluInit,
          /*free=*/nullptr,
          /*prepare=*/activations::PreluPrepare,
          /*invoke=*/activations::PreluEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
