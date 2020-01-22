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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

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

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* alpha = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t output_multiplier = 0;
  int output_shift = 0;
  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt16) {
    double real_multiplier = static_cast<double>(input->params.scale) *
                             static_cast<double>(alpha->params.scale) /
                             static_cast<double>(output->params.scale);
    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier,
                                        &output_shift);
  }
  switch (input->type) {
    case kTfLiteFloat32: {
      BroadcastPrelu4DSlowFloat(
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(alpha), GetTensorData<float>(alpha),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      PreluParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.alpha_offset = -alpha->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier = output_multiplier;
      op_params.output_shift = output_shift;
      reference_ops::BroadcastPrelu4DSlow(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(alpha), GetTensorData<uint8_t>(alpha),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(
          context, "Only float32 and uint8 are supported currently, got %d.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_PRELU() {
  static TfLiteRegistration r = {nullptr, nullptr, activations::PreluPrepare,
                                 activations::PreluEval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
