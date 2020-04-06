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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {

namespace {
struct TanhOpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};
}  // namespace
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename Q>
inline void ReluQuantized(int32_t lower, const RuntimeShape& input_shape,
                          const Q* input_data, const RuntimeShape& output_shape,
                          Q* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const Q val = input_data[i];
    const Q clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void ReluFloat(const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float lower = 0.0f;
    const float clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu6Float(const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6.0f;
    const float lower = 0.0f;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename Q>
inline void Relu6Quantized(Q lower, Q upper, const RuntimeShape& input_shape,
                           const Q* input_data,
                           const RuntimeShape& output_shape, Q* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const Q val = input_data[i];
    const Q clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      ReluFloat(GetTensorShape(input), GetTensorData<float>(input),
                GetTensorShape(output), GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      ReluQuantized<int8_t>(input->params.zero_point, GetTensorShape(input),
                            GetTensorData<int8_t>(input),
                            GetTensorShape(output),
                            GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      ReluQuantized<uint8_t>(input->params.zero_point, GetTensorShape(input),
                             GetTensorData<uint8_t>(input),
                             GetTensorShape(output),
                             GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Only float32 is supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

TfLiteStatus Relu6Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      Relu6Float(GetTensorShape(input), GetTensorData<float>(input),
                 GetTensorShape(output), GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const int8_t six = FloatToAsymmetricQuantizedInt8(
          6.0f, input->params.scale, input->params.zero_point);
      const int8_t zero = input->params.zero_point;
      Relu6Quantized<int8_t>(
          zero, six, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      const uint8_t six = FloatToAsymmetricQuantizedUInt8(
          6.0f, input->params.scale, input->params.zero_point);
      const uint8_t zero = input->params.zero_point;
      Relu6Quantized<uint8_t>(
          zero, six, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Only float32 is supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node,
                         TanhOpData* data) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
        input->params.scale *
        static_cast<double>(1 << (15 - kInputIntegerBits));

    const double q = std::frexp(input_real_multiplier, &data->input_left_shift);
    auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1ll << 15)));
    data->input_multiplier = static_cast<int16_t>(q_fixed);

    int16_t input_range_radius =
        CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
    data->input_range_radius = input_range_radius;
  }

  if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // These operators are implemented in fixed-point arithmetic,
    // which intrinsically wants symmetric ranges (zero_point==0)
    // and power-of-two scales (power-of-two is abbreviated below as POT).
    // While more general support would be possible by means of rescaling,
    // that would add some overhead and some loss of accuracy and wouldn't
    // be used at the moment as current quantized LSTM applications are
    // happy with symmetric, power-of-two-scales quantization. So we just
    // implement that narrow case only for now.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    TF_LITE_ENSURE(context,
                   CheckedLog2(input->params.scale, &input_scale_log2_rounded));

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    // Support for shifts is limited until we have a parameterized version of
    // SaturatingRoundingMultiplyByPOT().
    TF_LITE_ENSURE(context, data->input_left_shift >= 0);
    TF_LITE_ENSURE(context, data->input_left_shift <= 1);
  }

  return kTfLiteOk;
}

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  TanhOpData data;
  TanhPrepare(context, node, &data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      TanhParams params;
      params.input_left_shift = data.input_left_shift;
      reference_ops::Tanh(params, GetTensorShape(input),
                          GetTensorData<int16_t>(input), GetTensorShape(output),
                          GetTensorData<int16_t>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      TanhParams params;
      params.input_zero_point = input->params.zero_point;
      params.input_range_radius = data.input_range_radius;
      params.input_multiplier = data.input_multiplier;
      params.input_left_shift = data.input_left_shift;
      optimized_ops::Tanh16bitPrecision(
          params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(output), GetTensorData<uint8_t>(output));

      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      TanhParams params;
      params.input_zero_point = input->params.zero_point;
      params.input_range_radius = data.input_range_radius;
      params.input_multiplier = data.input_multiplier;
      params.input_left_shift = data.input_left_shift;
      optimized_ops::Tanh16bitPrecision(
          params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));

      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32, uint8, int16 and int8 are supported "
                           "currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_RELU() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::ReluPrepare,
                                 /*invoke=*/activations::ReluEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_RELU6() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::Relu6Prepare,
                                 /*invoke=*/activations::Relu6Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_TANH() {
  static TfLiteRegistration r = {/*init*/ nullptr,
                                 /*free*/ nullptr,
                                 /*prepare*/ nullptr,
                                 activations::TanhEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite
