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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {

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

inline std::int32_t RoundingDivideByPOT(std::int32_t numerator, int exponent) {
  std::int32_t sign = numerator >= 0 ? 1 : -1;
  std::int32_t abs_numerator = std::abs(numerator);
  std::int32_t mask = (1LL << exponent) - 1;
  std::int32_t remainder = abs_numerator & mask;
  std::int32_t threshold = mask >> 1;
  std::int32_t abs_result =
      (abs_numerator >> exponent) + (remainder > threshold ? 1 : 0);
  return sign * abs_result;
}

inline void HardSwishFloatOp(const RuntimeShape& input_shape, const float* input_data,
                           const RuntimeShape& output_shape, float* output_data) {
  auto matching_size = MatchingFlatSize(input_shape, output_shape);
  const float* in_end = input_data + matching_size;
  for (; input_data < in_end; input_data++, output_data++) {
    const float in = *input_data;
    *output_data =
        in * std::min(static_cast<float>(6), std::max(static_cast<float>(0), in + 3)) /
        6;
  }
}

template <typename T>
void HardSwishOp(HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int16_t input_value = input_data[i] - params.input_zero_point;
    // Left-shift as much as we can without overflow/saturation to put
    // significant bits in the high bits of our 16-bit fixedpoint values, so
    // that fixed-point approximate computations below are as accurate as
    // possible.
    const int16_t input_value_on_hires_input_scale = input_value << 7;
    // Compute the input value on essentially the output scale, just not
    // right-shifted yet. This is the value that we'll use in the (x >= +3)
    // case, and that in the general case we'll multiply against the "relu-ish"
    // fixed-point multiplier in [0, 1].
    const int16_t input_value_on_preshift_output_scale =
        SaturatingRoundingDoublingHighMul(input_value_on_hires_input_scale,
                                          params.output_multiplier_fixedpoint_int16);
    // Now compute the "relu-ish multiplier". In the (-3 <= x <= +3) case, that
    // is just an affine rescaling of x from [-3, 3] to [0, 1]. In the general
    // case, it is just that plus saturation at the boundaries of [-3, 3].
    // First, we rescale from [-3, 3] to [-1, 1], saturating.
    // That is done by rescaling the input value with a fixed-point multiplier
    // (reluish_multiplier_fixedpoint) and bit-shift such that we represent
    // that input value on the scale where the real value 3.0f is represented
    // by the quantized value 32768.  (+32768 is actually not representable as
    // int16, so this saturates at +32767, and that is seen empirically to be
    // a negligible contribution to numerical error/bias).
    //
    // This code is careful to correctly implement any magnitude of multiplier,
    // involving either a right shift or a left shift, with correct saturation
    // behavior in the left-shift case. This forces this code to be more
    // complicated, but is necessary for real applications: a partially
    // trained quantized MobileNet v3-small model that motivated this code
    // exhibits some large [min, max] range boundaries, of the order of
    // magnitude of 10 or 100 depending on layers.
    //
    // The next few lines are basically just an ordinary
    // MultiplyByQuantizedMultiplier, except that we are more careful here
    // about the fine details of saturation when left-shifting, because here
    // overflow in left-shift is a common case, not an anomaly as
    // MultiplyByQuantizedMultiplier assumes.
    int16_t reluish_value = input_value_on_hires_input_scale;
    // Shift left, saturating, as much as we can while ensuring that this
    // saturation will not contribute to the result. That is, left shift amount
    // reduced by 1.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(
          reluish_value, params.reluish_multiplier_exponent - 1);
    }
    // Apply the fixed-point multiplier, dividing the value by a divisor
    // ranging in [1, 2].
    reluish_value = SaturatingRoundingDoublingHighMul(reluish_value, params.reluish_multiplier_fixedpoint_int16);
    // Apply the last bit of left-shift. Thus, in the left-shifting case, if
    // any saturation affects the result, it is happening here --- any
    // saturation having occurred above is overwritten here, not affecting the
    // result.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(reluish_value, 1);
    }
    // Shift right, in the right-shifting case.
    if (params.reluish_multiplier_exponent < 0) {
      reluish_value = RoundingDivideByPOT(
          reluish_value, -params.reluish_multiplier_exponent);
    }
    // At this point we have rescaled the value into a 16bit fixedpoint
    // reluish_value in [-1, 1].
    // We now convert that to a 16bit fixedpoint value in [0, 1].
    reluish_value = (reluish_value + (1 << 15)) >> 1;
    // Use of SaturatingDoublingHighMul here is important to cancel the biases
    // from the above SaturatingRoundingDoublingHighMul.
    //
    const int16_t preshift_output_value = SaturatingDoublingHighMul(
        reluish_value, input_value_on_preshift_output_scale);
    // We were so far operating on the pre-shift output scale. Now we finally
    // apply that output shift, arriving at the final output scale.
    int16_t output_value = RoundingDivideByPOT(
        preshift_output_value, -params.output_multiplier_exponent);
    output_value += params.output_zero_point;
    output_value =
        std::min<int16_t>(output_value, std::numeric_limits<T>::max());
    output_value =
        std::max<int16_t>(output_value, std::numeric_limits<T>::min());
    output_data[i] = output_value;
  }
}


template <typename Q>
TfLiteStatus HardSwishQuantized(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  HardSwishParams params;

   params.input_zero_point = input->params.zero_point;
   params.output_zero_point = output->params.zero_point;

   const float input_scale = input->params.scale;
   const float hires_input_scale = (1.0f / 128.0f) * input_scale;
   const float reluish_scale = 3.0f / 32768.0f;
   const float output_scale = output->params.scale;

   const double output_multiplier = static_cast<double>(hires_input_scale / output_scale);
   int32_t output_multiplier_fixedpoint_int32;
   QuantizeMultiplier(output_multiplier, &output_multiplier_fixedpoint_int32,
                      &params.output_multiplier_exponent);
   DownScaleInt32ToInt16Multiplier(
       output_multiplier_fixedpoint_int32,
       &params.output_multiplier_fixedpoint_int16);

   TF_LITE_ENSURE(context, params.output_multiplier_exponent <= 0);

   const double reluish_multiplier = static_cast<double>(hires_input_scale / reluish_scale);
   int32_t reluish_multiplier_fixedpoint_int32;
   QuantizeMultiplier(reluish_multiplier, &reluish_multiplier_fixedpoint_int32,
                      &params.reluish_multiplier_exponent);
   DownScaleInt32ToInt16Multiplier(
       reluish_multiplier_fixedpoint_int32,
       &params.reluish_multiplier_fixedpoint_int16);

   HardSwishOp<Q>(params, GetTensorShape(input),
                GetTensorData<Q>(input), GetTensorShape(output), GetTensorData<Q>(output));
   return kTfLiteOk;
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
      TF_LITE_KERNEL_LOG(context, "Only float32/int8/uint8 are supported currently, got %s",
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
      TF_LITE_KERNEL_LOG(context, "Only float32/int8/uint8 are supported currently, got %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

TfLiteStatus HardSwishPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      HardSwishFloatOp(
          GetTensorShape(input),
          GetTensorData<float>(input),
          GetTensorShape(output),
          GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      return HardSwishQuantized<uint8>(context, node);
    } break;
    case kTfLiteInt8: {
      return HardSwishQuantized<int8>(context, node);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Only float32/int8/uint8 are supported currently, got %s",
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

TfLiteRegistration* Register_HARD_SWISH() {
  static TfLiteRegistration r = {};
  r.prepare = activations::HardSwishPrepare;
  r.invoke = activations::HardSwishEval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
