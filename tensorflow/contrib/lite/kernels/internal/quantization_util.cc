/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"

namespace tflite {

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  int64_t q_fixed = IntegerFrExp(double_multiplier, shift);
  TFLITE_CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

int64_t IntegerFrExp(double input, int* shift) {
  // Double-precision floating point format is:
  // Bit |  63  |  62-52   |   51-0   |
  //     | Sign | Exponent | Fraction |
  // To avoid 64-bit integers as much as possible, I break this into high and
  // low 32-bit chunks. High is:
  // Bit |  31  |  30-20   |      19-0     |
  //     | Sign | Exponent | High Fraction |
  // Low is:
  // Bit |     31-0     |
  //     | Low Fraction |
  // We then access the components through logical bit-wise operations to 
  // extract the parts needed, with the positions and masks derived from the
  // layout shown above.
  const uint32_t sign_mask = 0x80000000;
  const uint32_t exponent_mask = 0x7ff00000;
  const int32_t exponent_shift = 20;
  const int32_t exponent_bias = 1023;
  const uint32_t fraction_mask_high = 0x000fffff;
  const uint32_t fraction_shift_high = 10;
  const uint32_t fraction_mask_low = 0xffc00000;
  const uint32_t fraction_rounding_mask = 0x003fffff;
  const uint32_t fraction_rounding_threshold = 0x00200000;
  const uint32_t fraction_shift_low = 22;

  // We want to access the bits of the input double value directly, which is
  // tricky to do safely, so use a union to handle the casting.
  union {
    double double_value;
    uint32_t double_as_uints[2];
  } cast_union;
  cast_union.double_value = input;
  const uint32_t u_low = cast_union.double_as_uints[0];
  const uint32_t u_high = cast_union.double_as_uints[1];

  // If the bitfield is all zeros, this is a normalized zero value, so return
  // standard values for this special case.
  if ((u_low == 0) && (u_high == 0)) {
    *shift = 0;
    return 0;
  }
  
  // The shift is fairly easy to extract from the high bits of the double value,
  // just by masking it out and applying a bias. The std::frexp() implementation
  // always returns values between 0.5 and 1.0 though, whereas the exponent
  // assumes 1.0 to 2.0 is the standard range, so I add on one to match that
  // interface.
  *shift = (((u_high & exponent_mask) >> exponent_shift) - exponent_bias) + 1;

  // There's an implicit high bit in the double format definition, so make sure
  // we include that at the top, and then reconstruct the rest of the fractional
  // value from the upper and lower fragments.
  int64_t fraction = 0x40000000 + 
    (((u_high & fraction_mask_high) << fraction_shift_high) |
     ((u_low & fraction_mask_low) >> fraction_shift_low));
  // We're cutting off some bits at the bottom, so to exactly match the standard
  // frexp implementation here we'll apply rounding by adding one to the least
  // significant bit of the result if the discarded portion is over half of the
  // maximum.
  if ((u_low & fraction_rounding_mask) > fraction_rounding_threshold) {
    fraction += 1;
  }
  // Negate the fraction if the sign bit was set.
  if (u_high & sign_mask) {
    fraction *= -1;
  }

  return fraction;
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
  TFLITE_CHECK_GT(double_multiplier, 1.);
  QuantizeMultiplier(double_multiplier, quantized_multiplier, left_shift);
  TFLITE_CHECK_GE(*left_shift, 0);
}

void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift) {
  TFLITE_CHECK_LT(double_multiplier, 1.);
  TFLITE_CHECK_GT(double_multiplier, 0.);
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  TFLITE_CHECK_LE(shift, 0);
  *left_shift = shift;
}

void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift) {
  // If the overall multiplier (input and beta) is large, then exp() of an
  // input difference of 1 scaled by this will be large.  In other words, we
  // can cap the multiplier and know that, when it is used, the output will be
  // (round to) zero wherever the input is not at the maximum value.

  // If the overall scale is less than one, and input_integer_bits=0, then the
  // result is double equivalent of Q0.31 (actually with more precision). Thus
  // this generates a Q(input_integer_bits).(31-input_integer_bits)
  // representation.
  const double input_beta_real_multiplier = std::min(
      beta * input_scale * (1 << (31 - input_integer_bits)), (1ll << 31) - 1.0);

  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                   quantized_multiplier, left_shift);
}

void PreprocessLogSoftmaxScalingExp(double beta, double input_scale,
                                    int input_integer_bits,
                                    int32_t* quantized_multiplier,
                                    int* left_shift,
                                    int32_t* reverse_scaling_divisor,
                                    int* reverse_scaling_left_shift) {
  PreprocessSoftmaxScaling(beta, input_scale, input_integer_bits,
                           quantized_multiplier, left_shift);

  // Also calculate what amounts to the inverse scaling factor for the input.
  const double real_reverse_scaling_divisor =
      (1 << (31 - *left_shift)) / static_cast<double>(*quantized_multiplier);
  tflite::QuantizeMultiplierSmallerThanOneExp(real_reverse_scaling_divisor,
                                              reverse_scaling_divisor,
                                              reverse_scaling_left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift) {
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                    (1ll << (31 - input_integer_bits)) /
                                    (1ll << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int>(std::floor(max_input_rescaled));
}

void NudgeQuantizationRange(const float min, const float max,
                            const int quant_min, const int quant_max,
                            float* nudged_min, float* nudged_max,
                            float* nudged_scale) {
  // This code originates from tensorflow/core/kernels/fake_quant_ops_functor.h.
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *nudged_scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / *nudged_scale;
  uint16 nudged_zero_point;
  if (zero_point_from_min < quant_min_float) {
    nudged_zero_point = static_cast<uint16>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    nudged_zero_point = static_cast<uint16>(quant_max);
  } else {
    nudged_zero_point = static_cast<uint16>(TfLiteRound(zero_point_from_min));
  }
  *nudged_min = (quant_min_float - nudged_zero_point) * (*nudged_scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*nudged_scale);
}

void FakeQuantizeArray(const float nudged_scale, const float nudged_min,
                       const float nudged_max, const float* input_data,
                       float* output_data, const float size) {
  // This code originates from tensorflow/core/kernels/fake_quant_ops_functor.h.
  const float inv_nudged_scale = 1.0f / nudged_scale;

  for (int i = 0; i < size; i++) {
    const float src_val = input_data[i];
    const float clamped = std::min(nudged_max, std::max(nudged_min, src_val));
    const float clamped_shifted = clamped - nudged_min;
    const float dst_val =
        TfLiteRound(clamped_shifted * inv_nudged_scale) * nudged_scale +
        nudged_min;
    output_data[i] = dst_val;
  }
}

bool CheckedLog2(const float x, int* log2_result) {
  // Using TfLiteRound instead of std::round and std::log instead of
  // std::log2 to work around these fuctions being missing in a toolchain
  // used in some TensorFlow tests as of May 2018.
  const float x_log2 = std::log(x) * (1.0f / std::log(2.0f));
  const float x_log2_rounded = TfLiteRound(x_log2);
  const float x_log2_fracpart = x_log2 - x_log2_rounded;

  *log2_result = static_cast<int>(x_log2_rounded);
  return std::abs(x_log2_fracpart) < 1e-3;
}

}  // namespace tflite
