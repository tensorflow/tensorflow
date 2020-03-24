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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_FIXEDPOINT_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_FIXEDPOINT_UTILS_H_

#include <stdint.h>
#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/micro/kernels/xtensa_hifimini/utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xtensa {
namespace hifimini {

//
// Product of two fixed-point 24bit integers with right shift.
//
// Two 24bit integers from the HH side of a PR register entry are MAC into a QR
// register. That value will be right shifted if |shift_length| is greater than
// 0.
//
inline ae_q56s SaturatingMultiply(ae_p24x2s a_56, ae_p24x2s b_56,
                                  int shift_length) {
  ae_q56s result_56 = AE_ZEROQ56();
  AE_MULAS56P24S_HH(result_56, a_56, b_56);
  if (shift_length > 0) {
    return AE_Q56S_SRA(result_56, shift_length);
  }
  return result_56;
}

//
// Multiply 32bit value by a quantized multiplier (w/ shift) and returns a 48bit
// aligned value in the QR register.
//
inline ae_q56s MultiplyByQuantizedMultiplier(int32_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {
  // These boolean factors will carry an additional 2^8 (e.g 256) factor
  // throughout the equation to cover the missing 8 bits of precision when a
  // 32bit integer is outside the bounds of INT24. The additional scaling factor
  // will be adjusted on the final SaturatingMultiply() call in this method.
  //
  // The Q-notation comments in this method describe the calculations that take
  // place when both |x| and the shifted value of |1| overflow the INT24 limits.
  bool x_exceeds_24bits = (x <= INT24_MIN || x >= INT24_MAX);
  bool shift_exceeds_24bits = false;

  // Q31.0 -> Q23.0 / 2^8
  ae_p24x2s x_24x2 = AE_CONVERT_INT32_24x2(x);

  if (shift > 0) {
    int shifted = 1 << shift;
    if (shifted <= INT24_MIN || shifted >= INT24_MAX) {
      shift_exceeds_24bits = true;
    }

    // Load the shifted value into the PR register:
    // Q31.0 -> Q23.0 / 2^8
    ae_p24x2s shifted_24x2 = AE_CONVERT_INT32_24x2(shifted);

    // Multiply/accumulate sum and multiplier:
    // (Q23.0 / 2^8) * (Q23.0 / 2^8) = Q47.0 / 2^16
    ae_q56s sum_56 = AE_ZEROQ56();
    AE_MULAS56P24S_HH(sum_56, x_24x2, shifted_24x2);

    // Shift left into 24bit space:
    // ((Q47.0 / 2^16) << 24) = Q23.24 / 2^16
    sum_56 = AE_Q56S_SLAI(sum_56, 24);

    // Truncate and place on the PR register:
    // (Q23.24 / 2^16) -> Q23.0 / 2^16
    x_24x2 = AE_TRUNCP24Q48(sum_56);
  }

  // Load the quantized multiplier into the PR register.
  // NOTE: This method assumes that this param has been calculated for 24bit
  // space - not 32bits.
  // Q0.31 -> Q0.23
  ae_p24x2s quantized_multiplier_24x2 =
      AE_CONVERT_INT32_24x2(quantized_multiplier);

  // Adjust for the additional 8 bits of lost precision throughout this
  // function:
  int shift_amount = 23;
  if (x_exceeds_24bits) {
    shift_amount = shift_amount - 8;
  }
  if (shift_exceeds_24bits) {
    shift_amount = shift_amount - 8;
  }

  // Find the product of x and the quantized_multiplier and right shift
  // to 48bit aligned.
  // (Q23.0 / 2^16) * Q23.0 = Q47.0 / 2^16
  // (Q47.0 / 2^16) >> 7 = Q47.0
  ae_q56s result_56 =
      SaturatingMultiply(x_24x2, quantized_multiplier_24x2, shift_amount);

  if (shift < 0) {
    // Handle any negative shift directly on the 48 bit value.
    result_56 = AE_Q56S_SRA(result_56, -shift);
  }
  return result_56;
}

//
// Multiply 24bit value by a quantized multiplier (w/ shift) and returns a 48bit
// aligned value in the QR register.
//
inline ae_q56s MultiplyByQuantizedMultiplier(ae_p24x2s x_24x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  // NOTE: x_24x2 = Q23.0

  // This is an optimized version of a 32 bit MultiplyByQuantizedMultiplier
  // operation of TFLite. Sometimes, the shifted value of |x_24x2| can exceed
  // the limits of INT24, which requires |AE_CONVERT_INT32_24x2()| to load the
  // left-most 24 bits of a 32bit integer. When this occurs, all Q values here
  // carry an additional division of 2^8 to account for this loss in precision.
  // This division will be applied to the final shift of the result in
  // |SaturatingMultiply()|.
  //
  // The Q-notation comments in this method describe the calculations that take
  // place when both |x| and the shifted value of |1| overflow the INT24 limits.
  bool shift_exceeds_24bits = false;

  ae_p24x2s x_shifted_24x2 = x_24x2;
  if (shift > 0) {
    int shifted = 1 << shift;
    if (shifted <= INT24_MIN || shifted >= INT24_MAX) {
      shift_exceeds_24bits = true;
    }
    // Load the shifted value into the PR register:
    // Q31.0 -> Q23.0 / 2^8
    ae_p24x2s shifted_24x2 = AE_CONVERT_INT32_24x2(shifted);

    // Multiply/accumulate sum and multiplier:
    ae_q56s sum_56 = AE_ZEROQ56();
    // Multiply/accumulate sum and multiplier:
    // Q23.0 * (Q23.0 / 2^8) = Q47.0 / 2^8
    AE_MULAS56P24S_HH(sum_56, x_24x2, shifted_24x2);

    // Shift left into 24bit space:
    // ((Q47.0 / 2^8) << 24) = Q23.24 / 2^8
    sum_56 = AE_Q56S_SLAI(sum_56, 24);

    // Truncate and place on the PR register:
    // (Q23.24 / 2^8) -> Q23.0 / 2^8
    x_shifted_24x2 = AE_ROUNDSP24Q48SYM(sum_56);
  }

  // Load the quantized multiplier into the PR register.
  // NOTE: This method assumes that this param has been calculated for 24bit
  // space - not 32bits.
  // Q0.31 -> Q0.23
  ae_p24x2s quantized_multiplier_24x2 =
      AE_CONVERT_INT32_24x2(quantized_multiplier);

  // Find the product of x and the quantized_multiplier and right shift
  // to 48bit aligned.
  // NOTE: Adjust for the additional 8 bits of lost precision throughout this
  // function:
  // (Q23.0 / 2^8) * Q23.0 = Q47.0 / 2^8
  // (Q47.0 / 2^8) >> 7 = Q47.0
  ae_q56s result = SaturatingMultiply(x_shifted_24x2, quantized_multiplier_24x2,
                                      shift_exceeds_24bits ? 15 : 23);

  if (shift < 0) {
    // Handle any negative shift directly on the 48 bit value.
    result = AE_Q56S_SRA(result, -shift);
  }
  return result;
}

//
// Calculate quantization params for 24bit runtimes.
//
inline void QuantizeMultiplier(double double_multiplier,
                               int32_t* quantized_multiplier, int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  // Special cased to 24bit:
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1 << 23)));

  TFLITE_CHECK(q_fixed <= (1 << 23));
  if (q_fixed == (1 << 23)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_CHECK_LE(q_fixed, INT24_MAX);

  if (*shift < -23) {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

//
// Convert a floating point number to a Q representation for 24 bit integers.
//
inline int CreateQConstantForInt24(int integer_bits, float f) {
  const double min_bounds = static_cast<double>(INT24_MIN);
  const double max_bounds = static_cast<double>(INT24_MAX);

  int fractional_bits = 23 - integer_bits;
  double raw = std::round(f * static_cast<double>(1 << fractional_bits));
  raw = std::max(raw, min_bounds);
  raw = std::min(raw, max_bounds);
  return static_cast<int>(raw);
}

}  // namespace hifimini
}  // namespace xtensa
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_FIXEDPOINT_UTILS_H_
