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

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

#if defined(HIFIMINI)

// INT24 MIN/MAX
#define INT24_MIN -8388608
#define INT24_MAX 8388607

// Multiply 24bit value by a quantized multiplier (w/ shift) and returns a 48bit
// aligned value in the QR register.
inline ae_q56s MultiplyByQuantizedMultiplier(ae_p24x2s x_24x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  // A value with 1 sign bit, N integer bits and M fractional bits is
  // represented as QN+1.M since the sign bit is included in the integer bits.
  //
  // The Q notation in this method explains the values represented in each
  // variable, along with an implicit division since the quantized_multiplier
  // represents a value between 0.5 and 1.0 (Q1.X-1 where X is the bit precision
  // of the type).
  //
  // Load the quantized multiplier into the PR register.
  // NOTE: This method assumes that this param has been calculated for 24bit
  // space - not 32bits.
  // Q32.0 / 2^23 -> Q24.0 / 2^23 representing a Q1.23 multiplier.
  ae_p24x2s quantized_multiplier_24x2 = AE_MOVPA24(quantized_multiplier);
  // Shift right by 23 - 16 bits minus the specified shift.  This is because we
  // keep 16 fractional bits until the end to perform rounding.  Subtract shift
  // since shift is a left shift, and the 23-16 is a right shift.
  int shift_amount = 7 - shift;

  // Find the product of x and the quantized_multiplier.
  // Q24.0 / 2^23 * Q24.0 = Q48.0 / 2^23
  // Q48.0 / 2^23 >> 7 = Q48.0 / 2^16
  ae_q56s result_56 = AE_MULP24S_HH(x_24x2, quantized_multiplier_24x2);

  // Shift right if shift amount is positive, left if shift amount is negative.
  if (shift_amount >= 0) {
    result_56 = AE_Q56S_SRA(result_56, shift_amount);
  } else {
    result_56 = AE_Q56S_SLA(result_56, -shift_amount);
  }

  // Round off the bottom 16 bits.
  // Q48.0 / 2^16 -> Q32.0 aligned to 48 bits.
  result_56 = AE_ROUNDSQ32SYM(result_56);
  return result_56;
}

// Multiply 32bit value by a quantized multiplier (w/ shift) and returns a 48bit
// aligned value in the QR register.
inline ae_q56s MultiplyByQuantizedMultiplierResult48Bit(
    int32_t x, int32_t quantized_multiplier, int shift) {
  // Convert x into a 2x24bit PR register file. If x is outside the numerical
  // limits of a 24bit integer, the "fractional" or lower 8bits are discarded.
  // If x is within the range of a 24 bit integer, the "signed" or upper 8bits
  // are discarded.
  ae_p24x2s x_24x2;
  if (x > INT24_MIN && x < INT24_MAX) {
    x_24x2 = AE_MOVPA24(x);
  } else {
    x_24x2 = static_cast<ae_p24s>(*reinterpret_cast<ae_p24f*>(&x));
    shift += 8;
  }

  return MultiplyByQuantizedMultiplier(x_24x2, quantized_multiplier, shift);
}

// Calculate quantization params for 24bit runtimes.
inline void QuantizeMultiplierForInt24(float multiplier,
                                       int32_t* quantized_multiplier,
                                       int* shift) {
  if (multiplier == 0.0f) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  // Special cased to 24bit:
  const float q = std::frexp(multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1 << 23)));

  TFLITE_CHECK(q_fixed <= (1 << 23));
  if (q_fixed == (1 << 23)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_CHECK_LE(q_fixed, INT24_MAX);

  // Ensure shift does not exceed 24-bit range.
  TFLITE_CHECK_LE(*shift, 23);
  if (*shift < -23) {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

// Convert a floating point number to a Q representation for 24 bit integers.
inline int CreateQConstantForInt24(int integer_bits, float f) {
  const float min_bounds = static_cast<float>(INT24_MIN);
  const float max_bounds = static_cast<float>(INT24_MAX);

  int fractional_bits = 23 - integer_bits;
  float raw = std::round(f * static_cast<float>(1 << fractional_bits));
  raw = std::max(raw, min_bounds);
  raw = std::min(raw, max_bounds);
  return static_cast<int>(raw);
}

#endif  // defined(HIFIMINI)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_FIXEDPOINT_UTILS_H_
