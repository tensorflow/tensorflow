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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMMON_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifndef ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#ifdef GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#define ALLOW_SLOW_GENERIC_DEPTHWISECONV_FALLBACK
#endif
#endif

#include <cmath>
#include <functional>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/optimized/neon_check.h"

// LINT.IfChange

namespace tflite_migration {

constexpr int kReverseShift = -1;

TFLITE_NOINLINE int32_t MultiplyByQuantizedMultiplier(
    int32_t x, int32_t quantized_multiplier, int shift);

TFLITE_NOINLINE int32_t MultiplyByQuantizedMultiplier(
    int64_t x, int32_t quantized_multiplier, int shift);

// Single-rounding MultiplyByQuantizedMultiplier
#if TFLITE_SINGLE_ROUNDING
inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32_t x, int32_t quantized_multiplier, int shift) {
  TFLITE_DCHECK_LE(shift, 0);
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(
    int32_t x, int32_t quantized_multiplier, int shift) {
  TFLITE_DCHECK_GE(shift, 0);
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

#ifdef USE_NEON
inline int32x4x4_t MultiplyByQuantizedMultiplier4Rows(
    int32x4x4_t input_val, int32_t quantized_multiplier, int shift) {
  TFLITE_DCHECK(quantized_multiplier >= 0);

  const int right_shift = std::min(-1, shift);
  const int left_shift = shift - right_shift;

  const int32x4_t multiplier_dup = vdupq_n_s32(quantized_multiplier);
  const int32x4_t left_shift_dup = vdupq_n_s32(left_shift);
  const int32x4_t right_shift_dup = vdupq_n_s32(right_shift);

  int32x4x4_t result;
  result.val[0] = vrshlq_s32(
      vqdmulhq_s32(vshlq_s32(input_val.val[0], left_shift_dup), multiplier_dup),
      right_shift_dup);

  result.val[1] = vrshlq_s32(
      vqdmulhq_s32(vshlq_s32(input_val.val[1], left_shift_dup), multiplier_dup),
      right_shift_dup);

  result.val[2] = vrshlq_s32(
      vqdmulhq_s32(vshlq_s32(input_val.val[2], left_shift_dup), multiplier_dup),
      right_shift_dup);

  result.val[3] = vrshlq_s32(
      vqdmulhq_s32(vshlq_s32(input_val.val[3], left_shift_dup), multiplier_dup),
      right_shift_dup);

  return result;
}
#endif  // USE_NEON
// Double-rounding MultiplyByQuantizedMultiplier
#else
inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                           quantized_multiplier);
}

#ifdef USE_NEON
// Round uses ARM's rounding shift right.
inline int32x4x4_t MultiplyByQuantizedMultiplier4Rows(
    int32x4x4_t input_val, int32_t quantized_multiplier, int shift) {
  const int left_shift = std::max(shift, 0);
  const int right_shift = std::min(shift, 0);
  int32x4x4_t result;

  int32x4_t multiplier_dup = vdupq_n_s32(quantized_multiplier);
  int32x4_t left_shift_dup = vdupq_n_s32(left_shift);
  int32x4_t right_shift_dup = vdupq_n_s32(right_shift);

  result.val[0] =
      vrshlq_s32(vqrdmulhq_s32(vshlq_s32(input_val.val[0], left_shift_dup),
                               multiplier_dup),
                 right_shift_dup);

  result.val[1] =
      vrshlq_s32(vqrdmulhq_s32(vshlq_s32(input_val.val[1], left_shift_dup),
                               multiplier_dup),
                 right_shift_dup);

  result.val[2] =
      vrshlq_s32(vqrdmulhq_s32(vshlq_s32(input_val.val[2], left_shift_dup),
                               multiplier_dup),
                 right_shift_dup);

  result.val[3] =
      vrshlq_s32(vqrdmulhq_s32(vshlq_s32(input_val.val[3], left_shift_dup),
                               multiplier_dup),
                 right_shift_dup);

  return result;
}
#endif  // USE_NEON
#endif  // TFLITE_SINGLE_ROUNDING

template <typename T>
int CountLeadingZeros(T integer_input) {
  static_assert(std::is_unsigned<T>::value,
                "Only unsigned integer types handled.");
  if (integer_input == 0) {
    return std::numeric_limits<T>::digits;
  }
#if defined(__GNUC__)
  if (std::is_same<T, uint32_t>::value) {
    return __builtin_clz(integer_input);
  } else if (std::is_same<T, uint64_t>::value) {
    return __builtin_clzll(integer_input);
  }
#endif
  const T one_in_leading_positive = static_cast<T>(1)
                                    << (std::numeric_limits<T>::digits - 1);
  int leading_zeros = 0;
  while (integer_input < one_in_leading_positive) {
    integer_input <<= 1;
    ++leading_zeros;
  }
  return leading_zeros;
}

inline void GetInvSqrtQuantizedMultiplierExp(int32_t input, int reverse_shift,
                                             int32_t* output_inv_sqrt,
                                             int* output_shift) {
  TFLITE_DCHECK_GE(input, 0);
  if (input <= 1) {
    // Handle the input value 1 separately to avoid overflow in that case
    // in the general computation below (b/143972021). Also handle 0 as if it
    // were a 1. 0 is an invalid input here (divide by zero) and 1 is a valid
    // but rare/unrealistic input value. We can expect both to occur in some
    // incompletely trained models, but probably not in fully trained models.
    *output_inv_sqrt = std::numeric_limits<std::int32_t>::max();
    *output_shift = 0;
    return;
  }
  TFLITE_DCHECK_GT(input, 1);
  *output_shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*output_shift;
  }
  const unsigned max_left_shift_bits =
      CountLeadingZeros(static_cast<uint32_t>(input)) - 1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  TFLITE_DCHECK_GE(input, (1 << 27));
  TFLITE_DCHECK_LT(input, (1 << 29));
  using gemmlowp::FixedPoint;
  using gemmlowp::Rescale;
  using gemmlowp::SaturatingRoundingMultiplyByPOT;
  // Using 3 integer bits gives us enough room for the internal arithmetic in
  // this Newton-Raphson iteration.
  using F3 = FixedPoint<int32_t, 3>;
  using F0 = FixedPoint<int32_t, 0>;
  const F3 fixedpoint_input = F3::FromRaw(input >> 1);
  const F3 fixedpoint_half_input =
      SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
  const F3 fixedpoint_half_three =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
  // Newton-Raphson iteration
  // Naive unoptimized starting guess: x = 1
  F3 x = F3::One();
  // Naive unoptimized number of iterations: 5
  for (int i = 0; i < 5; i++) {
    const F3 x3 = Rescale<3>(x * x * x);
    x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
  }
  const F0 fixedpoint_half_sqrt_2 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
  x = x * fixedpoint_half_sqrt_2;
  *output_inv_sqrt = x.raw();
  if (*output_shift < 0) {
    *output_inv_sqrt <<= -*output_shift;
    *output_shift = 0;
  }
  // Convert right shift (right is positive) to left shift.
  *output_shift *= reverse_shift;
}

}  // namespace tflite_migration

// LINT.ThenChange(//tensorflow/lite/kernels/internal/common.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_COMMON_H_
