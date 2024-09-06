/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

#include <cmath>
#include <cstdint>
#include <limits>

namespace tflite_migration {

// LINT.IfChange

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);

// Splits a double input value into a returned fraction, and a shift value from
// the exponent, using only bitwise and integer operations to support
// microcontrollers and other environments without floating-point support.
//
// This is designed to be a replacement for how std::frexp() is used within the
// QuantizeMultiplier() function, and so has a different signature than the
// standard version, returning a 64-bit integer rather than a double. This
// result has a maximum value of 1<<31, with the fraction expressed as a
// proportion of that maximum.
//
// std::frexp() returns NaNs and infinities unmodified, but since we're
// returning integers that can't represent those values, instead we return
// a shift of std::numeric_limits<int>::max() for all bad numbers, with an int64
// result of 0 for NaNs, std:numeric_limits<int64_t>::max() for +INFINITY, and
// std::numeric_limits<int64_t>::min() for -INFINITY. Denormalized inputs will
// result in return values that end up truncating some bits at the end,
// reflecting the loss of precision inherent in denormalization.
int64_t IntegerFrExp(double input, int* shift);

// Converts an integer fraction in the format produced by IntegerFrExp (where
// 0x40000000 is 1.0) and an exponent shift (between -1022 and +1022) into an
// IEEE binary64 double format result. The implementation uses only integer and
// bitwise operators, so no floating point hardware support or emulation is
// needed. This is here so quantized operations can run non-time-critical
// preparation calculations on microcontrollers and other platforms without
// float support.
double DoubleFromFractionAndShift(int64_t fraction, int shift);

// Performs a multiplication of two numbers in double format, using only integer
// and bitwise instructions. This is aimed at supporting housekeeping functions
// for quantized operations on microcontrollers without floating-point hardware.
double IntegerDoubleMultiply(double a, double b);

// Returns -1 if a is less than b, 0 if a and b are equal, and +1 if a is
// greater than b. It is implemented using only integer and logical instructions
// so that it can be easily run on microcontrollers for quantized operations.
int IntegerDoubleCompare(double a, double b);

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);
// Like PreprocessSoftmaxScaling, but inverse scaling factors also calculated.

// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference in
// Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits = 31);

// Converts a floating-point number to an integer. For all inputs x where
// static_cast<IntOut>(x) is legal according to the C++ standard, the result
// is identical to that cast (i.e. the result is x with its fractional part
// truncated whenever that is representable as IntOut).
//
// static_cast would cause undefined behavior for the following cases, which
// have well-defined behavior for this function:
//
//  1. If x is NaN, the result is zero.
//
//  2. If the truncated form of x is above the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::max().
//
//  3. If the truncated form of x is below the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::min().
//
// Note that cases #2 and #3 cover infinities as well as finite numbers.
//
// The range of FloatIn must include the range of IntOut, otherwise
// the results are undefined.
// TODO(sfeuz): Replace by absl::SafeCast once available.
template <class IntOut, class FloatIn>
IntOut SafeCast(FloatIn x) {
  static_assert(!std::numeric_limits<FloatIn>::is_integer,
                "FloatIn is integer");
  static_assert(std::numeric_limits<IntOut>::is_integer,
                "IntOut is not integer");
  static_assert(std::numeric_limits<IntOut>::radix == 2, "IntOut is base 2");

  // Special case NaN, for which the logic below doesn't work.
  if (std::isnan(x)) {
    return 0;
  }

  // Negative values all clip to zero for unsigned results.
  if (!std::numeric_limits<IntOut>::is_signed && x < 0) {
    return 0;
  }

  // Handle infinities.
  if (std::isinf(x)) {
    return x < 0 ? std::numeric_limits<IntOut>::min()
                 : std::numeric_limits<IntOut>::max();
  }

  // Set exp such that x == f * 2^exp for some f with |f| in [0.5, 1.0),
  // unless x is zero in which case exp == 0. Note that this implies that the
  // magnitude of x is strictly less than 2^exp.
  int exp = 0;
  std::frexp(x, &exp);

  // Let N be the number of non-sign bits in the representation of IntOut. If
  // the magnitude of x is strictly less than 2^N, the truncated version of x
  // is representable as IntOut. The only representable integer for which this
  // is not the case is kMin for signed types (i.e. -2^N), but that is covered
  // by the fall-through below.
  if (exp <= std::numeric_limits<IntOut>::digits) {
    return x;
  }

  // Handle numbers with magnitude >= 2^N.
  return x < 0 ? std::numeric_limits<IntOut>::min()
               : std::numeric_limits<IntOut>::max();
}
// LINT.ThenChange(//tensorflow/lite/kernels/internal/quantization_util.h)
}  // namespace tflite_migration

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
