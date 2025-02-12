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

#include "tensorflow/compiler/mlir/lite/kernels/internal/quantization_util.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/cppmath.h"

namespace tflite_migration {

// LINT.IfChange

namespace {
// These constants are used to manipulate the binary representation of doubles.
// Double-precision binary64 floating point format is:
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
constexpr uint64_t kSignMask = 0x8000000000000000LL;
constexpr uint64_t kExponentMask = 0x7ff0000000000000LL;
constexpr int32_t kExponentShift = 52;
constexpr int32_t kExponentBias = 1023;
constexpr uint32_t kExponentIsBadNum = 0x7ff;
constexpr uint64_t kFractionMask = 0x000fffffffc00000LL;
constexpr uint32_t kFractionShift = 22;
constexpr uint32_t kFractionRoundingMask = 0x003fffff;
constexpr uint32_t kFractionRoundingThreshold = 0x00200000;
}  // namespace

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
#if TFLITE_SINGLE_ROUNDING
  // Single-rounding MultiplyByQuantizedMultiplier only supports positive
  // multipliers.
  // TFLITE_DCHECK(double_multiplier >= 0);
#endif
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
#ifdef TFLITE_EMULATE_FLOAT
  // If we're trying to avoid the use of floating-point instructions (for
  // example on microcontrollers) then use an alternative implementation
  // that only requires integer and bitwise operations. To enable this, you
  // need to set the define during the build process for your platform.
  int64_t q_fixed = IntegerFrExp(double_multiplier, shift);
#else   // TFLITE_EMULATE_FLOAT
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1LL << 31)));
#endif  // TFLITE_EMULATE_FLOAT
  TFLITE_DCHECK(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_DCHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
#if TFLITE_SINGLE_ROUNDING
  // Single-rounding MultiplyByQuantizedMultiplier doesn't support a shift > 30,
  // saturate it.
  if (*shift > 30) {
    *shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
#endif
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
  TFLITE_DCHECK_GT(double_multiplier, 1.);
  QuantizeMultiplier(double_multiplier, quantized_multiplier, left_shift);
  TFLITE_DCHECK_GE(*left_shift, 0);
}

int64_t IntegerFrExp(double input, int* shift) {
  // Make sure our assumptions about the double layout hold.
  TFLITE_DCHECK_EQ(8, sizeof(double));

  // We want to access the bits of the input double value directly, which is
  // tricky to do safely, so use a union to handle the casting.
  union {
    double double_value;
    uint64_t double_as_uint;
  } cast_union;
  cast_union.double_value = input;
  const uint64_t u = cast_union.double_as_uint;

  // If the bitfield is all zeros apart from the sign bit, this is a normalized
  // zero value, so return standard values for this special case.
  if ((u & ~kSignMask) == 0) {
    *shift = 0;
    return 0;
  }

  // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
  // the exponent, and distinguished by whether the fractions are zero or
  // non-zero.
  const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
  if (exponent_part == kExponentIsBadNum) {
    *shift = std::numeric_limits<int>::max();
    if (u & kFractionMask) {
      // NaN, so just return zero (with the exponent set to INT_MAX).
      return 0;
    } else {
      // Infinity, so return +/- INT_MAX.
      if (u & kSignMask) {
        return std::numeric_limits<int64_t>::min();
      } else {
        return std::numeric_limits<int64_t>::max();
      }
    }
  }

  // The shift is fairly easy to extract from the high bits of the double value,
  // just by masking it out and applying a bias. The std::frexp() implementation
  // always returns values between 0.5 and 1.0 though, whereas the exponent
  // assumes 1.0 to 2.0 is the standard range, so I add on one to match that
  // interface.
  *shift = (exponent_part - kExponentBias) + 1;

  // There's an implicit high bit in the double format definition, so make sure
  // we include that at the top, and then reconstruct the rest of the fractional
  // value from the remaining fragments.
  int64_t fraction = 0x40000000 + ((u & kFractionMask) >> kFractionShift);

  // We're cutting off some bits at the bottom, so to exactly match the standard
  // frexp implementation here we'll apply rounding by adding one to the least
  // significant bit of the result if the discarded portion is over half of the
  // maximum.
  if ((u & kFractionRoundingMask) > kFractionRoundingThreshold) {
    fraction += 1;
  }
  // Negate the fraction if the sign bit was set.
  if (u & kSignMask) {
    fraction *= -1;
  }

  return fraction;
}

double DoubleFromFractionAndShift(int64_t fraction, int shift) {
  union {
    double double_value;
    uint64_t double_as_uint;
  } result;

  // Detect NaNs and infinities.
  if (shift == std::numeric_limits<int>::max()) {
    if (fraction == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    } else if (fraction > 0) {
      return std::numeric_limits<double>::infinity();
    } else {
      return -std::numeric_limits<double>::infinity();
    }
  }

  // Return a normalized zero for a zero fraction.
  if (fraction == 0) {
    result.double_as_uint = 0;
    return result.double_value;
  }

  bool is_negative = (fraction < 0);
  int64_t encoded_fraction = is_negative ? -fraction : fraction;
  int64_t encoded_shift = (shift - 1);
  while (encoded_fraction < 0x40000000) {
    encoded_fraction *= 2;
    encoded_shift -= 1;
  }
  while (encoded_fraction > 0x80000000) {
    encoded_fraction /= 2;
    encoded_shift += 1;
  }
  encoded_fraction -= 0x40000000;
  if (encoded_shift < -1022) {
    encoded_shift = -1023;
  } else if (encoded_shift > 1022) {
    encoded_shift = 1023;
  }
  encoded_shift += kExponentBias;
  uint64_t encoded_sign = is_negative ? kSignMask : 0;
  result.double_as_uint = encoded_sign | (encoded_shift << kExponentShift) |
                          (encoded_fraction << kFractionShift);
  return result.double_value;
}

double IntegerDoubleMultiply(double a, double b) {
  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);
  // Detect NaNs and infinities.
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int result_shift = a_shift + b_shift + 1;
  const int64_t result_fraction = (a_fraction * b_fraction) >> 32;
  return DoubleFromFractionAndShift(result_fraction, result_shift);
}

int IntegerDoubleCompare(double a, double b) {
  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);

  // Detect NaNs and infinities.
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return 1;
  }

  if ((a_fraction == 0) && (b_fraction < 0)) {
    return 1;
  } else if ((a_fraction < 0) && (b_fraction == 0)) {
    return -1;
  } else if (a_shift < b_shift) {
    return -1;
  } else if (a_shift > b_shift) {
    return 1;
  } else if (a_fraction < b_fraction) {
    return -1;
  } else if (a_fraction > b_fraction) {
    return 1;
  } else {
    return 0;
  }
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
#if TFLITE_SINGLE_ROUNDING
  const double max_real_multiplier = (1LL << 30) - 1.0;
#else
  const double max_real_multiplier = (1LL << 31) - 1.0;
#endif

#ifdef TFLITE_EMULATE_FLOAT
  const double input_beta = IntegerDoubleMultiply(beta, input_scale);
  int shift;
  int64_t fraction = IntegerFrExp(input_beta, &shift);
  shift += (31 - input_integer_bits);
  double input_beta_real_multiplier =
      DoubleFromFractionAndShift(fraction, shift);
  if (IntegerDoubleCompare(input_beta_real_multiplier, max_real_multiplier) >
      0) {
    input_beta_real_multiplier = max_real_multiplier;
  }
#else   // TFLITE_EMULATE_FLOAT
  const double input_beta_real_multiplier =
      std::min<double>(beta * input_scale * (1 << (31 - input_integer_bits)),
                       max_real_multiplier);
#endif  // TFLITE_EMULATE_FLOAT

  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                   quantized_multiplier, left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits) {
#ifdef TFLITE_EMULATE_FLOAT
  int64_t result = (1 << input_integer_bits) - 1;
  result <<= (total_signed_bits - input_integer_bits);
  result >>= input_left_shift;
  return result;
#else   // TFLITE_EMULATE_FLOAT
  const double max_input_rescaled =
      1.0 * ((1 << input_integer_bits) - 1) *
      (1LL << (total_signed_bits - input_integer_bits)) /
      (1LL << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int>(std::floor(max_input_rescaled));
#endif  // TFLITE_EMULATE_FLOAT
}

// LINT.ThenChange(//tensorflow/lite/kernels/internal/quantization_util.cc)

}  // namespace tflite_migration
