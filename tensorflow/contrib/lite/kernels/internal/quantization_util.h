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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
template <typename T>
QuantizationParams ChooseQuantizationParams(double rmin, double rmax,
                                            bool narrow_range) {
  const T qmin = std::numeric_limits<T>::min() + (narrow_range ? 1 : 0);
  const T qmax = std::numeric_limits<T>::max();
  const double qmin_double = qmin;
  const double qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_CHECK_LE(rmin, 0.);
  TFLITE_CHECK_GE(rmax, 0.);
  if (rmin == rmax) {
    // Special case where the min,max range is a point. Should be {0}.
    TFLITE_CHECK_EQ(rmin, 0.);
    TFLITE_CHECK_EQ(rmax, 0.);
    QuantizationParams quantization_params;
    quantization_params.zero_point = 0;
    quantization_params.scale = 0.;
    return quantization_params;
  }

  // General case.
  //
  // First determine the scale.
  const double scale = (rmax - rmin) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);

  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(round(zero_point_double));
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  TFLITE_CHECK_GE(nudged_zero_point, qmin);
  TFLITE_CHECK_LE(nudged_zero_point, qmax);

  // Finally, store the result nudged quantization params.
  QuantizationParams quantization_params;
  quantization_params.zero_point = nudged_zero_point;
  quantization_params.scale = scale;
  return quantization_params;
}

template <typename T>
QuantizationParams ChooseQuantizationParams(double rmin, double rmax) {
  return ChooseQuantizationParams<T>(rmin, rmax, false);
}

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

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of NEGATIVE its exponent ---
// this is intended as a RIGHT-shift.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift);

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

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);
// Like PreprocessSoftmaxScaling, but inverse scaling factors also calculated.
void PreprocessLogSoftmaxScalingExp(double beta, double input_scale,
                                    int input_integer_bits,
                                    int32_t* quantized_multiplier,
                                    int* left_shift,
                                    int32_t* reverse_scaling_divisor,
                                    int* reverse_scaling_left_shift);
// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference in
// Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift);

// Nudges a min/max quantization range to ensure zero is zero.
// Gymnastics with nudged zero point is to ensure that real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
// Outputs nudged_min, nudged_max, nudged_scale.
void NudgeQuantizationRange(const float min, const float max,
                            const int quant_min, const int quant_max,
                            float* nudged_min, float* nudged_max,
                            float* nudged_scale);

// Fake quantizes (quantizes and dequantizes) input_data using the scale,
// nudged_min, and nudged_max from NudgeQuantizationRange. This matches the code
// in TensorFlow's FakeQuantizeWithMinMaxVarsFunctor.
void FakeQuantizeArray(const float nudged_scale, const float nudged_min,
                       const float nudged_max, const float* input_data,
                       float* output_data, const float size);

// If x is approximately a power of two (with any positive or negative
// exponent), stores that exponent (i.e. log2(x)) in *log2_result, otherwise
// returns false.
bool CheckedLog2(const float x, int* log2_result);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
