/* Copyright 2018 The OpenXLA Authors.

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
#ifndef XLA_FP_UTIL_H_
#define XLA_FP_UTIL_H_

#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>

#include "xla/types.h"
#include "xla/util.h"

namespace xla {

// Returns true if the value is zero as per the IEEE-754 isZero operation.
template <typename T>
constexpr bool IsZero(T x) {
  return x == static_cast<T>(0.0f);
}

// Returns true if the value is negative as per the IEEE-754 isSignMinus
// operation.
template <typename T>
constexpr bool IsSignMinus(T x) {
  return x < 0;
}

// Returns the absolute value of the value.
template <typename T>
constexpr T Abs(T x) {
  if (IsZero(x)) {
    // Positive zero plus negative zero is positive zero.
    // Positive zero plus positive zero is positive zero.
    return x + static_cast<T>(0.0f);
  }
  return IsSignMinus(x) ? -x : x;
}

// Returns true if the value is NaN as per the IEEE-754 isNaN operation.
template <typename T>
constexpr bool IsNaN(T x) {
  return x != x;
}

// Returns true if the value is infinite as per the IEEE-754 isInfinite
// operation.
template <typename T>
constexpr bool IsInfinite(T x) {
  return x == std::numeric_limits<T>::infinity() ||
         x == -std::numeric_limits<T>::infinity();
}

// Returns true if the value is finite as per the IEEE-754 isFinite operation.
template <typename T>
constexpr bool IsFinite(T x) {
  return !IsNaN(x) && !IsInfinite(x);
}

// Returns true if the value is normal as per the IEEE-754 isNormal operation.
template <typename T>
constexpr bool IsNormal(T x) {
  T abs_x = Abs(x);
  return abs_x >= std::numeric_limits<T>::min() &&
         abs_x <= std::numeric_limits<T>::max();
}

// Returns true if the value is subnormal as per the IEEE-754 isSubnormal
// operation.
template <typename T>
constexpr bool IsSubnormal(T x) {
  T abs_x = Abs(x);
  return abs_x > static_cast<T>(0) && abs_x < std::numeric_limits<T>::min();
}

// Scales a value by a power of the radix as per the IEEE-754 scaleB operation.
template <typename T>
constexpr T ScaleBase(T x, int n) {
  static_assert(is_specialized_floating_point_v<T>);
  // While n is positive, move the radix point right. This is the same as
  // multiplying by the radix. Rounding will not occur because the next radix
  // interval has at least as much precision as the last.
  while (n > 0 && IsFinite(x) && !IsZero(x)) {
    int multiplier_exponent =
        std::min(n, std::numeric_limits<T>::max_exponent - 1);
    x *= IPow(static_cast<T>(std::numeric_limits<T>::radix),
              multiplier_exponent);
    n -= multiplier_exponent;
  }
  // While n is negative, move the radix point left. For normal numbers, this
  // is the same as dividing by the radix. For subnormal numbers, we need to
  // divide by a scaled form of the radix so that we will not induce rounding.
  for (; n < 0 && IsFinite(x) && !IsZero(x); ++n) {
    T shifted_x = x / std::numeric_limits<T>::radix;
    // This shift would make the number subnormal which means our result is
    // either a subnormal or 0. We can compute the answer by just scaling the
    // smallest subnormal and multiplying by that.
    if (IsSubnormal(shifted_x)) {
      int scale_exponent = -((std::numeric_limits<T>::min_exponent - 1) -
                             (std::numeric_limits<T>::digits - 1)) +
                           n;
      // denorm_min is the smallest subnormal number so multiplying it by 2^m
      // where m < 0 is just zero.
      if (scale_exponent < 0) {
        return x * static_cast<T>(0);
      }
      return x *
             ScaleBase(std::numeric_limits<T>::denorm_min(), scale_exponent);
    }
    x = shifted_x;
  }
  return x;
}

// Returns the exponent of the given value as per the IEEE-754 logB operation.
template <typename T>
constexpr std::optional<int> LogBase(T x) {
  if (IsNaN(x)) {
    return std::nullopt;
  }
  if (IsInfinite(x)) {
    return std::numeric_limits<int>::max();
  }
  if (IsZero(x)) {
    return std::numeric_limits<int>::min();
  }
  T abs_x = Abs(x);
  int exponent = 0;
  while (abs_x < static_cast<T>(1)) {
    abs_x *= std::numeric_limits<T>::radix;
    exponent -= 1;
  }
  while (abs_x >= std::numeric_limits<T>::radix) {
    abs_x /= std::numeric_limits<T>::radix;
    exponent += 1;
  }
  return exponent;
}

enum class RoundingDirection {
  kRoundTiesToEven,
  kRoundTowardsZero,
};

// Splits a double in two floats, high and low such that high + low approximates
// the double very closely. The high float will have `kNumHighFloatZeroLsbs`
// clear. Returns {high, low}.
// This lets us turn a double with 53 bits of precision into a result with
// `49 - kNumHighFloatZeroLsbs` bits of precision.
// N.B. The number 49 comes from 2*24 + 1. The extra bit of precision comes from
// the sign bit of the low component (e.g. 0x1.ffffffffffffp+0 which has 49 bits
// of precision can be represented via 0x1p+1 - 0x1p-48.)
template <typename DstT, typename SrcT>
constexpr std::pair<DstT, DstT> SplitToFpPair(
    SrcT to_split, int num_high_trailing_zeros,
    RoundingDirection rounding_direction =
        RoundingDirection::kRoundTiesToEven) {
  constexpr auto kError =
      std::make_pair(std::numeric_limits<DstT>::quiet_NaN(),
                     std::numeric_limits<DstT>::quiet_NaN());
  if (num_high_trailing_zeros < 0) {
    return kError;
  }
  if (!IsFinite(to_split)) {
    return kError;
  }
  if (IsZero(to_split)) {
    DstT zero = static_cast<DstT>(to_split);
    return std::make_pair(zero, zero);
  }
  if (IsSignMinus(to_split)) {
    auto [high, low] =
        SplitToFpPair<DstT, SrcT>(Abs(to_split), num_high_trailing_zeros);
    return std::make_pair(-high, -low);
  }
  // First, let's round our double to fewer bits of precision.
  auto maybe_exponent = LogBase(to_split);
  if (!maybe_exponent.has_value()) {
    return kError;
  }
  int exponent = *maybe_exponent;
  constexpr int kMinNormalExponent =
      std::numeric_limits<DstT>::min_exponent - 1;
  const int effective_precision = std::numeric_limits<DstT>::digits -
                                  std::max(kMinNormalExponent - exponent, 0);
  const int high_bits_to_keep = effective_precision - num_high_trailing_zeros;
  if (high_bits_to_keep < 1) {
    return kError;
  }
  // Rescale the input value to a fixed point representation such that the bits
  // that we want to round-off are always in the fractional part.
  static_assert(std::numeric_limits<SrcT>::max_exponent - 1 >=
                std::numeric_limits<DstT>::digits);
  SrcT scaled_significand =
      ScaleBase(to_split, high_bits_to_keep - (exponent + 1));
  // `integer_part` is the value of the significand with the bits we want to
  // keep in the high float.
  uint64_t integer_part = static_cast<uint64_t>(scaled_significand);
  // `fractional_part` is the value of the significand with the bits we want to
  // round-off in the high float.
  SrcT fractional_part = scaled_significand - static_cast<SrcT>(integer_part);
  switch (rounding_direction) {
    case RoundingDirection::kRoundTiesToEven: {
      // Perform RTNE: if the fractional part is greater than 0.5 or if the
      // fractional part is 0.5 and the integer part is odd, we need to round
      // up.
      if (fractional_part > static_cast<SrcT>(0.5f) ||
          (fractional_part == static_cast<SrcT>(0.5f) &&
           integer_part % 2 == 1)) {
        integer_part += 1;
      }
      break;
    }
    case RoundingDirection::kRoundTowardsZero: {
      // Perform RTZ: do nothing.
      break;
    }
  }
  // Rescale the integer part to the original exponent.
  SrcT rounded = ScaleBase(static_cast<SrcT>(integer_part),
                           (exponent + 1) - high_bits_to_keep);
  // Now, we will turn our double into a float. This is merely a format change,
  // no rounding should occur.
  DstT high = static_cast<DstT>(rounded);
  // This conversion should not result in any bits changing in kHigh.
  if (static_cast<SrcT>(high) != rounded) {
    return kError;
  }
  DstT low = static_cast<DstT>(to_split - double{high});
  return std::make_pair(high, low);
}

// Rounds a floating point number to less precision.
template <typename DstT, typename SrcT>
constexpr DstT RoundToPrecision(
    SrcT to_round, int precision = std::numeric_limits<DstT>::digits,
    RoundingDirection rounding_direction =
        RoundingDirection::kRoundTiesToEven) {
  auto [high, low] = SplitToFpPair<DstT, SrcT>(
      to_round,
      /*num_high_trailing_zeros=*/std::numeric_limits<DstT>::digits - precision,
      rounding_direction);
  return high;
}

// Use splitting to find high + low == log(2) where high has the bottom
// `kBitsToDrop` clear. Returns {high, low}.
template <typename DstT>
constexpr std::pair<DstT, DstT> Log2FloatPair(int num_high_trailing_zeros) {
  return SplitToFpPair<DstT>(M_LN2, num_high_trailing_zeros);
}

// There are many different definitions of ulp(x) in the literature. Here, we
// are using the "GoldbergUlp" definition as found in: Jean-Michel Muller. On
// the definition of ulp(x). [Research Report] RR-5504, LIP RR-2005-09, INRIA,
// LIP. 2005, pp.16. ⟨inria-00070503⟩
template <typename T>
constexpr T GoldbergUlp(T x) {
  if (IsZero(x) || IsSubnormal(x)) {
    return GoldbergUlp(std::numeric_limits<T>::min());
  }
  std::optional<int> maybe_exponent = LogBase(x);
  if (maybe_exponent.has_value(); const int exponent = *maybe_exponent) {
    return ScaleBase(std::numeric_limits<T>::epsilon(), exponent);
  }
  if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
    return std::numeric_limits<T>::quiet_NaN();
  } else if constexpr (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else {
    return GoldbergUlp(std::numeric_limits<T>::max());
  }
}

// Returns the number of FP values between two floating point values. Please
// note that +/-0 are considered equivalent.
template <typename T>
int64_t CalculateDistanceInFloats(T a, T b) {
  auto a_sign_and_magnitude = SignAndMagnitude(a);
  auto b_sign_and_magnitude = SignAndMagnitude(b);
  uint64_t a_distance_from_zero = a_sign_and_magnitude.first
                                      ? -a_sign_and_magnitude.second
                                      : a_sign_and_magnitude.second;
  uint64_t b_distance_from_zero = b_sign_and_magnitude.first
                                      ? -b_sign_and_magnitude.second
                                      : b_sign_and_magnitude.second;
  // Bitcast into signed type after doing subtraction in unsigned to allow for
  // integer overflow.
  int64_t signed_distance = a_distance_from_zero - b_distance_from_zero;
  return std::abs(signed_distance);
}

}  // namespace xla

#endif  // XLA_FP_UTIL_H_
