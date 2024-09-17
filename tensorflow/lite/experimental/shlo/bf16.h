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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_BF16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_BF16_H_

#if defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
namespace shlo_ref {
using BF16 = ::std::bfloat16_t;
}  // namespace shlo_ref

#else
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/base/casts.h"
#include "absl/log/absl_check.h"

// On arm64 the compiler is not yet able to generate code for __bf16
// operations. Therefore, we resort to a software-based implementation of BF16
// based on promoting ops to float.
namespace shlo_ref {
class BF16;

namespace internal {
BF16 NumericF32ToBF16RoundNearestEven(float v);
BF16 F32ToBF16RoundNearestEven(float v);
float BF16ToFloat(BF16 v);
}  // namespace internal
}  // namespace shlo_ref

namespace std {
template <>
struct numeric_limits< ::shlo_ref::BF16>;
}  // namespace std

namespace shlo_ref {

class BF16 {
 public:
  constexpr BF16() : value_(0) {}

  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, float> > >
  explicit BF16(T x) {
    if constexpr (std::is_same_v<T, bool>) {
      value_ = static_cast<uint16_t>(x) * 0x3f80;
    } else if constexpr (std::numeric_limits<T>::is_integer) {
      *this = internal::NumericF32ToBF16RoundNearestEven(static_cast<float>(x));
    } else {
      *this = internal::F32ToBF16RoundNearestEven(static_cast<float>(x));
    }
  }

  // Tagged constructor to allow construction from bits
  struct bitcast_construct_t {};
  explicit constexpr BF16(bitcast_construct_t, uint16_t value)
      : value_(value) {}

  // Lossless conversion to `float`.
  operator float() const {  // NOLINT: Allow implicit conversions to float.
    return internal::BF16ToFloat(*this);
  }

  // Assignment operators
  BF16& operator=(float v) { return *this = static_cast<BF16>(v); }
  BF16& operator=(bool v) { return *this = static_cast<BF16>(v); }

  template <typename T>
  // NOLINTNEXTLINE(misc-unconventional-assign-operator)
  std::enable_if_t<std::numeric_limits<T>::is_integer, BF16&> operator=(T v) {
    return *this = static_cast<BF16>(v);
  }

#define INTERNAL_BF16_ARITHMETIC_OP(OP)                          \
  friend BF16 operator OP(BF16 x, BF16 y) {                      \
    return BF16(static_cast<float>(x) OP static_cast<float>(y)); \
  }

#define INTERNAL_BF16_ARITHMETIC_ASSIGN_OP(OP)                       \
  friend BF16& operator OP##=(BF16 & x, BF16 y) {                    \
    return x = BF16(static_cast<float>(x) OP static_cast<float>(y)); \
  }

  INTERNAL_BF16_ARITHMETIC_OP(+)
  INTERNAL_BF16_ARITHMETIC_ASSIGN_OP(+)
  INTERNAL_BF16_ARITHMETIC_OP(-)
  INTERNAL_BF16_ARITHMETIC_ASSIGN_OP(-)
  INTERNAL_BF16_ARITHMETIC_OP(*)
  INTERNAL_BF16_ARITHMETIC_ASSIGN_OP(*)
  INTERNAL_BF16_ARITHMETIC_OP(/)
  INTERNAL_BF16_ARITHMETIC_ASSIGN_OP(/)
  INTERNAL_BF16_ARITHMETIC_OP(==)
  INTERNAL_BF16_ARITHMETIC_OP(!=)
  INTERNAL_BF16_ARITHMETIC_OP(<)
  INTERNAL_BF16_ARITHMETIC_OP(<=)
  INTERNAL_BF16_ARITHMETIC_OP(>)
  INTERNAL_BF16_ARITHMETIC_OP(>=)

#undef INTERNAL_BF16_ARITHMETIC_OP
#undef INTERNAL_BF16_ARITHMETIC_ASSIGN_OP

  // Unary negation.
  friend BF16 operator-(BF16 x) {
    BF16 result;
    result.value_ = x.value_ ^ 0x8000;
    return result;
  }

  // Unary plus
  friend BF16 operator+(BF16 x) { return x; }

 private:
  uint16_t value_;
};

inline bool isinf(BF16 x) { return std::isinf(static_cast<float>(x)); }
inline bool signbit(BF16 x) { return std::signbit(static_cast<float>(x)); }
inline bool isnan(BF16 x) { return std::isnan(static_cast<float>(x)); }
inline bool isfinite(BF16 x) { return std::isfinite(static_cast<float>(x)); }
inline BF16 abs(BF16 x) { return BF16(std::abs(static_cast<float>(x))); }
inline BF16 exp(BF16 x) { return BF16(std::exp(static_cast<float>(x))); }
inline BF16 exp2(BF16 x) { return BF16(std::exp2(static_cast<float>(x))); }
inline BF16 expm1(BF16 x) { return BF16(std::expm1(static_cast<float>(x))); }
inline BF16 log(BF16 x) { return BF16(std::log(static_cast<float>(x))); }
inline BF16 log1p(BF16 x) { return BF16(std::log1p(static_cast<float>(x))); }
inline BF16 log10(BF16 x) { return BF16(std::log10(static_cast<float>(x))); }
inline BF16 log2(BF16 x) { return BF16(std::log2(static_cast<float>(x))); }
inline BF16 sqrt(BF16 x) { return BF16(std::sqrt(static_cast<float>(x))); }
inline BF16 pow(BF16 x, BF16 y) {
  return BF16(std::pow(static_cast<float>(x), static_cast<float>(y)));
}
inline BF16 sin(BF16 x) { return BF16(std::sin(static_cast<float>(x))); }
inline BF16 cos(BF16 x) { return BF16(std::cos(static_cast<float>(x))); }
inline BF16 tan(BF16 x) { return BF16(std::tan(static_cast<float>(x))); }
inline BF16 asin(BF16 x) { return BF16(std::asin(static_cast<float>(x))); }
inline BF16 acos(BF16 x) { return BF16(std::acos(static_cast<float>(x))); }
inline BF16 atan(BF16 x) { return BF16(std::atan(static_cast<float>(x))); }
inline BF16 sinh(BF16 x) { return BF16(std::sinh(static_cast<float>(x))); }
inline BF16 cosh(BF16 x) { return BF16(std::cosh(static_cast<float>(x))); }
inline BF16 tanh(BF16 x) { return BF16(std::tanh(static_cast<float>(x))); }
inline BF16 asinh(BF16 x) { return BF16(std::asinh(static_cast<float>(x))); }
inline BF16 acosh(BF16 x) { return BF16(std::acosh(static_cast<float>(x))); }
inline BF16 atanh(BF16 x) { return BF16(std::atanh(static_cast<float>(x))); }
inline BF16 floor(BF16 x) { return BF16(std::floor(static_cast<float>(x))); }
inline BF16 trunc(BF16 x) { return BF16(std::trunc(static_cast<float>(x))); }
inline BF16 rint(BF16 x) { return BF16(std::rint(static_cast<float>(x))); }
inline BF16 ceil(BF16 x) { return BF16(std::ceil(static_cast<float>(x))); }
inline BF16 fmod(BF16 x, BF16 y) {
  return BF16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}
inline BF16 fmin(BF16 a, BF16 b) {
  return BF16(std::fmin(static_cast<float>(a), static_cast<float>(b)));
}
inline BF16 fmax(BF16 a, BF16 b) {
  return BF16(std::fmax(static_cast<float>(a), static_cast<float>(b)));
}

namespace internal {

inline BF16 NumericF32ToBF16RoundNearestEven(float v) {
  ABSL_CHECK(!std::isnan(v));

  uint32_t input = absl::bit_cast<uint32_t>(v);
  const uint32_t lsb = (input >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  return absl::bit_cast<BF16, uint16_t>(input >> 16);
}

inline BF16 F32ToBF16RoundNearestEven(float v) {
  if (std::isnan(v)) {
    return BF16(BF16::bitcast_construct_t{},
                static_cast<uint16_t>(
                    (absl::bit_cast<uint32_t>(v) | 0x00200000u) >> 16));
  }
  return NumericF32ToBF16RoundNearestEven(v);
}

inline float BF16ToFloat(BF16 v) {
  return absl::bit_cast<float>(
      static_cast<uint32_t>(absl::bit_cast<uint16_t>(v)) << 16);
}
}  // namespace internal
}  // namespace shlo_ref

// Specialized std::numeric_limits for BF16
namespace std {
template <>
class numeric_limits<shlo_ref::BF16> {
 public:
  static constexpr bool is_specialized = true;      // NOLINT
  static constexpr bool is_signed = true;           // NOLINT
  static constexpr bool is_integer = false;         // NOLINT
  static constexpr bool is_exact = false;           // NOLINT
  static constexpr bool has_infinity = true;        // NOLINT
  static constexpr bool has_quiet_NaN = true;       // NOLINT
  static constexpr bool has_signaling_NaN = true;   // NOLINT
  static constexpr float_denorm_style has_denorm =  // NOLINT
      std::denorm_present;
  static constexpr bool has_denorm_loss = false;    // NOLINT
  static constexpr float_round_style round_style =  // NOLINT
      numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;   // NOLINT
  static constexpr bool is_bounded = true;  // NOLINT
  static constexpr bool is_modulo = false;  // NOLINT
  static constexpr int digits = 8;          // NOLINT
  static constexpr int digits10 = 2;        // NOLINT
  static constexpr int max_digits10 = 4;    // NOLINT
  static constexpr int radix = 2;           // NOLINT
  static constexpr int min_exponent =       // NOLINT
      numeric_limits<float>::min_exponent;
  static constexpr int min_exponent10 =  // NOLINT
      numeric_limits<float>::min_exponent10;
  static constexpr int max_exponent =  // NOLINT
      numeric_limits<float>::max_exponent;
  static constexpr int max_exponent10 =  // NOLINT
      numeric_limits<float>::max_exponent10;
  static constexpr bool traps = numeric_limits<float>::traps;  // NOLINT
  static constexpr bool tinyness_before =                      // NOLINT
      numeric_limits<float>::tinyness_before;

  static constexpr shlo_ref::BF16(min)() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x0080));
  }
  static constexpr shlo_ref::BF16 lowest() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0xff7f));
  }
  static constexpr shlo_ref::BF16(max)() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x7f7f));
  }
  static constexpr shlo_ref::BF16 epsilon() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x3c00));
  }
  static constexpr shlo_ref::BF16 round_error() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x3f00));
  }
  static constexpr shlo_ref::BF16 infinity() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x7f80));
  }
  static constexpr shlo_ref::BF16 quiet_NaN() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x7fc0));
  }
  static constexpr shlo_ref::BF16 signaling_NaN() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x7f81));
  }
  static constexpr shlo_ref::BF16 denorm_min() {
    return shlo_ref::BF16(shlo_ref::BF16::bitcast_construct_t{},
                          static_cast<uint16_t>(0x0001));
  }
};
}  // namespace std

#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_BF16_H_
