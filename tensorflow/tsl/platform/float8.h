/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
#define TENSORFLOW_TSL_PLATFORM_FLOAT8_H_

// 8-bit Floating Point Interchange Format, as described by
//   https://arxiv.org/abs/2209.05433

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

#include "absl/numeric/bits.h"
#include "third_party/eigen3/Eigen/Core"

namespace tsl {

namespace float8_internal {

// Forward-declarations of classes.
class float8_e4m3fn;
class float8_e4m3b11;
class float8_e5m2;

template <typename Derived>
class float8_base {
 public:
  constexpr float8_base() : rep_(0) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_base(T f)
      : float8_base(ConvertFrom(static_cast<float>(f)).rep(),
                    ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC float8_base(double f64)
      : float8_base(ConvertFrom(f64).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC float8_base(float f32)
      : float8_base(ConvertFrom(f32).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC float8_base(Eigen::bfloat16 bf16)
      : float8_base(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC float8_base(Eigen::half f16)
      : float8_base(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

  constexpr uint8_t rep() const { return rep_; }

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(derived()));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return (rep() & 0x7F) != 0;
  }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep() ^ 0x80), ConstructFromRepTag{});
  }

  constexpr const Derived& derived() const {
    return *static_cast<const Derived*>(this);
  }

  constexpr Derived& derived() { return *static_cast<Derived*>(this); }

  static constexpr Derived FromRep(uint8_t rep) {
    return Derived(rep, ConstructFromRepTag{});
  }

  // Conversions allowing saturation and truncation.
  template <bool kSaturate = false, bool kTruncate = false, typename From>
  static inline EIGEN_DEVICE_FUNC Derived ConvertFrom(const From& from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static inline EIGEN_DEVICE_FUNC To ConvertTo(const Derived& from);

  // Operators via float32.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator+(const Derived& other) const {
    return Derived{float{derived()} + float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator-(const Derived& other) const {
    return Derived{float{derived()} - float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator*(const Derived& other) const {
    return Derived{float{derived()} * float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator/(const Derived& other) const {
    return Derived{float{derived()} / float{other}};
  }

  constexpr bool operator==(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }

  constexpr bool operator!=(const Derived& other) const {
    return Compare(derived(), other) != Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
      const Derived& other) const {
    return Compare(derived(), other) == Ordering::kLess;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
      const Derived& other) const {
    return Compare(derived(), other) <= Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
      const Derived& other) const {
    return Compare(derived(), other) == Ordering::kGreater;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
      const Derived& other) const {
    Ordering ordering = Compare(derived(), other);
    return ordering == Ordering::kGreater || ordering == Ordering::kEquivalent;
  }

  // Compound assignment.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator+=(
      const Derived& other) {
    derived() = derived() + other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator-=(
      const Derived& other) {
    derived() = derived() - other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator*=(
      const Derived& other) {
    derived() = derived() * other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator/=(
      const Derived& other) {
    derived() = derived() / other;
    return derived();
  }

 private:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};

  constexpr float8_base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::pair<uint8_t, uint8_t>
  SignAndMagnitude(Derived x) {
    const uint8_t x_abs_bits =
        Eigen::numext::bit_cast<uint8_t>(Eigen::numext::abs(x));
    const uint8_t x_bits = Eigen::numext::bit_cast<uint8_t>(x);
    const uint8_t x_sign = x_bits ^ x_abs_bits;
    return {x_sign, x_abs_bits};
  }
  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int8_t
  SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
    return magnitude ^ (static_cast<int8_t>(sign) < 0 ? -1 : 0);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC friend Ordering Compare(
      const Derived& lhs, const Derived& rhs) {
    if (Eigen::numext::isnan(lhs) || Eigen::numext::isnan(rhs)) {
      return Ordering::kUnordered;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return Ordering::kEquivalent;
    }
    int8_t lhs_twos_complement =
        SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
    int8_t rhs_twos_complement =
        SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
    if (lhs_twos_complement < rhs_twos_complement) {
      return Ordering::kLess;
    }
    if (lhs_twos_complement > rhs_twos_complement) {
      return Ordering::kGreater;
    }
    return Ordering::kEquivalent;
  }

  uint8_t rep_;
};

template <typename Derived, typename T>
using EnableIsFloat8Subtype =
    std::enable_if_t<std::is_base_of_v<float8_base<Derived>, Derived>, T>;

}  // namespace float8_internal
}  // namespace tsl

// Eigen-specific overrides.
namespace Eigen {
namespace numext {

// NOLINTBEGIN: bit_cast expects to take its parameter by reference.
template <typename Tgt>
constexpr tsl::float8_internal::EnableIsFloat8Subtype<Tgt, Tgt> bit_cast(
    const uint8_t& src) {
  return Tgt::FromRep(src);
}
// NOLINTEND

template <typename Float8>
constexpr uint8_t bit_cast(
    const tsl::float8_internal::float8_base<Float8>& src) {
  return src.rep();
}

}  // namespace numext
}  // namespace Eigen

namespace tsl {

namespace float8_internal {
class float8_e4m3fn : public float8_base<float8_e4m3fn> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by 0bS111'1111.
  // The "fn" suffix is for consistency with the corresponding LLVM/MLIR type,
  // signaling this type is not consistent with IEEE-754.  The "f" indicates
  // it is finite values only. The "n" indicates it includes NaNs, but only
  // at the outer range.
 private:
  using Base = float8_base<float8_e4m3fn>;
  friend class float8_base<float8_e4m3fn>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e5m2& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e4m3b11& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
};

class float8_e4m3b11 : public float8_base<float8_e4m3b11> {
  // Exponent: 4, Mantissa: 3, bias: 11.
  // Extended range: no inf, NaN represented by 0b1000'0000.
 private:
  using Base = float8_base<float8_e4m3b11>;
  friend class float8_base<float8_e4m3b11>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(const float8_e5m2& f8)
      : float8_e4m3b11(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(const float8_e4m3fn& f8)
      : float8_e4m3b11(ConvertFrom(f8)) {}

  constexpr float8_e4m3b11 operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return Base::operator-();
  }

  float8_e4m3b11 operator-(const float8_e4m3b11& other) const {
    return Base::operator-(other);
  }

  explicit EIGEN_DEVICE_FUNC operator bool() const { return rep() != 0; }
};

class float8_e5m2 : public float8_base<float8_e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = float8_base<float8_e5m2>;
  friend class float8_base<float8_e5m2>;
  using Base::Base;

 public:
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3fn f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3b11 f8)
      : float8_e5m2(ConvertFrom(f8)) {}
};

constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }

constexpr double ConstexprCeil(double x) {
  constexpr double kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<double>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;
  }
  const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
  return x_trunc < x ? x_trunc + 1.0 : x_trunc;
}

constexpr double ConstexprFloor(double x) { return -ConstexprCeil(-x); }

constexpr double kLog10Of2 = 0.3010299956639812;
// C17 5.2.4.2.2p11:
// "number of decimal digits, q, such that any floating-point number with q
// decimal digits can be rounded into a floating-point number with p radix b
// digits and back again without change to the q decimal digits"
// floor((p - 1) * log10(2));
constexpr int Digits10FromDigits(int digits) {
  return static_cast<int>(ConstexprFloor((digits - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "number of decimal digits, n, such that any floating-point number with p
// radix b digits can be rounded to a floating-point number with n decimal
// digits and back again without change to the value"
// ceil(1 + p * log10(2));
constexpr int MaxDigits10FromDigits(int digits) {
  return static_cast<int>(ConstexprCeil(1.0 + (digits * kLog10Of2)));
}

// C17 5.2.4.2.2p11:
// "minimum negative integer such that 10 raised to that power is in the range
// of normalized floating-point numbers"
// ceil(log10(2**(emin - 1))) == ceil((emin - 1) * log10(2));
constexpr int MinExponent10FromMinExponent(int min_exponent) {
  return static_cast<int>(ConstexprCeil((min_exponent - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "maximum integer such that 10 raised to that power is in the range of
// representable finite floating-point numbers"
// floor(log10((1 - 2**-p) * 2**emax)) == floor(log10(1 - 2**-p) +
// emax * log10(2))
constexpr int MaxExponent10FromMaxExponentAndDigits(int max_exponent,
                                                    int digits) {
  // We only support digits in {3,4}. This table would grow if we wanted to
  // handle more values.
  constexpr double kLog10OfOnePredecessor[] = {
      // log10(1 - 2**-3)
      -0.057991946977686754,
      // log10(1 - 2**-4)
      -0.028028723600243537,
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 3] +
                                         max_exponent * kLog10Of2));
}

// Structures for use in specializing std::numeric_limits.
struct numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_integer = false;
  static inline constexpr const bool is_exact = false;
  static inline constexpr const bool has_quiet_NaN = true;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_present;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_to_nearest;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const bool is_modulo = false;
  static inline constexpr const int radix = std::numeric_limits<float>::radix;
  static inline constexpr const bool traps = std::numeric_limits<float>::traps;
  static inline constexpr const bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;
  static inline constexpr const int digits = 0;
  static inline constexpr const int digits10 = 0;
  static inline constexpr const int max_digits10 = 0;
  static inline constexpr const int min_exponent = 0;
  static inline constexpr const int min_exponent10 = 0;
  static inline constexpr const int max_exponent = 0;
  static inline constexpr const int max_exponent10 = 0;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND
};

struct numeric_limits_float8_e4m3fn : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 7;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1111 - 7) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  // 1.0 * 2^(0b0001 - 7) = 1.0 * 2^-6 = 0.015625
  static constexpr float8_e4m3fn min() {
    return float8_e4m3fn::FromRep(0b0'0001 << kMantissaBits);
  }
  // -(1 + 0b110 * 2^-3) * 2^(0b1111 - 7) = -1.75 * 2^8 = 448
  static constexpr float8_e4m3fn lowest() {
    return float8_e4m3fn::FromRep(0b1'1111'110);
  }
  // (1 + 0b110 * 2^-3) * 2**(0b1111 - 7) = 1.75 * 2^8 = 448
  static constexpr float8_e4m3fn max() {
    return float8_e4m3fn::FromRep(0b0'1111'110);
  }
  // 1.0 * 2^-3 = 0.125
  static constexpr float8_e4m3fn epsilon() {
    return float8_e4m3fn::FromRep((-kMantissaBits + kExponentBias)
                                  << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr float8_e4m3fn round_error() {
    return float8_e4m3fn::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr float8_e4m3fn infinity() {
    return float8_e4m3fn::FromRep(0b0'1111'111);
  }
  // NaN.
  static constexpr float8_e4m3fn quiet_NaN() {
    return float8_e4m3fn::FromRep(0b0'1111'111);
  }
  static constexpr float8_e4m3fn signaling_NaN() {
    return float8_e4m3fn::FromRep(0b0'1111'111);
  }
  // 1.0 * 2^(-7 - 3 + 1) = 1.0 * 2^-9 = 0.001953125
  static constexpr float8_e4m3fn denorm_min() {
    return float8_e4m3fn::FromRep(0b0'0000'001);
  }
};

struct numeric_limits_float8_e4m3b11 : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 11;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1111 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  // 1.0 * 2^(0b0001 - 11) = 1.0 * 2^-10 = 0.0009765625
  static constexpr float8_e4m3b11 min() {
    return float8_e4m3b11::FromRep(1 << kMantissaBits);
  }
  // -(1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = -1.875 * 2^4 = -30
  static constexpr float8_e4m3b11 lowest() {
    return float8_e4m3b11::FromRep(0b1'1111'111);
  }
  // (1 + 0b111 * 2^-3) * 2^(0b1111 - 11) = 1.875 * 2^4 = 30
  static constexpr float8_e4m3b11 max() {
    return float8_e4m3b11::FromRep(0b0'1111'111);
  }
  // 1.0 * 2^-3 = 0.125
  static constexpr float8_e4m3b11 epsilon() {
    return float8_e4m3b11::FromRep((-kMantissaBits + kExponentBias)
                                   << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr float8_e4m3b11 round_error() {
    return float8_e4m3b11::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr float8_e4m3b11 infinity() {
    return float8_e4m3b11::FromRep(0b1'0000'000);
  }
  // NaN.
  static constexpr float8_e4m3b11 quiet_NaN() {
    return float8_e4m3b11::FromRep(0b1'0000'000);
  }
  static constexpr float8_e4m3b11 signaling_NaN() {
    return float8_e4m3b11::FromRep(0b1'0000'000);
  }
  // 1.0 * 2^(-11 - 3 + 1) = 1.0 * 2^-13 = 0.0001220703125
  static constexpr float8_e4m3b11 denorm_min() {
    return float8_e4m3b11::FromRep(0b0'0000'001);
  }
};

struct numeric_limits_float8_e5m2 : public numeric_limits_float8_base {
 private:
  static inline constexpr const int kExponentBias = 15;
  static inline constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 3;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent = 0b11111 - kExponentBias;
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  static inline constexpr const bool is_iec559 = true;
  static inline constexpr const bool has_infinity = true;
  static inline constexpr const bool has_signaling_NaN = true;
  // NOLINTEND

  // 1.0 * 2^(0b00001 - 15) = 1.0 * 2^-14 = 0.00006103515625
  static constexpr float8_e5m2 min() {
    return float8_e5m2::FromRep(1 << kMantissaBits);
  }
  // -(1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = -1.75 * 2^15 = -57344
  static constexpr float8_e5m2 lowest() {
    return float8_e5m2::FromRep(0b1'11110'11);
  }
  // (1 + 0b11 * 2^-2) * 2^(0b11110 - 15) = 1.75 * 2^15 = 57344
  static constexpr float8_e5m2 max() {
    return float8_e5m2::FromRep(0b0'11110'11);
  }
  // 1.0 * 2^-2 = 0.25
  static constexpr float8_e5m2 epsilon() {
    return float8_e5m2::FromRep((-kMantissaBits + kExponentBias)
                                << kMantissaBits);
  }
  // 1.0 * 2^-1 = 0.5
  static constexpr float8_e5m2 round_error() {
    return float8_e5m2::FromRep((-1 + kExponentBias) << kMantissaBits);
  }
  static constexpr float8_e5m2 infinity() {
    return float8_e5m2::FromRep(0b0'11111'00);
  }
  static constexpr float8_e5m2 quiet_NaN() {
    // IEEE 754-2019 6.2.1: "All binary NaN bit strings have the sign bit S set
    // to 0 or 1 and all the bits of the biased exponent field E set to 1
    // (see 3.4). A quiet NaN bit string should be encoded with the first bit
    // (d1) of the trailing significand field T being 1."
    return float8_e5m2::FromRep(0b0'11111'10);
  }
  static constexpr float8_e5m2 signaling_NaN() {
    // IEEE 754-2019 6.2.1: "A signaling NaN bit string should be encoded with
    // the first bit of the trailing significand field being 0."
    return float8_e5m2::FromRep(0b0'11111'01);
  }
  // 1.0 * 2^(-15 - 2 + 1) = 1.0 * 2^-16 = 0.0000152587890625
  static constexpr float8_e5m2 denorm_min() {
    return float8_e5m2::FromRep(0b0'00000'01);
  }
};

}  // namespace float8_internal
}  // namespace tsl

namespace std {
// Standard-library overrides.  Note that these are picked up by Eigen as well.
template <>
struct numeric_limits<tsl::float8_internal::float8_e4m3fn>
    : public tsl::float8_internal::numeric_limits_float8_e4m3fn {};

template <>
struct numeric_limits<tsl::float8_internal::float8_e4m3b11>
    : public tsl::float8_internal::numeric_limits_float8_e4m3b11 {};

template <>
struct numeric_limits<tsl::float8_internal::float8_e5m2>
    : public tsl::float8_internal::numeric_limits_float8_e5m2 {};
}  // namespace std

namespace tsl {
namespace float8_internal {

// Free-functions for use with ADL and in Eigen.
constexpr inline float8_e4m3fn abs(const float8_e4m3fn& a) {
  return float8_e4m3fn::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const float8_e4m3fn& a) {
  return abs(a).rep() == std::numeric_limits<float8_e4m3fn>::quiet_NaN().rep();
}

constexpr inline float8_e4m3b11 abs(const float8_e4m3b11& a) {
  return (a.rep() & 0b0'1111'111) == 0
             ? float8_e4m3b11::FromRep(a.rep())
             : float8_e4m3b11::FromRep(a.rep() & 0b0'1111'111);
}

constexpr inline bool(isnan)(const float8_e4m3b11& a) {
  return a.rep() == std::numeric_limits<float8_e4m3b11>::quiet_NaN().rep();
}

constexpr inline float8_e5m2 abs(const float8_e5m2& a) {
  return float8_e5m2::FromRep(a.rep() & 0b0'11111'11);
}

constexpr inline bool(isnan)(const float8_e5m2& a) {
  return abs(a).rep() > std::numeric_limits<float8_e5m2>::infinity().rep();
}

template <typename Float8>
constexpr inline bool(isinf)(const float8_base<Float8>& a) {
  return std::numeric_limits<Float8>::has_infinity
             ? abs(a.derived()).rep() ==
                   std::numeric_limits<Float8>::infinity().rep()
             : false;  // No inf representation.
}

template <typename Float8>
constexpr inline bool(isfinite)(const float8_base<Float8>& a) {
  return !isnan(a.derived()) && !isinf(a.derived());
}

}  // namespace float8_internal
}  // namespace tsl

namespace Eigen {
namespace numext {

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isinf)(
    const tsl::float8_internal::float8_e4m3fn& x) {
  return (tsl::float8_internal::isinf)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isinf)(
    const tsl::float8_internal::float8_e4m3b11& x) {
  return (tsl::float8_internal::isinf)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isinf)(
    const tsl::float8_internal::float8_e5m2& x) {
  return (tsl::float8_internal::isinf)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isnan)(
    const tsl::float8_internal::float8_e4m3fn& x) {
  return (tsl::float8_internal::isnan)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isnan)(
    const tsl::float8_internal::float8_e4m3b11& x) {
  return (tsl::float8_internal::isnan)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isnan)(
    const tsl::float8_internal::float8_e5m2& x) {
  return (tsl::float8_internal::isnan)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isfinite)(
    const tsl::float8_internal::float8_e4m3fn& x) {
  return (tsl::float8_internal::isfinite)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isfinite)(
    const tsl::float8_internal::float8_e4m3b11& x) {
  return (tsl::float8_internal::isfinite)(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool(isfinite)(
    const tsl::float8_internal::float8_e5m2& x) {
  return (tsl::float8_internal::isfinite)(x);
}

}  // namespace numext
}  // namespace Eigen

namespace tsl {
namespace float8_internal {

template <typename Float8>
std::ostream& operator<<(std::ostream& os, const float8_base<Float8>& f8) {
  os << static_cast<float>(f8.derived());
  return os;
}

//==============================================================================
// Inline conversion routines between float8 and other types.
//==============================================================================

// Helper for getting a bit representation provided a byte size.
template <int kNumBytes>
using GetUnsignedInteger =
    typename Eigen::numext::get_integer_by_size<kNumBytes>::unsigned_type;

// Converts between two floating-point types.
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& from) {
    return from;
  }
};

template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/false, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/false,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};
template <typename Scalar>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/true, /*kTruncate=*/true,
                   /*EnableIf=*/void> : public IdentityConversion<Scalar> {};

template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = Eigen::NumTraits<Float>::digits() - 1;
  static constexpr int kExponentBits = kBits - kMantissaBits - 1;
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};

template <typename Float>
struct Traits : public TraitsBase<Float> {};

template <>
struct Traits<float8_e4m3b11> : public TraitsBase<float8_e4m3b11> {
  static constexpr int kExponentBias = 11;
};

template <typename Bits>
constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
  // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = roundoff == 0
                  ? 0
                  : ((bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
  return bits + bias;
}

template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<From, To>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToBits = ToTraits::kBits;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;

  // `WideBits` is wide enough to accomodate the largest exponent and mantissa
  // in either `From` or `To`.
  static constexpr int kWideBits =
      (std::max(kToMantissaBits, kFromMantissaBits)) +  // Max significand.
      (std::max(kToExponentBits, kFromExponentBits));   // Max exponent.
  static constexpr int kWideBytes = (kWideBits + (CHAR_BIT - 1)) / CHAR_BIT;
  using WideBits = GetUnsignedInteger<kWideBytes>;
  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;

  static EIGEN_DEVICE_FUNC inline To run(const From& from) {
    // Shift bits to destination type, without sign bit.
    const bool from_sign_bit =
        Eigen::numext::bit_cast<FromBits>(from) >> (kFromBits - 1);
    const FromBits from_bits =
        Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::infinity()
                           : Eigen::NumTraits<To>::infinity();
    }
    if (Eigen::numext::isnan(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::quiet_NaN()
                           : Eigen::NumTraits<To>::quiet_NaN();
    }
    if (from_bits == 0) {
      return from_sign_bit ? -To{} : To{};
    }

    const int biased_from_exponent = from_bits >> kFromMantissaBits;

    // `To` supports more exponents near zero which means that some subnormal
    // values in `From` may become normal.
    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent) {
      if (biased_from_exponent == 0) {
        // Subnormals.
        ToBits bits = from_bits;

        // All subnormals become normalized when casting to a type with a larger
        // number of exponent bits.  We do this by setting the normalized
        // mantissa bits in the source type, shifting it up to the destination
        // type, then inserting the exponent bits.
        const int normalization_factor =
            absl::countl_zero(from_bits) - (kFromBits - kFromMantissaBits) + 1;
        // Shift the mantissa to account for the number of leading zeros.
        bits <<= normalization_factor + kDigitShift;
        // Clear the hidden bit.
        bits &= ~(ToBits{1} << kToMantissaBits);
        // Insert the exponent bits.
        bits |= static_cast<ToBits>(kExponentOffset - normalization_factor + 1)
                << kToMantissaBits;
        To to = Eigen::numext::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }
    // `To` supports fewer exponents near zero which means that some values in
    // `From` may become subnormal.
    if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent) {
      const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
      const int biased_to_exponent = unbiased_exponent + kToExponentBias;
      // Subnormals and zero.
      if (biased_to_exponent <= 0) {
        // Round and shift mantissa down.
        int exponent_shift = -kDigitShift - biased_to_exponent + 1;

        // Insert the implicit leading 1 bit on the mantissa.  This assumes
        // the input is normalized.  If it is not, then the mantissa bits -
        // including the implicit one - will be shifted to zero.
        // NOTE: we need to round again from the original from_bits, otherwise
        // the lower precision bits may already be lost.  There is an edge-case
        // where rounding to a normalized value would normally round down,
        // but for a subnormal, we need to round up.
        FromBits rounded_from_bits = ((from_bits & FromTraits::kMantissaMask) |
                                      (FromBits{1} << kFromMantissaBits));
        ToBits bits = 0;
        // To avoid UB, limit rounding and shifting to the full mantissa plus
        // leading 1.
        if (exponent_shift <= kFromMantissaBits + 1) {
          if constexpr (!kTruncate) {
            rounded_from_bits =
                RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
          }
          bits = (rounded_from_bits >> exponent_shift);
        }
        // Insert sign and return.
        To to = Eigen::numext::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }

    // Round the mantissa if it is shrinking.
    WideBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
      if constexpr (!kTruncate) {
        rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
      }
      // Zero-out tail bits.
      rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
    }

    // Re-bias the exponent.
    rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                         << kFromMantissaBits;

    ToBits bits;
    // Check for overflows by aligning the significands. We always align the
    // narrower significand to the wider significand.
    const WideBits kToHighestRep =
        Eigen::numext::bit_cast<ToBits>(Eigen::NumTraits<To>::highest());
    WideBits aligned_highest{kToHighestRep};
    if constexpr (kDigitShift < 0) {
      aligned_highest <<= -kDigitShift;
      // Shift down, all dropped bits should already be zero.
      bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
    } else if constexpr (kDigitShift >= 0) {
      // Shift up, inserting zeros in the newly created digits.
      rounded_from_bits <<= kDigitShift;
      bits = ToBits{rounded_from_bits};
    }

    To to = Eigen::numext::bit_cast<To>(bits);
    // `From` supports larger values than `To`, we may overflow.
    if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                 std::numeric_limits<To>::digits) <
                  std::make_pair(std::numeric_limits<From>::max_exponent,
                                 std::numeric_limits<From>::digits)) {
      if (rounded_from_bits > aligned_highest) {
        // Overflowed values map to highest or infinity depending on kSaturate.
        to = kSaturate ? Eigen::NumTraits<To>::highest()
                       : Eigen::NumTraits<To>::infinity();
      }
    }
    // Insert sign bit.
    return from_sign_bit ? -to : to;
  }
};

// Saturation has no impact when casting e4m3 to e5m2.
template <bool kTruncate>
struct ConvertImpl<float8_e4m3fn, float8_e5m2, true, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const float8_e4m3fn& from) {
    return ConvertImpl<float8_e4m3fn, float8_e5m2, false, kTruncate>::run(from);
  }
};

template <bool kSaturate, bool kTruncate>
struct ConvertImpl<Eigen::half, float8_e5m2, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline float8_e5m2 run(const Eigen::half& from) {
    uint16_t from_bits = Eigen::numext::bit_cast<uint16_t>(from);

    // Special values (Inf or NaN).
    uint16_t abs_bits = from_bits & 0x7FFF;
    if (abs_bits == 0x7C00) {
      return float8_e5m2::FromRep(from_bits >> 8);
    } else if (abs_bits > 0x7C00) {
      // IEEE 754-2019 6.2.1: "A quiet NaN bit string should be encoded with the
      // first bit (d1) of the trailing significand field T being 1."
      // IEEE 754-2019 6.2.3: "Conversion of a quiet NaN to a floating-point
      // format of the same or a different radix that does not allow the payload
      // to be preserved, shall return a quiet NaN [...]"
      return float8_e5m2::FromRep((from_bits >> 8) | 0b0'00000'10);
    }

    if constexpr (!kTruncate) {
      from_bits = RoundBitsToNearestEven(from_bits, 8);
      // Rounding can cause an overflow to infinity. Clamp to the largest finite
      // value if saturation is requested.
      if constexpr (kSaturate) {
        const float8_e5m2 kHighest = Eigen::NumTraits<float8_e5m2>::highest();
        if ((from_bits & 0x7F00) > static_cast<uint16_t>(kHighest.rep()) << 8) {
          const bool from_sign_bit = from_bits >> 15;
          return from_sign_bit ? -kHighest : kHighest;
        }
      }
    }
    return float8_e5m2::FromRep(from_bits >> 8);
  }
};

// Saturation and truncation have no impact when casting e5m2 to Eigen::half.
template <bool kSaturate, bool kTruncate>
struct ConvertImpl<float8_e5m2, Eigen::half, kSaturate, kTruncate> {
  static EIGEN_DEVICE_FUNC inline Eigen::half run(const float8_e5m2& from) {
    return Eigen::numext::bit_cast<Eigen::half>(
        static_cast<uint16_t>(static_cast<uint16_t>(from.rep()) << 8));
  }
};

template <typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC Derived float8_base<Derived>::ConvertFrom(const From& from) {
  return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from);
}

template <typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To float8_base<Derived>::ConvertTo(const Derived& from) {
  return ConvertImpl<Derived, To, kSaturate, kTruncate>::run(from);
}

}  // namespace float8_internal

// Exported types.
using float8_e4m3fn = float8_internal::float8_e4m3fn;
using float8_e4m3b11 = float8_internal::float8_e4m3b11;
using float8_e5m2 = float8_internal::float8_e5m2;

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
