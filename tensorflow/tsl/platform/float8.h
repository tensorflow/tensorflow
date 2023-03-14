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

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
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
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};

  constexpr float8_base() : rep_(0) {}
  constexpr float8_base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

 public:
  constexpr uint8_t rep() const { return rep_; }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep() ^ 0x80), ConstructFromRepTag{});
  }

  constexpr bool operator==(const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(derived());
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(other);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return true;
    }
    return rep() == other.rep();
  }

  constexpr bool operator!=(const Derived& other) const {
    return !(derived() == other);
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

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
      const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(derived());
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(other);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return false;
    }
    return SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag) <
           SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
      const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(derived());
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(other);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return true;
    }
    return SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag) <=
           SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
      const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(derived());
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(other);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return false;
    }
    return SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag) >
           SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
      const Derived& other) const {
    if (Eigen::numext::isnan(derived()) || Eigen::numext::isnan(other)) {
      return false;
    }
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(derived());
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(other);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return true;
    }
    return SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag) >=
           SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
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
  uint8_t rep_;
};

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

  constexpr float8_e4m3fn(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  constexpr float8_e4m3fn() = default;

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(T f)
      : float8_e4m3fn(ConvertFrom(static_cast<float>(f))) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(double f64)
      : float8_e4m3fn(ConvertFrom(f64)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(float f32)
      : float8_e4m3fn(ConvertFrom(f32)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(Eigen::bfloat16 bf16)
      : float8_e4m3fn(ConvertFrom(bf16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(Eigen::half f16)
      : float8_e4m3fn(ConvertFrom(f16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e5m2& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3fn(const float8_e4m3b11& f8)
      : float8_e4m3fn(ConvertFrom(f8)) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return (rep() & 0x7F) != 0;
  }
};

class float8_e4m3b11 : public float8_base<float8_e4m3b11> {
  // Exponent: 4, Mantissa: 3, bias: 11.
  // Extended range: no inf, NaN represented by 0b1000'0000.
 private:
  using Base = float8_base<float8_e4m3b11>;
  friend class float8_base<float8_e4m3b11>;

  constexpr float8_e4m3b11(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  constexpr float8_e4m3b11() = default;

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(T f)
      : float8_e4m3b11(ConvertFrom(static_cast<float>(f))) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(double f64)
      : float8_e4m3b11(ConvertFrom(f64)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(float f32)
      : float8_e4m3b11(ConvertFrom(f32)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(Eigen::bfloat16 bf16)
      : float8_e4m3b11(ConvertFrom(bf16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(Eigen::half f16)
      : float8_e4m3b11(ConvertFrom(f16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(const float8_e5m2& f8)
      : float8_e4m3b11(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e4m3b11(const float8_e4m3fn& f8)
      : float8_e4m3b11(ConvertFrom(f8)) {}

  constexpr float8_e4m3b11 operator-() const {
    if ((rep() & 0x7f) == 0x00) {
      return float8_e4m3b11(rep(), ConstructFromRepTag{});
    }
    return Base::operator-();
  }

  float8_e4m3b11 operator-(const float8_e4m3b11& other) const {
    return Base::operator-(other);
  }

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const { return rep() != 0; }
};

class float8_e5m2 : public float8_base<float8_e5m2> {
  // Exponent: 5, Mantissa: 2, bias: 15.
  // IEEE 754.
 private:
  using Base = float8_base<float8_e5m2>;
  friend class float8_base<float8_e5m2>;

  constexpr float8_e5m2(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  constexpr float8_e5m2() = default;

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC float8_e5m2(T f)
      : float8_e5m2(ConvertFrom(static_cast<float>(f))) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(double f64)
      : float8_e5m2(ConvertFrom(f64)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float f32)
      : float8_e5m2(ConvertFrom(f32)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(Eigen::bfloat16 bf16)
      : float8_e5m2(ConvertFrom(bf16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(Eigen::half f16)
      : float8_e5m2(ConvertFrom(f16)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3fn f8)
      : float8_e5m2(ConvertFrom(f8)) {}
  explicit EIGEN_DEVICE_FUNC float8_e5m2(float8_e4m3b11 f8)
      : float8_e5m2(ConvertFrom(f8)) {}

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return (rep() & 0x7F) != 0;
  }
};

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
  // NOLINTEND
};

template <typename Derived>
struct numeric_limits_float8 {
  // NOLINTBEGIN: these names must match std::numeric_limits.
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

template <>
struct numeric_limits_float8<float8_e4m3fn>
    : public numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = 0;      // floor(3 * log10(2));
  static inline constexpr const int max_digits10 = 3;  // ceil(4 * log10(2) + 1)
  static inline constexpr const int min_exponent = -5;
  static inline constexpr const int min_exponent10 = -1;
  static inline constexpr const int max_exponent = 9;  // Extended format.
  static inline constexpr const int max_exponent10 = 2;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr float8_e4m3fn min() { return float8_e4m3fn::FromRep(0x08); }
  static constexpr float8_e4m3fn lowest() {
    return float8_e4m3fn::FromRep(0xFE);
  }
  static constexpr float8_e4m3fn max() { return float8_e4m3fn::FromRep(0x7E); }
  static constexpr float8_e4m3fn epsilon() {
    return float8_e4m3fn::FromRep(0x20);
  }
  static constexpr float8_e4m3fn round_error() {
    return float8_e4m3fn::FromRep(0x30);
  }
  static constexpr float8_e4m3fn infinity() {
    return float8_e4m3fn::FromRep(0x7F);
  }  // NaN.
  static constexpr float8_e4m3fn quiet_NaN() {
    return float8_e4m3fn::FromRep(0x7F);
  }
  static constexpr float8_e4m3fn signaling_NaN() {
    return float8_e4m3fn::FromRep(0x7F);
  }
  static constexpr float8_e4m3fn denorm_min() {
    return float8_e4m3fn::FromRep(0x01);
  }
};

template <>
struct numeric_limits_float8<float8_e4m3b11>
    : public numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 4;
  static inline constexpr const int digits10 = 0;      // floor(3 * log10(2));
  static inline constexpr const int max_digits10 = 3;  // ceil(4 * log10(2) + 1)
  static inline constexpr const int min_exponent = (1 - 11) + 1;
  static inline constexpr const int min_exponent10 = -2;
  static inline constexpr const int max_exponent =
      (0b1111 - 11) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 = 1;
  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND

  static constexpr float8_e4m3b11 min() {
    return float8_e4m3b11::FromRep(0x08);
  }
  static constexpr float8_e4m3b11 lowest() {
    return float8_e4m3b11::FromRep(0xFF);
  }
  static constexpr float8_e4m3b11 max() {
    return float8_e4m3b11::FromRep(0x7F);
  }
  static constexpr float8_e4m3b11 epsilon() {
    constexpr int kExponentBias = 11;
    constexpr int kMantissaBits = 3;
    return float8_e4m3b11::FromRep((kExponentBias - kMantissaBits)
                                   << kMantissaBits);
  }
  static constexpr float8_e4m3b11 round_error() {
    constexpr int kExponentBias = 11;
    constexpr int kMantissaBits = 3;
    return float8_e4m3b11::FromRep((kExponentBias - 1) << kMantissaBits);
  }
  static constexpr float8_e4m3b11 infinity() {
    return float8_e4m3b11::FromRep(0x80);
  }  // NaN.
  static constexpr float8_e4m3b11 quiet_NaN() {
    return float8_e4m3b11::FromRep(0x80);
  }
  static constexpr float8_e4m3b11 signaling_NaN() {
    return float8_e4m3b11::FromRep(0x80);
  }
  static constexpr float8_e4m3b11 denorm_min() {
    return float8_e4m3b11::FromRep(0x01);
  }
};

template <>
struct numeric_limits_float8<float8_e5m2> : public numeric_limits_float8_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = 3;
  static inline constexpr const int digits10 = 0;      // floor(2 * log10(2))
  static inline constexpr const int max_digits10 = 2;  // ceil(3 * log10(2) + 1)
  static inline constexpr const int min_exponent = -13;
  static inline constexpr const int min_exponent10 = -4;
  static inline constexpr const int max_exponent = 16;
  static inline constexpr const int max_exponent10 = 4;
  static inline constexpr const bool is_iec559 = true;
  static inline constexpr const bool has_infinity = true;
  static inline constexpr const bool has_signaling_NaN = true;
  // NOLINTEND

  static constexpr float8_e5m2 min() { return float8_e5m2::FromRep(0x04); }
  static constexpr float8_e5m2 lowest() { return float8_e5m2::FromRep(0xFB); }
  static constexpr float8_e5m2 max() { return float8_e5m2::FromRep(0x7B); }
  static constexpr float8_e5m2 epsilon() { return float8_e5m2::FromRep(0x34); }
  static constexpr float8_e5m2 round_error() {
    return float8_e5m2::FromRep(0x38);
  }
  static constexpr float8_e5m2 infinity() { return float8_e5m2::FromRep(0x7C); }
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
  static constexpr float8_e5m2 denorm_min() {
    return float8_e5m2::FromRep(0x01);
  }
};

}  // namespace float8_internal
}  // namespace tsl

namespace std {
// Standard-library overrides.  Note that these are picked up by Eigen as well.
template <>
struct numeric_limits<tsl::float8_internal::float8_e4m3fn>
    : public tsl::float8_internal::numeric_limits_float8<
          tsl::float8_internal::float8_e4m3fn> {};

template <>
struct numeric_limits<tsl::float8_internal::float8_e4m3b11>
    : public tsl::float8_internal::numeric_limits_float8<
          tsl::float8_internal::float8_e4m3b11> {};

template <>
struct numeric_limits<tsl::float8_internal::float8_e5m2>
    : public tsl::float8_internal::numeric_limits_float8<
          tsl::float8_internal::float8_e5m2> {};
}  // namespace std

namespace tsl {
namespace float8_internal {

// Free-functions for use with ADL and in Eigen.
constexpr inline float8_e4m3fn abs(const float8_e4m3fn& a) {
  return float8_e4m3fn::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e4m3fn& a) {
  return (a.rep() & 0x7F) == 0x7F;
}

constexpr inline bool isinf(const float8_e4m3fn& a) {
  return false;  // No inf representation.
}

constexpr inline bool isfinite(const float8_e4m3fn& a) {
  return !isnan(a) && !isinf(a);
}

constexpr inline float8_e4m3b11 abs(const float8_e4m3b11& a) {
  return (a.rep() & 0x7F) == 0 ? float8_e4m3b11::FromRep(a.rep())
                               : float8_e4m3b11::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e4m3b11& a) { return a.rep() == 0x80; }

constexpr inline bool isinf(const float8_e4m3b11& a) {
  return false;  // No inf representation.
}

constexpr inline bool isfinite(const float8_e4m3b11& a) {
  return !isnan(a) && !isinf(a);
}

constexpr inline float8_e5m2 abs(const float8_e5m2& a) {
  return float8_e5m2::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e5m2& a) {
  return (a.rep() & 0x7F) > 0x7C;
}

constexpr inline bool isinf(const float8_e5m2& a) {
  return (a.rep() & 0x7F) == 0x7C;
}

constexpr inline bool isfinite(const float8_e5m2& a) {
  return !isnan(a) && !isinf(a);
}

template <typename Float8>
std::ostream& operator<<(std::ostream& os, const float8_base<Float8>& f8) {
  os << static_cast<float>(f8.derived());
  return os;
}

//==============================================================================
// Inline conversion routines between float8 and other types.
//==============================================================================

// Helper struct for getting a bit representation provided a byte size.
template <int kNumBytes>
struct GetUnsignedInteger;

template <>
struct GetUnsignedInteger<1> {
  using type = uint8_t;
};

template <>
struct GetUnsignedInteger<2> {
  using type = uint16_t;
};

template <>
struct GetUnsignedInteger<4> {
  using type = uint32_t;
};

template <>
struct GetUnsignedInteger<8> {
  using type = uint64_t;
};

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
  using BitsType = typename GetUnsignedInteger<sizeof(Float)>::type;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = Eigen::NumTraits<Float>::digits() - 1;
  static constexpr int kExponentBits = kBits - kMantissaBits - 1;
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
};

template <typename Float>
struct Traits : public TraitsBase<Float> {
  using Base = TraitsBase<Float>;
  static constexpr int kExponentBias = (1 << (Base::kExponentBits - 1)) - 1;
  static EIGEN_DEVICE_FUNC Float ConstructFromSignAndBits(
      typename Base::BitsType sign, typename Base::BitsType bits) {
    return Eigen::numext::bit_cast<Float>(
        static_cast<typename Base::BitsType>(bits | sign));
  }
};

template <>
struct Traits<float8_e4m3b11> : public TraitsBase<float8_e4m3b11> {
  using Base = TraitsBase<float8_e4m3b11>;
  static constexpr int kExponentBias = 11;
  static EIGEN_DEVICE_FUNC float8_e4m3b11 ConstructFromSignAndBits(
      typename Base::BitsType sign, typename Base::BitsType bits) {
    // float8_e4m3b11 does not support signed zero, ignore the sign if we try to
    // make one.
    if (bits == 0) {
      sign = 0;
    }
    return Eigen::numext::bit_cast<float8_e4m3b11>(
        static_cast<typename Base::BitsType>(bits | sign));
  }
};

// Shift bits in the appropriate directions and add the exponent offset
// to convert between bit representations.  The input `in` must be a
// positive normalized value.
template <typename From, typename To,
          typename FromBits = typename Traits<From>::BitsType,
          typename ToBits = typename Traits<To>::BitsType>
constexpr FromBits ToFromBits(ToBits in) {
  using FromTraits = Traits<From>;
  constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  constexpr int kFromExponentBias = FromTraits::kExponentBias;

  using ToTraits = Traits<To>;
  constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  constexpr int kToExponentBias = ToTraits::kExponentBias;

  constexpr int kExponentOffset = kFromExponentBias - kToExponentBias;
  constexpr int kDigitShift = kFromMantissaBits - kToMantissaBits;

  FromBits out = static_cast<FromBits>(in);
  if constexpr (kDigitShift > 0) {
    out <<= kDigitShift;
  } else if constexpr (kDigitShift < 0) {
    out >>= -kDigitShift;
  }
  out += static_cast<FromBits>(kExponentOffset) << kFromMantissaBits;
  return out;
}

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
  using FromBits = typename GetUnsignedInteger<sizeof(From)>::type;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;

  using ToTraits = Traits<To>;
  using ToBits = typename GetUnsignedInteger<sizeof(To)>::type;
  static constexpr int kToBits = ToTraits::kBits;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;

  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;
  static constexpr int kSignShift = kToBits - kFromBits;

  static EIGEN_DEVICE_FUNC inline To run(const From& from) {
    // Shift bits to destination type, without sign bit.
    FromBits from_bits = Eigen::numext::bit_cast<FromBits>(from);
    const FromBits from_sign =
        from_bits ^ Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));
    ToBits sign;
    if constexpr (kSignShift >= 0) {
      sign = ToBits{from_sign} << kSignShift;
    } else if constexpr (kSignShift < 0) {
      sign = static_cast<ToBits>(from_sign >> -kSignShift);
    }
    from_bits ^= from_sign;  // Zeros sign bit to obtain absolute value.

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return sign != 0 ? -Eigen::NumTraits<To>::infinity()
                       : Eigen::NumTraits<To>::infinity();
    }
    if (Eigen::numext::isnan(from)) {
      return sign != 0 ? -Eigen::NumTraits<To>::quiet_NaN()
                       : Eigen::NumTraits<To>::quiet_NaN();
    }
    if (from_bits == 0) {
      return ToTraits::ConstructFromSignAndBits(/*sign=*/sign, /*bits=*/0);
    }

    // Adjust mantissa.
    FromBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
      if constexpr (!kTruncate) {
        rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
      }
      // Zero-out tail bits.
      rounded_from_bits &= ~((FromBits{1} << (-kDigitShift)) - 1);
    }

    if constexpr (kExponentOffset > 0) {
      if ((from.rep() & kFromExponentMask) == 0) {
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
        return ToTraits::ConstructFromSignAndBits(/*sign=*/sign, /*bits=*/bits);
      }
    } else if constexpr (kExponentOffset < 0) {
      // Check for overflows.

      // Shift up exponent and mantissa, add offset to adjust exponent to
      // source type.
      constexpr ToBits kToHighest = Eigen::NumTraits<To>::highest().rep();
      constexpr FromBits kHighest = ToFromBits<From, To>(kToHighest);

      if (rounded_from_bits > kHighest) {
        ToBits bits =
            kSaturate ? kToHighest : Eigen::NumTraits<To>::infinity().rep();
        return ToTraits::ConstructFromSignAndBits(/*sign=*/sign, /*bits=*/bits);
      }

      // Subnormals and zero.
      constexpr FromBits kLowestNormal =
          ToFromBits<From, To>(std::numeric_limits<To>::min().rep());
      if (rounded_from_bits < kLowestNormal) {
        // Round and shift mantissa down.
        int exponent = ((from_bits >> kFromMantissaBits) - kFromExponentBias);
        int exponent_shift = -kDigitShift - exponent - kToExponentBias + 1;

        // Insert the implicit leading 1 bit on the mantissa.  This assumes
        // the input is normalized.  If it is not, then the mantissa bits -
        // including the implicit one - will be shifted to zero.
        // NOTE: we need to round again from the original from_bits, otherwise
        // the lower precision bits may already be lost.  There is an edge-case
        // where rounding to a normalized value would normally round down,
        // but for a subnormal, we need to round up.
        rounded_from_bits = ((from_bits & FromTraits::kMantissaMask) |
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
        return ToTraits::ConstructFromSignAndBits(/*sign=*/sign, /*bits=*/bits);
      }
    }

    // Shift bits.
    ToBits bits;
    if constexpr (kDigitShift < 0) {
      bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
    } else if constexpr (kDigitShift >= 0) {
      bits = ToBits{rounded_from_bits} << kDigitShift;
    }
    // Increase exponent by offset difference.
    bits += static_cast<ToBits>(kExponentOffset) << kToMantissaBits;

    // Insert sign bit.
    return ToTraits::ConstructFromSignAndBits(/*sign=*/sign, /*bits=*/bits);
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
        if ((from_bits & 0x7FFF) > static_cast<uint16_t>(kHighest.rep()) << 8) {
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

// Eigen-specific overrides.
namespace Eigen {
namespace numext {

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tsl::float8_e4m3fn
bit_cast<tsl::float8_e4m3fn, uint8_t>(const uint8_t& src) {
  return tsl::float8_e4m3fn::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tsl::float8_e4m3fn>(const tsl::float8_e4m3fn& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tsl::float8_e5m2
bit_cast<tsl::float8_e5m2, uint8_t>(const uint8_t& src) {
  return tsl::float8_e5m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tsl::float8_e5m2>(const tsl::float8_e5m2& src) {
  return src.rep();
}

}  // namespace numext

// Work-around for isinf/isnan/isfinite issue on aarch64.
namespace internal {
template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tsl::float8_e4m3fn>(
    const tsl::float8_e4m3fn& x) {
  return tsl::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tsl::float8_e4m3b11>(
    const tsl::float8_e4m3b11& x) {
  return tsl::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tsl::float8_e5m2>(
    const tsl::float8_e5m2& x) {
  return tsl::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tsl::float8_e4m3fn>(
    const tsl::float8_e4m3fn& x) {
  return tsl::float8_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tsl::float8_e4m3b11>(
    const tsl::float8_e4m3b11& x) {
  return tsl::float8_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tsl::float8_e5m2>(
    const tsl::float8_e5m2& x) {
  return tsl::float8_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isfinite_impl<tsl::float8_e4m3fn>(
    const tsl::float8_e4m3fn& x) {
  return tsl::float8_internal::isfinite(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isfinite_impl<tsl::float8_e4m3b11>(
    const tsl::float8_e4m3b11& x) {
  return tsl::float8_internal::isfinite(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isfinite_impl<tsl::float8_e5m2>(
    const tsl::float8_e5m2& x) {
  return tsl::float8_internal::isfinite(x);
}

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
