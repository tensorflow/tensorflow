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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FLOAT8_H_
#define TENSORFLOW_CORE_FRAMEWORK_FLOAT8_H_

// 8-bit Floating Point Interchange Format, as described by
//   https://arxiv.org/abs/2209.05433

#include <cstdint>

#include "third_party/eigen3/Eigen/Core"

namespace tensorflow {

namespace float8_internal {

// Forward-declarations of classes.
class float8_e4m3;
class float8_e5m2;

template <typename Derived>
class float8_base {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};

  constexpr float8_base(uint8_t rep, ConstructFromRepTag) : rep_{rep} {}

 public:
  constexpr uint8_t rep() const { return rep_; }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep_ ^ 0x80), ConstructFromRepTag{});
  }

  constexpr bool operator==(const Derived& other) const {
    if (Eigen::numext::isnan(derived())) {
      return false;
    }
    return rep() == other.rep();
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
  static EIGEN_DEVICE_FUNC Derived ConvertFrom(const From& from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static EIGEN_DEVICE_FUNC To ConvertTo(const Derived& from);

 private:
  uint8_t rep_;
};

class float8_e4m3 : public float8_base<float8_e4m3> {
  // Exponent: 4, Mantissa: 3, bias: 7.
  // Extended range: no inf, NaN represented by S1111111.
 private:
  using Base = float8_base<float8_e4m3>;
  friend class float8_base<float8_e4m3>;

  constexpr float8_e4m3(uint8_t rep, ConstructFromRepTag)
      : Base(rep, ConstructFromRepTag{}) {}

 public:
  explicit float8_e4m3(double f64) : float8_e4m3(ConvertFrom(f64)) {}
  explicit float8_e4m3(float f32) : float8_e4m3(ConvertFrom(f32)) {}
  explicit float8_e4m3(Eigen::bfloat16 bf16) : float8_e4m3(ConvertFrom(bf16)) {}
  explicit float8_e4m3(Eigen::half f16) : float8_e4m3(ConvertFrom(f16)) {}
  explicit float8_e4m3(const float8_e5m2& f8) : float8_e4m3(ConvertFrom(f8)) {}

  explicit operator double() const { return ConvertTo<double>(*this); }
  explicit operator float() const { return ConvertTo<float>(*this); }
  explicit operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }

  using Base::operator==;
  using Base::operator-;
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
  explicit float8_e5m2(double f64) : float8_e5m2(ConvertFrom(f64)) {}
  explicit float8_e5m2(float f32) : float8_e5m2(ConvertFrom(f32)) {}
  explicit float8_e5m2(Eigen::bfloat16 bf16) : float8_e5m2(ConvertFrom(bf16)) {}
  explicit float8_e5m2(Eigen::half f16) : float8_e5m2(ConvertFrom(f16)) {}
  explicit float8_e5m2(float8_e4m3 f8) : float8_e5m2(ConvertFrom(f8)) {}

  explicit operator double() const { return ConvertTo<double>(*this); }
  explicit operator float() const { return ConvertTo<float>(*this); }
  explicit operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(*this);
  }
  explicit operator Eigen::half() const {
    return ConvertTo<Eigen::half>(*this);
  }

  using Base::operator==;
  using Base::operator-;
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
struct numeric_limits_float8<float8_e4m3> : public numeric_limits_float8_base {
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

  static constexpr float8_e4m3 min() { return float8_e4m3::FromRep(0x08); }
  static constexpr float8_e4m3 lowest() { return float8_e4m3::FromRep(0xFE); }
  static constexpr float8_e4m3 max() { return float8_e4m3::FromRep(0x7E); }
  static constexpr float8_e4m3 epsilon() { return float8_e4m3::FromRep(0x20); }
  static constexpr float8_e4m3 round_error() {
    return float8_e4m3::FromRep(0x30);
  }
  static constexpr float8_e4m3 infinity() {
    return float8_e4m3::FromRep(0x7F);
  }  // NaN.
  static constexpr float8_e4m3 quiet_NaN() {
    return float8_e4m3::FromRep(0x7F);
  }
  static constexpr float8_e4m3 signaling_NaN() {
    return float8_e4m3::FromRep(0x7F);
  }
  static constexpr float8_e4m3 denorm_min() {
    return float8_e4m3::FromRep(0x01);
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
    return float8_e5m2::FromRep(0x7F);
  }
  static constexpr float8_e5m2 signaling_NaN() {
    return float8_e5m2::FromRep(0x7D);
  }
  static constexpr float8_e5m2 denorm_min() {
    return float8_e5m2::FromRep(0x01);
  }
};

// Free-functions for use with ADL and in Eigen.
constexpr inline float8_e4m3 abs(const float8_e4m3& a) {
  return float8_e4m3::FromRep(a.rep() & 0x7F);
}

constexpr inline bool isnan(const float8_e4m3& a) {
  return (a.rep() & 0x7F) == 0x7F;
}

constexpr inline bool isinf(const float8_e4m3& a) {
  return false;  // No inf representation.
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

}  // namespace float8_internal

// Exported types.
using float8_e4m3 = float8_internal::float8_e4m3;
using float8_e5m2 = float8_internal::float8_e5m2;

}  // namespace tensorflow

// Standard-library overrides.  Note that these are picked up by Eigen as well.
namespace std {
template <>
struct numeric_limits<tensorflow::float8_e4m3>
    : public tensorflow::float8_internal::numeric_limits_float8<
          tensorflow::float8_e4m3> {};

template <>
struct numeric_limits<tensorflow::float8_e5m2>
    : public tensorflow::float8_internal::numeric_limits_float8<
          tensorflow::float8_e5m2> {};

}  // namespace std

// Eigen-specific overrides.
namespace Eigen {
namespace numext {

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tensorflow::float8_e4m3
bit_cast<tensorflow::float8_e4m3, uint8_t>(const uint8_t& src) {
  return tensorflow::float8_e4m3::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tensorflow::float8_e4m3>(const tensorflow::float8_e4m3& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC tensorflow::float8_e5m2
bit_cast<tensorflow::float8_e5m2, uint8_t>(const uint8_t& src) {
  return tensorflow::float8_e5m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, tensorflow::float8_e5m2>(const tensorflow::float8_e5m2& src) {
  return src.rep();
}

}  // namespace numext

// Work-around for isinf/isnan issue on aarch64.
namespace internal {
template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tensorflow::float8_e4m3>(
    const tensorflow::float8_e4m3& x) {
  return tensorflow::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<tensorflow::float8_e5m2>(
    const tensorflow::float8_e5m2& x) {
  return tensorflow::float8_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tensorflow::float8_e4m3>(
    const tensorflow::float8_e4m3& x) {
  return tensorflow::float8_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<tensorflow::float8_e5m2>(
    const tensorflow::float8_e5m2& x) {
  return tensorflow::float8_internal::isnan(x);
}

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_FRAMEWORK_FLOAT8_H_
