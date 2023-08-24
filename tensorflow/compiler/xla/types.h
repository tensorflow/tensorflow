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

#ifndef TENSORFLOW_COMPILER_XLA_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_TYPES_H_

#include <complex>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/strings/str_format.h"
#include "third_party/eigen3/Eigen/Core"

namespace xla {

using ::Eigen::bfloat16;  // NOLINT(misc-unused-using-decls)
using ::Eigen::half;      // NOLINT(misc-unused-using-decls)

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
constexpr bool is_specialized_floating_point_v =
    std::numeric_limits<T>::is_specialized &&
    !std::numeric_limits<T>::is_integer;

// LINT.IfChange
template <typename UnderlyingTy>
struct i4 {
 private:
  UnderlyingTy v : 4;

 public:
  constexpr i4() : v(0) {}
  constexpr explicit i4(UnderlyingTy val) : v(val & 0x0F) {}
  template <typename T>
  constexpr explicit i4(T t) : i4(static_cast<UnderlyingTy>(t)) {}
  constexpr i4(const i4& other) = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator UnderlyingTy() const {
    return static_cast<UnderlyingTy>(v);
  }

  template <typename T>
  i4 operator>>(const T amount) const {
    return i4(v >> amount);
  }
  template <typename T>
  i4 operator<<(const T amount) const {
    return i4(v << amount);
  }

  constexpr bool operator==(const i4 other) const { return v == other.v; }
  constexpr bool operator!=(const i4 other) const { return v != other.v; }
  constexpr bool operator<(const i4 other) const { return v < other.v; }
  constexpr bool operator>(const i4 other) const { return v > other.v; }
  constexpr bool operator<=(const i4 other) const { return v <= other.v; }
  constexpr bool operator>=(const i4 other) const { return v >= other.v; }

  constexpr i4 operator-() const { return i4(-v); }
  constexpr i4 operator~() const { return i4(~v); }
  constexpr i4 operator++(int) {
    i4 tmp(*this);
    v = (v + 1) & 0x0F;
    return tmp;
  }
  constexpr i4& operator++() {
    v = (v + 1) & 0x0F;
    return *this;
  }
  constexpr i4& operator+=(const i4 other) {
    v = (v + other.v) & 0x0F;
    return *this;
  }

  friend ::std::ostream& operator<<(::std::ostream& os, const i4& num) {
    os << static_cast<int16_t>(num.v);
    return os;
  }

  friend ::std::istream& operator>>(::std::istream& is, i4& num) {
    UnderlyingTy value;
    is >> value;
    num = i4(static_cast<UnderlyingTy>(value));
    return is;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const i4& i) {
    absl::Format(&sink, "%d", i.v);
  }

  std::string to_string() const { return std::to_string(v); }
};

using u4 = i4<uint8_t>;
using s4 = i4<int8_t>;
// LINT.ThenChange(//tensorflow/compiler/xla/literal.cc)

}  // namespace xla

// Alias namespace ::stream_executor as ::xla::se.
namespace stream_executor {}
namespace xla {
namespace se = ::stream_executor;  // NOLINT(misc-unused-alias-decls)
}  // namespace xla

namespace std {
// NOLINTBEGIN: these names must match std::numeric_limits.
template <typename Int4T>
class numeric_limits_int4t {
 public:
  static constexpr bool is_specialized = true;
  static constexpr const bool is_integer = true;
  static constexpr const bool is_exact = true;

  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr float_denorm_style has_denorm = denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_toward_zero;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr int max_digits10 = 0;
  static constexpr int radix = 2;
  static constexpr int min_exponent = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent = 0;
  static constexpr int max_exponent10 = 0;
  static constexpr bool tinyness_before = false;

  static constexpr Int4T epsilon() { return Int4T(0); }
  static constexpr Int4T round_error() { return Int4T(0); }
  static constexpr Int4T infinity() { return Int4T(0); }
  static constexpr Int4T quiet_NaN() { return Int4T(0); }
  static constexpr Int4T signaling_NaN() { return Int4T(0); }
  static constexpr Int4T denorm_min() { return Int4T(0); }
};

template <>
class numeric_limits<xla::u4> : public numeric_limits_int4t<xla::u4> {
 public:
  static constexpr const bool is_signed = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 2;
  static constexpr bool is_modulo = true;
  static constexpr bool traps = numeric_limits<uint8_t>::traps;

  static constexpr xla::u4(min)() { return xla::u4(0); }
  static constexpr xla::u4 lowest() { return xla::u4(0); }
  static constexpr xla::u4(max)() { return xla::u4(15); }
};

template <>
class numeric_limits<xla::s4> : public numeric_limits_int4t<xla::s4> {
 public:
  static constexpr const bool is_signed = true;
  static constexpr int digits = 3;
  static constexpr int digits10 = 1;
  static constexpr bool is_modulo = false;
  static constexpr bool traps = numeric_limits<int8_t>::traps;

  static constexpr xla::s4(min)() { return xla::s4(-8); }
  static constexpr xla::s4 lowest() { return xla::s4(-8); }
  static constexpr xla::s4(max)() { return xla::s4(7); }
};
// NOLINTEND
}  // namespace std

namespace xla {

// std::make_signed_t is “behavior undefined” for custom types, so provide a
// general util to make signed/unsigned for both primitive and custom types.
template <typename T>
struct make_specialized_unsigned {
  using type = std::make_unsigned_t<T>;
};

template <typename UnderlyingTy>
struct make_specialized_unsigned<xla::i4<UnderlyingTy>> {
  using type = xla::i4<std::make_unsigned_t<UnderlyingTy>>;
};

template <typename T>
using make_specialized_unsigned_t = typename make_specialized_unsigned<T>::type;

template <typename T>
struct make_specialized_signed {
  using type = std::make_signed_t<T>;
};

template <typename UnderlyingTy>
struct make_specialized_signed<xla::i4<UnderlyingTy>> {
  using type = xla::i4<std::make_signed_t<UnderlyingTy>>;
};

template <typename T>
using make_specialized_signed_t = typename make_specialized_signed<T>::type;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TYPES_H_
