/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TYPES_H_
#define XLA_TYPES_H_

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // IWYU pragma: export
#include "tsl/platform/ml_dtypes.h"  // IWYU pragma: export

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
struct is_specialized_floating_point
    : std::bool_constant<std::numeric_limits<T>::is_specialized &&
                         !std::numeric_limits<T>::is_integer> {};

template <typename T>
inline constexpr bool is_specialized_floating_point_v =
    is_specialized_floating_point<T>::value;

template <typename T>
struct is_specialized_integral
    : std::bool_constant<std::numeric_limits<T>::is_specialized &&
                         std::numeric_limits<T>::is_integer> {};

template <typename T>
inline constexpr bool is_specialized_integral_v =
    is_specialized_integral<T>::value;

using u1 = tsl::uint1;
using s1 = tsl::int1;
using u2 = tsl::uint2;
using s2 = tsl::int2;
using u4 = tsl::uint4;
using s4 = tsl::int4;

template <class T>
struct is_intN : std::false_type {};
template <int kN, typename UnderlyingType>
struct is_intN<::ml_dtypes::intN<kN, UnderlyingType>> : std::true_type {};

template <typename T>
inline constexpr bool is_intN_v = is_intN<T>::value;

}  // namespace xla

// Extend ml_dtypes to allow absl::String functions.
namespace ml_dtypes {

template <typename Sink, typename T,
          std::enable_if_t<xla::is_intN_v<T>, int> = 0>
void AbslStringify(Sink& sink, const T& i) {
  static_assert(xla::is_specialized_integral_v<T>);
  if constexpr (std::numeric_limits<T>::is_signed) {
    sink.Append(absl::StrCat(static_cast<int32_t>(i)));
  } else {
    sink.Append(absl::StrCat(static_cast<uint32_t>(i)));
  }
}
}  // namespace ml_dtypes

// Alias namespace ::stream_executor as ::xla::se.
namespace stream_executor {}
namespace xla {
namespace se = ::stream_executor;  // NOLINT(misc-unused-alias-decls)

// std::make_signed_t is “behavior undefined” for custom types, so provide a
// general util to make signed/unsigned for both primitive and custom types.
template <typename T, typename = void>
struct make_specialized_unsigned {
  using type = std::make_unsigned_t<T>;
};

template <typename T>
struct make_specialized_unsigned<T, typename std::enable_if_t<is_intN_v<T>>> {
  static_assert(std::is_integral_v<typename T::underlying_type>);
  using type =
      ::ml_dtypes::intN<T::bits,
                        std::make_unsigned_t<typename T::underlying_type>>;
};

template <typename T>
using make_specialized_unsigned_t = typename make_specialized_unsigned<T>::type;

template <typename T, typename = void>
struct make_specialized_signed {
  using type = std::make_signed_t<T>;
};

template <typename T>
struct make_specialized_signed<T, typename std::enable_if_t<is_intN_v<T>>> {
  static_assert(std::is_integral_v<typename T::underlying_type>);
  using type =
      ::ml_dtypes::intN<T::bits,
                        std::make_signed_t<typename T::underlying_type>>;
};

template <typename T>
using make_specialized_signed_t = typename make_specialized_signed<T>::type;

// has_negative_zero[_v]

template <typename T>
struct has_negative_zero
    : std::bool_constant<std::numeric_limits<T>::is_iec559> {};

template <>
struct has_negative_zero<tsl::float4_e2m1fn> : std::bool_constant<true> {};

template <>
struct has_negative_zero<tsl::float8_e4m3fn> : std::bool_constant<true> {};

template <typename T>
inline constexpr bool has_negative_zero_v = has_negative_zero<T>::value;

// has_zero[_v]

template <typename T>
struct has_zero : std::bool_constant<true> {};

template <>
struct has_zero<tsl::float8_e8m0fnu> : std::bool_constant<false> {};

template <typename T>
inline constexpr bool has_zero_v = has_zero<T>::value;

}  // namespace xla

#endif  // XLA_TYPES_H_
