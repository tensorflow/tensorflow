/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_FRAMEWORK_TYPE_TRAITS_H_
#define XLA_TSL_FRAMEWORK_TYPE_TRAITS_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "xla/tsl/framework/numeric_types.h"
#include "tsl/platform/types.h"

namespace tsl {

// Functions to define quantization attribute of types.
struct true_type {
  static constexpr bool value = true;
};
struct false_type {
  static constexpr bool value = false;
};

// Default is_quantized is false.
template <typename T>
struct is_quantized : false_type {};

// Specialize the quantized types.
template <>
struct is_quantized<qint8> : true_type {};
template <>
struct is_quantized<quint8> : true_type {};
template <>
struct is_quantized<qint32> : true_type {};
template <>
struct is_quantized<qint16> : true_type {};
template <>
struct is_quantized<quint16> : true_type {};

// Default is_complex is false.
template <typename T>
struct is_complex : false_type {};

// Specialize std::complex<float> and std::complex<double> types.
template <>
struct is_complex<std::complex<float>> : true_type {};
template <>
struct is_complex<std::complex<double>> : true_type {};

// is_simple_type<T>::value if T[] can be safely constructed and destructed
// without running T() and ~T().  We do not use std::is_trivial<T>
// directly because std::complex<float> and std::complex<double> are
// not trivial, but their arrays can be constructed and destructed
// without running their default ctors and dtors.
template <typename T>
struct is_simple_type {
  static constexpr bool value =
      std::is_trivial<T>::value || std::is_same<T, Eigen::half>::value ||
      std::is_same<T, complex64>::value || std::is_same<T, complex128>::value ||
      is_quantized<T>::value || std::is_same<T, bfloat16>::value ||
      std::is_same<T, float8_e3m4>::value ||
      std::is_same<T, float8_e4m3>::value ||
      std::is_same<T, float8_e4m3fn>::value ||
      std::is_same<T, float8_e4m3fnuz>::value ||
      std::is_same<T, float8_e4m3b11fnuz>::value ||
      std::is_same<T, float8_e5m2>::value ||
      std::is_same<T, float8_e5m2fnuz>::value || std::is_same<T, int4>::value ||
      std::is_same<T, uint4>::value;
};

}  // namespace tsl

// Define numeric limits for our quantized as subclasses of the
// standard types.
namespace std {
template <>
class numeric_limits<tsl::qint8> : public numeric_limits<tsl::int8> {};
template <>
class numeric_limits<tsl::quint8> : public numeric_limits<tsl::uint8> {};
template <>
class numeric_limits<tsl::qint16> : public numeric_limits<tsl::int16> {};
template <>
class numeric_limits<tsl::quint16> : public numeric_limits<tsl::uint16> {};
template <>
class numeric_limits<tsl::qint32> : public numeric_limits<tsl::int32> {};

// Specialize is_signed for quantized types.
template <>
struct is_signed<tsl::qint8> : public is_signed<tsl::int8> {};
template <>
struct is_signed<tsl::quint8> : public is_signed<tsl::uint8> {};
template <>
struct is_signed<tsl::qint16> : public is_signed<tsl::int16> {};
template <>
struct is_signed<tsl::quint16> : public is_signed<tsl::uint16> {};
template <>
struct is_signed<tsl::qint32> : public is_signed<tsl::int32> {};

}  // namespace std

#endif  // XLA_TSL_FRAMEWORK_TYPE_TRAITS_H_
