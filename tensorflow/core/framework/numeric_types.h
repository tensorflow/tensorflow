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

#ifndef TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// We use Eigen's QInt implementations for our quantized int types.
typedef Eigen::QInt8 qint8;
typedef Eigen::QUInt8 quint8;
typedef Eigen::QInt32 qint32;
typedef Eigen::QInt16 qint16;
typedef Eigen::QUInt16 quint16;

// see framework/bfloat16.h for description.
struct bfloat16 {
  EIGEN_DEVICE_FUNC bfloat16() {}

  EIGEN_DEVICE_FUNC explicit bfloat16(const float v) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = p[0];
#else
    value = p[1];
#endif
  }

  template <class T>
  explicit EIGEN_DEVICE_FUNC bfloat16(const T& val)
      : bfloat16(static_cast<float>(val)) {}

  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(float) const {
    float result;

    uint16_t* q = reinterpret_cast<uint16_t*>(&result);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    q[0] = value;
    q[1] = 0;
#else
    q[0] = 0;
    q[1] = value;
#endif
    return result;
  }

  EIGEN_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator Eigen::half() const {
    return static_cast<Eigen::half>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  EIGEN_DEVICE_FUNC explicit operator double() const {
    return static_cast<double>(float(*this));
  }

  uint16_t value;
};

inline bool operator==(const bfloat16 a, const bfloat16 b) {
  return a.value == b.value;
}

inline bool operator!=(const bfloat16 a, const bfloat16 b) {
  return a.value != b.value;
}

}  // end namespace tensorflow

namespace Eigen {
template <>
struct NumTraits<tensorflow::bfloat16> : GenericNumTraits<uint16_t> {};

using ::tensorflow::operator==;
using ::tensorflow::operator!=;
}  // namespace Eigen

#ifdef COMPILER_MSVC
namespace std {
template <>
struct hash<Eigen::half> {
  std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
}  // namespace std
#endif  // COMPILER_MSVC

#endif  // TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
