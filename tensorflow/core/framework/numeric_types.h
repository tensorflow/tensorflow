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

}  // namespace tensorflow

namespace Eigen {
// TODO(xpan): We probably need to overwrite more methods to have correct eigen
// behavior. E.g. loest(), is_integer, etc. See NumTraits.h in eigen.
template <>
struct NumTraits<tensorflow::bfloat16>
    : GenericNumTraits<tensorflow::bfloat16> {};

using ::tensorflow::operator==;
using ::tensorflow::operator!=;

namespace numext {

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 log(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::logf(static_cast<float>(x)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 exp(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::expf(static_cast<float>(x)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 abs(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::fabsf(static_cast<float>(x)));
}

}  // namespace numext
}  // namespace Eigen

#if defined(COMPILER_MSVC) && !defined(__clang__)
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
