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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_

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

static inline tensorflow::bfloat16 FloatToBFloat16(float float_val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return *reinterpret_cast<tensorflow::bfloat16*>(
      reinterpret_cast<uint16_t*>(&float_val));
#else
  return *reinterpret_cast<tensorflow::bfloat16*>(
      &(reinterpret_cast<uint16_t*>(&float_val)[1]));
#endif
}

namespace Eigen {
template <>
struct NumTraits<tensorflow::tstring> : GenericNumTraits<tensorflow::tstring> {
  enum {
    RequireInitialization = 1,
    ReadCost = HugeCost,
    AddCost = HugeCost,
    MulCost = HugeCost
  };

  static inline int digits10() { return 0; }

 private:
  static inline tensorflow::tstring epsilon();
  static inline tensorflow::tstring dummy_precision();
  static inline tensorflow::tstring lowest();
  static inline tensorflow::tstring highest();
  static inline tensorflow::tstring infinity();
  static inline tensorflow::tstring quiet_NaN();
};

}  // namespace Eigen

#endif  // TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_
