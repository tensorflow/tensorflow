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

#ifndef XLA_TSL_FRAMEWORK_NUMERIC_TYPES_H_
#define XLA_TSL_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

#include "xla/tsl/framework/fixedpoint_types.h"
#include "xla/tsl/platform/types.h"

namespace tsl {

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

}  // namespace tsl

static inline tsl::bfloat16 FloatToBFloat16(float float_val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return *reinterpret_cast<tsl::bfloat16*>(
      reinterpret_cast<uint16_t*>(&float_val));
#else
  return *reinterpret_cast<tsl::bfloat16*>(
      &(reinterpret_cast<uint16_t*>(&float_val)[1]));
#endif
}

namespace Eigen {
template <>
struct NumTraits<tsl::tstring> : GenericNumTraits<tsl::tstring> {
  enum {
    RequireInitialization = 1,
    ReadCost = HugeCost,
    AddCost = HugeCost,
    MulCost = HugeCost
  };

  static constexpr inline int digits10() { return 0; }
  static constexpr inline int max_digits10() { return 0; }

 private:
  static inline tsl::tstring epsilon();
  static inline tsl::tstring dummy_precision();
  static inline tsl::tstring lowest();
  static inline tsl::tstring highest();
  static inline tsl::tstring infinity();
  static inline tsl::tstring quiet_NaN();
};

}  // namespace Eigen

#endif  // XLA_TSL_FRAMEWORK_NUMERIC_TYPES_H_
