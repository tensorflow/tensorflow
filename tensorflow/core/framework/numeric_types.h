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

  explicit EIGEN_DEVICE_FUNC bfloat16(float v) {
    uint32_t input;
    memcpy(&input, &v, sizeof(uint32_t));

    if ((~input & 0x7f800000) == 0 && (input & 0x007fffff) != 0) {
      // If the value is a NaN, squash it to a qNaN with msb of fraction set,
      // this makes sure after truncation we don't end up with an inf.
      //
      // qNaN magic: All exponent bits set + most significant bit of fraction
      // set.
      value = 0x7fc0;
    } else {
      // Fast rounding algorithm that rounds a half value to nearest even. This
      // reduces expected error when we convert a large number of floats. Here
      // is how it works:
      //
      // Definitions:
      // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
      // with the following tags:
      //
      // Sign |  Exp (8 bits) | Frac (23 bits)
      //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
      //
      //  S: Sign bit.
      //  E: Exponent bits.
      //  F: First 6 bits of fraction.
      //  L: Least significant bit of resulting bfloat16 if we truncate away the
      //  rest of the float32. This is also the 7th bit of fraction
      //  R: Rounding bit, 8th bit of fraction.
      //  T: Sticky bits, rest of fraction, 15 bits.
      //
      // To round half to nearest even, there are 3 cases where we want to round
      // down (simply truncate the result of the bits away, which consists of
      // rounding bit and sticky bits) and two cases where we want to round up
      // (truncate then add one to the result).
      //
      // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
      // 1s) as the rounding bias, adds the rounding bias to the input, then
      // truncates the last 16 bits away.
      //
      // To understand how it works, we can analyze this algorithm case by case:
      //
      // 1. L = 0, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input may create any carry, depending on
      //   whether there is any value set to 1 in T bits.
      //   - R may be set to 1 if there is a carry.
      //   - L remains 0.
      //   - Note that this case also handles Inf and -Inf, where all fraction
      //   bits, including L, R and Ts are all 0. The output remains Inf after
      //   this algorithm.
      //
      // 2. L = 1, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits but
      //   adds 1 to rounding bit.
      //   - L remains 1.
      //
      // 3. L = 0, R = 1, all of T are 0:
      //   Expect: round down, this is exactly at half, the result is already
      //   even (L=0).
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input sets all sticky bits to 1, but
      //   doesn't create a carry.
      //   - R remains 1.
      //   - L remains 0.
      //
      // 4. L = 1, R = 1:
      //   Expect: round up, this is exactly at half, the result needs to be
      //   round to the next even number.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits, but
      //   creates a carry from rounding bit.
      //   - The carry sets L to 0, creates another carry bit and propagate
      //   forward to F bits.
      //   - If all the F bits are 1, a carry then propagates to the exponent
      //   bits, which then creates the minimum value with the next exponent
      //   value. Note that we won't have the case where exponents are all 1,
      //   since that's either a NaN (handled in the other if condition) or inf
      //   (handled in case 1).
      //
      // 5. L = 0, R = 1, any of T is 1:
      //   Expect: round up, this is greater than half.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input creates a carry from sticky bits,
      //   sets rounding bit to 0, then create another carry.
      //   - The second carry sets L to 1.
      //
      // Examples:
      //
      //  Exact half value that is already even:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0     1000000000000000
      //
      //     This falls into case 3. We truncate the rest of 16 bits and no
      //     carry is created into F and L:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //  Exact half value, round to next even number:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 0 1     1000000000000000
      //
      //     This falls into case 4. We create a carry from R and T,
      //     which then propagates into L and F:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //
      //  Max denormal value round to min normal value:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 1      0 0 0 0 0 0 0
      //
      //  Max normal value round to Inf:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     1 1 1 1 1 1 1 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     1 1 1 1 1 1 1 1      0 0 0 0 0 0 0
      //
      //
      // Least significant bit of resulting bfloat.
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;
      value = static_cast<uint16_t>(input >> 16);
    }
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
