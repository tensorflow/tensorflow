/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
#define TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_

#include <string.h>

#include <cstdint>

#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {
namespace random {

// Helper function to convert an 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32_t x) {
  // IEEE754 floats are formatted as follows (MSB first):
  //    sign(1) exponent(8) mantissa(23)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 127  -- an excess 127 representation of a zero exponent
  //    mantissa == 23 random bits
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  // Assumes that endian-ness is same for float and uint32_t.
  float result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0f;
}

// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ToDouble(uint32_t x0, uint32_t x1) {
  // IEEE754 doubles are formatted as follows (MSB first):
  //    sign(1) exponent(11) mantissa(52)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 1023  -- an excess 1023 representation of a zero exponent
  //    mantissa == 52 random bits
  const uint32_t mhi = x0 & 0xfffffu;  // upper 20 bits of mantissa
  const uint32_t mlo = x1;             // lower 32 bits of mantissa
  const uint64_t man = (static_cast<uint64_t>(mhi) << 32) | mlo;  // mantissa
  const uint64_t exp = static_cast<uint64_t>(1023);
  const uint64_t val = (exp << 52) | man;
  // Assumes that endian-ness is same for double and uint64_t.
  double result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0;
}

// Helper function to convert two 32-bit uniform integers to two floats
// under the unit normal distribution.
PHILOX_DEVICE_INLINE
void BoxMullerFloat(uint32_t x0, uint32_t x1, float* f0, float* f1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const float epsilon = 1.0e-7f;
  float u1 = Uint32ToFloat(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const float v1 = 2.0f * M_PI * Uint32ToFloat(x1);
  const float u2 = sqrt(-2.0f * log(u1));
#if !defined(__linux__)
  *f0 = sin(v1);
  *f1 = cos(v1);
#else
  sincosf(v1, f0, f1);
#endif
  *f0 *= u2;
  *f1 *= u2;
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
