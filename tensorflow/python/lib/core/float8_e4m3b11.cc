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

#include "tensorflow/python/lib/core/float8_e4m3b11.h"

#include <stdio.h>

namespace tensorflow {

uint8_t float_to_float8_e4m3b11(float v) {
  static_assert(sizeof(float) == sizeof(uint32_t), "Invalid");
  uint32_t tmp = *reinterpret_cast<uint32_t*>(&v);

  uint32_t sign = (tmp & 0x80000000) >> 24;
  uint32_t exponent = (tmp >> 23) & 0xff;
  uint32_t mantissa = tmp & 0x7fffff;
  // subnormals
  if (exponent < 127 - 10) {
    if (exponent < 127 - 14) {
      return 0x00;
    }
    uint32_t shifted_mantissa =
        (0x800000 | mantissa) >> (10 - ((exponent - 127)));
    if (shifted_mantissa == 0) return 0x00;
    return sign | shifted_mantissa;
  }
  if (exponent > 127 + 4) {
    if (exponent == 255 && mantissa != 0) {
      return 0x80;  // nan.
    }
    return 0x7f | sign;
  }
  exponent = exponent - (127 - 11);
  uint8_t result = sign | (exponent << 3) | (mantissa >> 20);
  if (result == 0x80) {
    result = 0;
  }
  return result;
}

static uint32_t clz_uint32(uint32_t x) {
#ifdef __GNUC__
  return __builtin_clz(x);
#else
  uint32_t out = 32;
  while (x != 0) {
    x = x >> 1;
    out -= 1;
  }
  return out;
#endif
}

float float8_e4m3b11_to_float(uint8_t v) {
  if (v == 0x80) {
    return NAN;
  }
  if (v == 0) {
    return 0;
  }
  uint32_t sign = (0x80 & v) << 24;
  uint32_t exponent = (((v & 0x78) >> 3) + (127 - 11));
  uint32_t mantissa = (v & 0x7) << 20;
  // subnormals
  if ((v & 0x78) == 0) {
    uint32_t nzeros = clz_uint32(v & 0x7);
    mantissa = ((v & 0x7) << (nzeros - 29 + 21)) & (0x3 << 21);
    uint32_t tmp = sign | ((0x72 - nzeros + 31) << 23) | mantissa;
    return *reinterpret_cast<float*>(&tmp);
  }
  uint32_t tmp = sign | (exponent << 23) | mantissa;
  return *reinterpret_cast<float*>(&tmp);
}

}  // namespace tensorflow
