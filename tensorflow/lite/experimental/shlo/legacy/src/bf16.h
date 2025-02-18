/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_BF16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_BF16_H_

#include "tensorflow/lite/experimental/shlo/legacy/src/has_keyword.h"

#if defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
namespace stablehlo {
using BF16 = bfloat16_t;
}  // namespace stablehlo

#elif __has_keyword(__bf16) && __x86_64__
namespace stablehlo {
// On x86 the compiler is able to generate code for __bf16 operations.
using BF16 = __bf16;
}  // namespace stablehlo

#elif __has_keyword(__bf16) && __aarch64__
#include <cmath>
#include <cstdint>

namespace stablehlo {

// On arm64 the compiler is not yet able to generate code for __bf16
// operations. Therefore, we resort to a software-based implementation of BF16
// based on promoting ops to float.
class BF16 {
 public:
  BF16(float f = 0.0f) {
    if (std::isnan(f)) {
      // If the value is a NaN, squash it to a NaN with the msb of the
      // mantissa. This avoids that after the truncation below we don't end up
      // with an inf.
      value_ = std::signbit(f) ? 0xFFC0 : 0x7FC0;
    } else {
      // Fast rounding algorithm that rounds a half value to nearest even. This
      // reduces expected error when we convert a large number of floats.
      uint32_t input = *reinterpret_cast<const uint32_t*>(&f);

      // Least significant bit of resulting bfloat.
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;

      value_ = static_cast<uint16_t>(input >> 16u);
    }
  }

  BF16& operator=(BF16 other) {
    value_ = other.value_;
    return *this;
  }

  bool operator==(BF16 other) const { return value_ == other.value_; }
  bool operator!=(BF16 other) const { return !(*this == other); }

  operator float() const {
    uint32_t tmp = value_ << 16;
    return *reinterpret_cast<float*>(&tmp);
  }

  BF16 operator-() const { return BF16(-static_cast<float>(*this)); }

  BF16& operator+=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) + static_cast<float>(other)).value_;
    return *this;
  }

  BF16& operator-=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) - static_cast<float>(other)).value_;
    return *this;
  }

  BF16& operator*=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) * static_cast<float>(other)).value_;
    return *this;
  }

  BF16& operator/=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) / static_cast<float>(other)).value_;
    return *this;
  }

 private:
  uint16_t value_;
};

inline BF16 operator+(BF16 x, BF16 y) {
  x += y;
  return x;
}

inline BF16 operator-(BF16 x, BF16 y) {
  x -= y;
  return x;
}

inline BF16 operator*(BF16 x, BF16 y) {
  x *= y;
  return x;
}

inline BF16 operator/(BF16 x, BF16 y) {
  x /= y;
  return x;
}

}  // namespace stablehlo

#else
#error Type BF16 is not available
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_BF16_H_
