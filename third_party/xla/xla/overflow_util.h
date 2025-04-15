/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_OVERFLOW_UTIL_H_
#define XLA_OVERFLOW_UTIL_H_

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"

namespace xla {

// Multiply two non-negative int64_t's, returning the two's complement result
// and a bool which is true when overflow or negative inputs occurs and false
// otherwise.
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::pair<int64_t, bool>
OverflowSafeMultiply(const int64_t x, const int64_t y) {
#if ABSL_HAVE_BUILTIN(__builtin_mul_overflow)
  int64_t result;
  bool bad = __builtin_mul_overflow(x, y, &result);
  bad |= x < 0;
  bad |= y < 0;
  return std::make_pair(result, bad);
#else
  // Multiply in uint64_t rather than int64_t since signed overflow is
  // undefined. Negative values will wrap around to large unsigned values in the
  // casts (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64_t ux = x;
  const uint64_t uy = y;
  const uint64_t uxy = ux * uy;

  // Cast back to signed.
  int64_t result = static_cast<int64_t>(uxy);
  bool bad = result < 0;

  // Check if we overflow uint64_t, using a cheap check if both inputs are small
  if (ABSL_PREDICT_FALSE((ux | uy) >> 32 != 0)) {
    if (x < 0 || y < 0) {
      // Ensure nonnegativity.  Note that negative numbers will appear "large"
      // to the unsigned comparisons above.
      bad = true;
    } else if (ux != 0 && uxy / ux != uy) {
      // Otherwise, detect overflow using a division
      bad = true;
    }
  }
  return std::make_pair(result, bad);
#endif
}

// Computes x + y and returns nullopt if it overflows.
//
// x and y must be signed integers.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline std::optional<T> OverflowSafeAdd(T x, T y) {
  static_assert(std::is_signed<T>::value,
                "Only implemented for signed numbers T.");
  static_assert(std::is_integral<T>::value, "Only implemented for integers T.");
#if ABSL_HAVE_BUILTIN(__builtin_add_overflow)
  T result;
  if (ABSL_PREDICT_FALSE(__builtin_add_overflow(x, y, &result))) {
    return std::nullopt;
  }
  return result;
#else
  // "Signed integer overflow occurs on integer addition iff the operands have
  // the same sign and the sum has a sign opposite to that of the operands."
  // Hacker's Delight 2nd ed, p 28.
  using U = typename std::make_unsigned<T>::type;
  const U ux = x;
  const U uy = y;
  const U usum = ux + uy;
  const T sum = usum;
  if (x >= 0 == y >= 0 && sum >= 0 != x >= 0) {
    return std::nullopt;
  }
  return sum;
#endif
}

}  // namespace xla

#endif  // XLA_OVERFLOW_UTIL_H_
