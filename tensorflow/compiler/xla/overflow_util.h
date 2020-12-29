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

#ifndef TENSORFLOW_COMPILER_XLA_OVERFLOW_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_OVERFLOW_UTIL_H_

#include <type_traits>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Multiply two nonnegative int64's, returning negative for overflow
inline int64 MultiplyWithoutOverflow(const int64 x, const int64 y) {
  // Multiply in uint64 rather than int64 since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64 ux = x;
  const uint64 uy = y;
  const uint64 uxy = ux * uy;

  // Check if we overflow uint64, using a cheap check if both inputs are small
  if (TF_PREDICT_FALSE((ux | uy) >> 32 != 0)) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    CHECK(x >= 0 && y >= 0);

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return -1;
  }

  // Cast back to signed.  Any negative value will signal an error.
  return static_cast<int64>(uxy);
}

// Computes x + y and returns nullopt if it overflows.
//
// x and y must be signed integers.
template <typename T>
inline absl::optional<T> OverflowSafeAdd(T x, T y) {
  static_assert(std::is_signed<T>::value,
                "Only implemented for signed numbers T.");
  static_assert(std::is_integral<T>::value, "Only implemented for integers T.");
  // "Signed integer overflow occurs on integer addition iff the operands have
  // the same sign and the sum has a sign opposite to that of the operands."
  // Hacker's Delight 2nd ed, p 28.
  using U = typename std::make_unsigned<T>::type;
  const U ux = x;
  const U uy = y;
  const U usum = ux + uy;
  const T sum = usum;
  if (x >= 0 == y >= 0 && sum >= 0 != x >= 0) {
    return absl::nullopt;
  }
  return sum;
}

inline bool FitsInIntegralType(int64 x, PrimitiveType ty) {
  switch (ty) {
    case S8:
      return std::numeric_limits<int8>::min() <= x &&
             std::numeric_limits<int8>::max() >= x;
    case S16:
      return std::numeric_limits<int16>::min() <= x &&
             std::numeric_limits<int16>::max() >= x;
    case S32:
      return std::numeric_limits<int32>::min() <= x &&
             std::numeric_limits<int32>::max() >= x;
    case S64:
      return true;
    case U8:
      return 0 <= x && std::numeric_limits<uint8>::max() >= x;
    case U16:
      return 0 <= x && std::numeric_limits<uint16>::max() >= x;
    case U32:
      return 0 <= x && std::numeric_limits<uint32>::max() >= x;
    case U64:
      return 0 <= x;
    default:
      LOG(FATAL) << "Invalid primitive type " << PrimitiveType_Name(ty);
  }
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_OVERFLOW_UTIL_H_
