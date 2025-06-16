/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_SAFE_CAST_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_SAFE_CAST_H_

#include <cmath>
#include <limits>

namespace tools {

// Converts a floating-point number to an integer. For all inputs x where
// static_cast<IntOut>(x) is legal according to the C++ standard, the result
// is identical to that cast (i.e. the result is x with its fractional part
// truncated whenever that is representable as IntOut).
//
// static_cast would cause undefined behavior for the following cases, which
// have well-defined behavior for this function:
//
//  1. If x is NaN, the result is zero.
//
//  2. If the truncated form of x is above the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::max().
//
//  3. If the truncated form of x is below the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::min().
//
// Note that cases #2 and #3 cover infinities as well as finite numbers.
//
// The range of FloatIn must include the range of IntOut, otherwise
// the results are undefined.
// TODO(sfeuz): Replace by absl::SafeCast once available.
template <class IntOut, class FloatIn>
IntOut SafeCast(FloatIn x) {
  static_assert(!std::numeric_limits<FloatIn>::is_integer,
                "FloatIn is integer");
  static_assert(std::numeric_limits<IntOut>::is_integer,
                "IntOut is not integer");
  static_assert(std::numeric_limits<IntOut>::radix == 2, "IntOut is base 2");

  // Special case NaN, for which the logic below doesn't work.
  if (std::isnan(x)) {
    return 0;
  }

  // Negative values all clip to zero for unsigned results.
  if (!std::numeric_limits<IntOut>::is_signed && x < 0) {
    return 0;
  }

  // Handle infinities.
  if (std::isinf(x)) {
    return x < 0 ? std::numeric_limits<IntOut>::min()
                 : std::numeric_limits<IntOut>::max();
  }

  // Set exp such that x == f * 2^exp for some f with |f| in [0.5, 1.0),
  // unless x is zero in which case exp == 0. Note that this implies that the
  // magnitude of x is strictly less than 2^exp.
  int exp = 0;
  std::frexp(x, &exp);

  // Let N be the number of non-sign bits in the representation of IntOut. If
  // the magnitude of x is strictly less than 2^N, the truncated version of x
  // is representable as IntOut. The only representable integer for which this
  // is not the case is kMin for signed types (i.e. -2^N), but that is covered
  // by the fall-through below.
  if (exp <= std::numeric_limits<IntOut>::digits) {
    return x;
  }

  // Handle numbers with magnitude >= 2^N.
  return x < 0 ? std::numeric_limits<IntOut>::min()
               : std::numeric_limits<IntOut>::max();
}
}  // namespace tools

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_SAFE_CAST_H_
