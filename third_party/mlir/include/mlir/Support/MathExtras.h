//===- MathExtras.h - Math functions relevant to MLIR -----------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains math functions relevant to MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_MATHEXTRAS_H_
#define MLIR_SUPPORT_MATHEXTRAS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

namespace mlir {

/// Returns the result of MLIR's ceildiv operation on constants. The RHS is
/// expected to be positive.
inline int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  // C/C++'s integer division rounds towards 0.
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}

/// Returns the result of MLIR's floordiv operation on constants. The RHS is
/// expected to be positive.
inline int64_t floorDiv(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  // C/C++'s integer division rounds towards 0.
  return lhs % rhs < 0 ? lhs / rhs - 1 : lhs / rhs;
}

/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
inline int64_t mod(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

/// Returns the least common multiple of 'a' and 'b'.
inline int64_t lcm(int64_t a, int64_t b) {
  uint64_t x = std::abs(a);
  uint64_t y = std::abs(b);
  int64_t lcm = (x * y) / llvm::GreatestCommonDivisor64(x, y);
  assert((lcm >= a && lcm >= b) && "LCM overflow");
  return lcm;
}
} // end namespace mlir

#endif // MLIR_SUPPORT_MATHEXTRAS_H_
