/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/mlir/utils/math_util.h"

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/bit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"

namespace xla {

int64_t GetKnownAlignment(mlir::Value value, int depth) {
  if (depth > 8) {
    return 1;
  }

  // Match integer/index constants.
  llvm::APInt const_val;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&const_val))) {
    if (const_val.isZero()) {
      return kMaxAlignment;
    }
    uint64_t abs_val = const_val.abs().getZExtValue();
    return int64_t{1} << llvm::countr_zero(abs_val);
  }

  // muli: if a is a multiple of 2^A and b is a multiple of 2^B, then a*b is a
  // multiple of 2^(A+B).
  if (auto muli = value.getDefiningOp<mlir::arith::MulIOp>()) {
    int64_t lhs_align = GetKnownAlignment(muli.getLhs(), depth + 1);
    int64_t rhs_align = GetKnownAlignment(muli.getRhs(), depth + 1);
    // Since alignments are powers of 2, their product is also a power of 2.
    // The exponent of the product is the sum of the exponents of the operands.
    // To avoid overflow when calculating (1LL << exponent), the sum of
    // exponents must be less than 63. We also clamp the alignment at 1LL << 62.
    int lhs_exp = llvm::countr_zero(static_cast<uint64_t>(lhs_align));
    int rhs_exp = llvm::countr_zero(static_cast<uint64_t>(rhs_align));
    if (lhs_exp + rhs_exp < 63) {
      return lhs_align * rhs_align;
    }
    return kMaxAlignment;
  }

  // addi: gcd of the two operand alignments.
  if (auto addi = value.getDefiningOp<mlir::arith::AddIOp>()) {
    return std::min(GetKnownAlignment(addi.getLhs(), depth + 1),
                    GetKnownAlignment(addi.getRhs(), depth + 1));
  }

  // Index casts preserve alignment.
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    return GetKnownAlignment(cast.getIn(), depth + 1);
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    return GetKnownAlignment(cast.getIn(), depth + 1);
  }
  return 1;
}

}  // namespace xla
