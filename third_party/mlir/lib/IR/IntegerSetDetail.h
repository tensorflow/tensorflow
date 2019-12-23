//===- IntegerSetDetail.h - MLIR IntegerSet storage details -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of IntegerSet.
//
//===----------------------------------------------------------------------===//

#ifndef INTEGERSETDETAIL_H_
#define INTEGERSETDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct IntegerSetStorage {
  unsigned dimCount;
  unsigned symbolCount;

  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;
};

} // end namespace detail
} // end namespace mlir
#endif // INTEGERSETDETAIL_H_
