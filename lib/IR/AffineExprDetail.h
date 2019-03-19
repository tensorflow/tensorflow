//===- AffineExprDetail.h - MLIR Affine Expr storage details ----*- C++ -*-===//
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
// This holds implementation details of AffineExpr. Ideally it would not be
// exposed and would be kept local to AffineExpr.cpp however, MLIRContext.cpp
// needs to know the sizes for placement-new style Allocation.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_IR_AFFINEEXPRDETAIL_H_
#define MLIR_IR_AFFINEEXPRDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Base storage class appearing in an affine expression.
struct AffineExprStorage {
  AffineExprStorage(AffineExprKind kind, MLIRContext *context)
      : contextAndKind(context, kind) {}
  llvm::PointerIntPair<MLIRContext *, 3, AffineExprKind> contextAndKind;
};

/// A binary operation appearing in an affine expression.
struct AffineBinaryOpExprStorage : public AffineExprStorage {
  AffineBinaryOpExprStorage(AffineExprStorage base, AffineExpr lhs,
                            AffineExpr rhs)
      : AffineExprStorage(base), lhs(lhs), rhs(rhs) {}
  static AffineExpr get(AffineExprKind kind, AffineExpr lhs, AffineExpr rhs);
  AffineExpr lhs;
  AffineExpr rhs;
};

/// A dimensional identifier appearing in an affine expression.
struct AffineDimExprStorage : public AffineExprStorage {
  AffineDimExprStorage(AffineExprStorage base, unsigned position)
      : AffineExprStorage(base), position(position) {}
  /// Position of this identifier in the argument list.
  unsigned position;
};

/// A symbolic identifier appearing in an affine expression.
struct AffineSymbolExprStorage : public AffineExprStorage {
  AffineSymbolExprStorage(AffineExprStorage base, unsigned position)
      : AffineExprStorage(base), position(position) {}
  /// Position of this identifier in the symbol list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
struct AffineConstantExprStorage : public AffineExprStorage {
  AffineConstantExprStorage(AffineExprStorage base, int64_t constant)
      : AffineExprStorage(base), constant(constant) {}
  // The constant.
  int64_t constant;
};

} // end namespace detail
} // end namespace mlir
#endif // MLIR_IR_AFFINEEXPRDETAIL_H_
