//===- SDBMExprDetail.h - MLIR SDBM Expression storage details --*- C++ -*-===//
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
// This holds implementation details of SDBMExpr, in particular underlying
// storage types.  MLIRContext.cpp needs to know the storage layout for
// allocation and unique'ing purposes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SDBMEXPRDETAIL_H
#define MLIR_IR_SDBMEXPRDETAIL_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SDBMExpr.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {
namespace detail {

struct SDBMExprStorage {
  SDBMExprStorage(SDBMExprKind kind, MLIRContext *context)
      : contextAndKind(context, kind) {}

  SDBMExprKind getKind() { return contextAndKind.getInt(); }

  MLIRContext *getContext() { return contextAndKind.getPointer(); }

  // This needs to know the layout of MLIRContext so the relevant file is
  // included.
  llvm::PointerIntPair<MLIRContext *, 3, SDBMExprKind> contextAndKind;
};

struct SDBMBinaryExprStorage : public SDBMExprStorage {
  SDBMBinaryExprStorage(SDBMExprKind kind, MLIRContext *context,
                        SDBMVaryingExpr left, SDBMConstantExpr right)
      : SDBMExprStorage(kind, context), lhs(left), rhs(right) {}
  SDBMVaryingExpr lhs;
  SDBMConstantExpr rhs;
};

struct SDBMDiffExprStorage : public SDBMExprStorage {
  SDBMDiffExprStorage(MLIRContext *context, SDBMPositiveExpr left,
                      SDBMPositiveExpr right)
      : SDBMExprStorage(SDBMExprKind::Diff, context), lhs(left), rhs(right) {}
  SDBMPositiveExpr lhs;
  SDBMPositiveExpr rhs;
};

struct SDBMConstantExprStorage : public SDBMExprStorage {
  SDBMConstantExprStorage(MLIRContext *context, int64_t value)
      : SDBMExprStorage(SDBMExprKind::Constant, context), constant(value) {}
  int64_t constant;
};

struct SDBMPositiveExprStorage : public SDBMExprStorage {
  SDBMPositiveExprStorage(SDBMExprKind kind, MLIRContext *context, unsigned pos)
      : SDBMExprStorage(kind, context), position(pos) {}
  unsigned position;
};

struct SDBMNegExprStorage : public SDBMExprStorage {
  SDBMNegExprStorage(SDBMPositiveExpr expr)
      : SDBMExprStorage(SDBMExprKind::Neg, expr.getContext()), dim(expr) {}
  SDBMPositiveExpr dim;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_IR_SDBMEXPRDETAIL_H
