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
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Base storage class appearing in an affine expression.
struct AffineExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
};

/// A binary operation appearing in an affine expression.
struct AffineBinaryOpExprStorage : public AffineExprStorage {
  using KeyTy = std::pair<AffineExpr, AffineExpr>;

  bool operator==(const KeyTy &key) const {
    return key.first == lhs && key.second == rhs;
  }

  static AffineBinaryOpExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineBinaryOpExprStorage>();
    result->lhs = key.first;
    result->rhs = key.second;
    result->context = result->lhs.getContext();
    return result;
  }

  AffineExpr lhs;
  AffineExpr rhs;
};

/// A dimensional or symbolic identifier appearing in an affine expression.
struct AffineDimExprStorage : public AffineExprStorage {
  using KeyTy = unsigned;

  bool operator==(const KeyTy &key) const { return position == key; }

  static AffineDimExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineDimExprStorage>();
    result->position = key;
    return result;
  }

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
struct AffineConstantExprStorage : public AffineExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static AffineConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineConstantExprStorage>();
    result->constant = key;
    return result;
  }

  // The constant.
  int64_t constant;
};

} // end namespace detail
} // end namespace mlir
#endif // MLIR_IR_AFFINEEXPRDETAIL_H_
