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
// storage types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SDBMEXPRDETAIL_H
#define MLIR_IR_SDBMEXPRDETAIL_H

#include "mlir/Dialect/SDBM/SDBMExpr.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class SDBMDialect;

namespace detail {

// Base storage class for SDBMExpr.
struct SDBMExprStorage : public StorageUniquer::BaseStorage {
  SDBMExprKind getKind() {
    return static_cast<SDBMExprKind>(BaseStorage::getKind());
  }

  SDBMDialect *dialect;
};

// Storage class for SDBM sum and stripe expressions.
struct SDBMBinaryExprStorage : public SDBMExprStorage {
  using KeyTy = std::pair<SDBMDirectExpr, SDBMConstantExpr>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == lhs && std::get<1>(key) == rhs;
  }

  static SDBMBinaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMBinaryExprStorage>();
    result->lhs = std::get<0>(key);
    result->rhs = std::get<1>(key);
    result->dialect = result->lhs.getDialect();
    return result;
  }

  SDBMDirectExpr lhs;
  SDBMConstantExpr rhs;
};

// Storage class for SDBM difference expressions.
struct SDBMDiffExprStorage : public SDBMExprStorage {
  using KeyTy = std::pair<SDBMDirectExpr, SDBMTermExpr>;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == lhs && std::get<1>(key) == rhs;
  }

  static SDBMDiffExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMDiffExprStorage>();
    result->lhs = std::get<0>(key);
    result->rhs = std::get<1>(key);
    result->dialect = result->lhs.getDialect();
    return result;
  }

  SDBMDirectExpr lhs;
  SDBMTermExpr rhs;
};

// Storage class for SDBM constant expressions.
struct SDBMConstantExprStorage : public SDBMExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static SDBMConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMConstantExprStorage>();
    result->constant = key;
    return result;
  }

  int64_t constant;
};

// Storage class for SDBM dimension and symbol expressions.
struct SDBMTermExprStorage : public SDBMExprStorage {
  using KeyTy = unsigned;

  bool operator==(const KeyTy &key) const { return position == key; }

  static SDBMTermExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMTermExprStorage>();
    result->position = key;
    return result;
  }

  unsigned position;
};

// Storage class for SDBM negation expressions.
struct SDBMNegExprStorage : public SDBMExprStorage {
  using KeyTy = SDBMDirectExpr;

  bool operator==(const KeyTy &key) const { return key == expr; }

  static SDBMNegExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<SDBMNegExprStorage>();
    result->expr = key;
    result->dialect = key.getDialect();
    return result;
  }

  SDBMDirectExpr expr;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_IR_SDBMEXPRDETAIL_H
