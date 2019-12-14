//===- Builders.h - MLIR Declarative Linalg Builders ------------*- C++ -*-===//
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
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_
#define MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"

namespace mlir {
class BlockArgument;
namespace edsc {

enum class IterType { Parallel, Reduction };

inline StringRef toString(IterType t) {
  switch (t) {
  case IterType::Parallel:
    return getParallelIteratorTypeName();
  case IterType::Reduction:
    return getParallelIteratorTypeName();
  default:
    llvm_unreachable("Unsupport IterType");
  }
}

/// A StructuredIndexed represents a captured value that can be indexed and
/// passed to the `makeLinalgGenericOp`. It allows writing intuitive index
/// expressions such as:
///
/// ```
///      StructuredIndexed A(vA), B(vB), C(vC);
///      makeLinalgGenericOp({A({m, n}), B({k, n})}, {C({m, n})}, ... );
/// ```
struct StructuredIndexed {
  StructuredIndexed(Value *v) : value(v) {}
  StructuredIndexed operator()(ArrayRef<AffineExpr> indexings) {
    return StructuredIndexed(value, indexings);
  }

  operator Value *() const /* implicit */ { return value; }
  ArrayRef<AffineExpr> getExprs() { return exprs; }

private:
  StructuredIndexed(Value *v, ArrayRef<AffineExpr> indexings)
      : value(v), exprs(indexings.begin(), indexings.end()) {
    assert(v->getType().isa<MemRefType>() && "MemRefType expected");
  }
  StructuredIndexed(ValueHandle v, ArrayRef<AffineExpr> indexings)
      : StructuredIndexed(v.getValue(), indexings) {}

  Value *value;
  SmallVector<AffineExpr, 4> exprs;
};

inline void defaultRegionBuilder(ArrayRef<BlockArgument *> args) {}

Operation *makeLinalgGenericOp(
    ArrayRef<IterType> iteratorTypes, ArrayRef<StructuredIndexed> inputs,
    ArrayRef<StructuredIndexed> outputs,
    decltype(defaultRegionBuilder) regionBuilder = defaultRegionBuilder,
    ArrayRef<Value *> otherValues = {},
    ArrayRef<Attribute> otherAttributes = {});

//===----------------------------------------------------------------------===//
// EDSC builders for linalg generic operations.
//===----------------------------------------------------------------------===//

/// TODO(ntv): In the future we should tie these implementations to something in
/// Tablegen that generates the proper interfaces and the proper sugared named
/// ops.

/// Build a linalg.generic that represents C = A * B in the current
/// ScopedContext.
Operation *linalg_matmul(ValueHandle vA, ValueHandle vB, ValueHandle vC);

template <typename Container> Operation *linalg_matmul(Container values) {
  assert(values.size() == 3 && "Expected exactly 3 values");
  return linalg_matmul(values[0], values[1], values[2]);
}

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_
