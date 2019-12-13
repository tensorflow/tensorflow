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

#include "mlir/IR/Builders.h"

namespace mlir {
class BlockArgument;
namespace edsc {

inline void defaultRegionBuilder(ArrayRef<BlockArgument *> args) {}

/// EDSC entry point to build linalg.generic operations programmatically.
Operation *makeLinalgGenericOp(
    ArrayRef<AffineExpr> indices, ArrayRef<ArrayRef<AffineExpr>> mapExpressions,
    ArrayRef<Value *> inputViews, ArrayRef<Value *> outputViews,
    ArrayRef<StringRef> iteratorTypes,
    decltype(defaultRegionBuilder) regionBuilder = defaultRegionBuilder);

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_BUILDERS_H_
