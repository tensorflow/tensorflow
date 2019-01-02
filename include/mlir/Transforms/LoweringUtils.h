//===- LoweringUtils.h ---- Utilities for Lowering Passes -------*- C++ -*-===//
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
// This file implements miscellaneous utility functions for lowering passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H
#define MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class AffineApplyOp;
class FuncBuilder;
class Value;

/// Expand the `affineMap` applied to `operands` into a sequence of primitive
/// arithmetic instructions that have the same effect.  The list of operands
/// contains the values of dimensions, followed by those of symbols.  Use
/// `builder` to create new instructions.  Report errors at the specificed
/// location `loc`.  Return a list of results, or `None` if any expansion
/// failed.
Optional<SmallVector<Value *, 8>> expandAffineMap(FuncBuilder *builder,
                                                  Location loc,
                                                  AffineMap affineMap,
                                                  ArrayRef<Value *> operands);

} // namespace mlir

#endif // MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H
