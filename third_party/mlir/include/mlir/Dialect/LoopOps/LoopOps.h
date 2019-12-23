//===- Ops.h - Loop MLIR Operations -----------------------------*- C++ -*-===//
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
// This file defines convenience types for working with loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LOOPOPS_OPS_H_
#define MLIR_LOOPOPS_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/LoopLikeInterface.h"

namespace mlir {
namespace loop {

class TerminatorOp;

class LoopOpsDialect : public Dialect {
public:
  LoopOpsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "loop"; }
};

#define GET_OP_CLASSES
#include "mlir/Dialect/LoopOps/LoopOps.h.inc"

// Insert `loop.terminator` at the end of the only region's only block if it
// does not have a terminator already.  If a new `loop.terminator` is inserted,
// the location is specified by `loc`. If the region is empty, insert a new
// block first.
void ensureLoopTerminator(Region &region, Builder &builder, Location loc);

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
ForOp getForInductionVarOwner(ValuePtr val);

} // end namespace loop
} // end namespace mlir
#endif // MLIR_LOOPOPS_OPS_H_
