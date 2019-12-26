//===- Ops.h - Loop MLIR Operations -----------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
ForOp getForInductionVarOwner(Value val);

} // end namespace loop
} // end namespace mlir
#endif // MLIR_LOOPOPS_OPS_H_
