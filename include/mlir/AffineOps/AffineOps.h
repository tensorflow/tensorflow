//===- AffineOps.h - MLIR Affine Operations -------------------------------===//
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
// This file defines convenience types for working with Affine operations
// in the MLIR instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AFFINEOPS_AFFINEOPS_H
#define MLIR_AFFINEOPS_AFFINEOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {

class AffineOpsDialect : public Dialect {
public:
  AffineOpsDialect(MLIRContext *context);
};

/// The "if" operation represents an if–then–else construct for conditionally
/// executing two regions of code. The operands to an if operation are an
/// IntegerSet condition and a set of symbol/dimension operands to the
/// condition set. The operation produces no results. For example:
///
///    if #set(%i)  {
///      ...
///    } else {
///      ...
///    }
///
/// The 'else' blocks to the if operation are optional, and may be omitted. For
/// example:
///
///    if #set(%i)  {
///      ...
///    }
///
class AffineIfOp
    : public Op<AffineIfOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  // Hooks to customize behavior of this op.
  static void build(Builder *builder, OperationState *result,
                    IntegerSet condition, ArrayRef<Value *> conditionOperands);

  static StringRef getOperationName() { return "if"; }
  static StringRef getConditionAttrName() { return "condition"; }

  IntegerSet getIntegerSet() const;
  void setIntegerSet(IntegerSet newSet);

  /// Returns the list of 'then' blocks.
  BlockList &getThenBlocks();
  const BlockList &getThenBlocks() const {
    return const_cast<AffineIfOp *>(this)->getThenBlocks();
  }

  /// Returns the list of 'else' blocks.
  BlockList &getElseBlocks();
  const BlockList &getElseBlocks() const {
    return const_cast<AffineIfOp *>(this)->getElseBlocks();
  }

  bool verify() const;
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p) const;

private:
  friend class OperationInst;
  explicit AffineIfOp(const OperationInst *state) : Op(state) {}
};

} // end namespace mlir

#endif
