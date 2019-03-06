//===- Intrinsics.h - MLIR Operations for Declarative Builders ---*- C++-*-===//
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
// Provides intuitive composable intrinsics for building snippets of MLIR
// declaratively
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_INTRINSICS_H_
#define MLIR_EDSC_INTRINSICS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {

namespace edsc {

class BlockHandle;
class ValueHandle;

/// Provides a set of first class intrinsics.
/// In the future, most of intrinsics reated to Instruction that don't contain
/// other instructions should be Tablegen'd.
namespace intrinsics {

/// Branches into the mlir::Block* captured by BlockHandle `b` with `operands`.
///
/// Prerequisites:
///   All Handles have already captured previously constructed IR objects.
ValueHandle BR(BlockHandle bh, ArrayRef<ValueHandle> operands);

/// Creates a new mlir::Block* and branches to it from the current block.
/// Argument types are specified by `operands`.
/// Captures the new block in `bh` and the actual `operands` in `captures`. To
/// insert the new mlir::Block*, a local ScopedContext is constructed and
/// released to the current block. The branch instruction is then added to the
/// new block.
///
/// Prerequisites:
///   `b` has not yet captured an mlir::Block*.
///   No `captures` have captured any mlir::Value*.
///   All `operands` have already captured an mlir::Value*
///   captures.size() == operands.size()
///   captures and operands are pairwise of the same type.
ValueHandle BR(BlockHandle *bh, ArrayRef<ValueHandle *> captures,
               ArrayRef<ValueHandle> operands);

/// Branches into the mlir::Block* captured by BlockHandle `trueBranch` with
/// `trueOperands` if `cond` evaluates to `true` (resp. `falseBranch` and
/// `falseOperand` if `cond` evaluates to `false`).
///
/// Prerequisites:
///   All Handles have captured previouly constructed IR objects.
ValueHandle COND_BR(ValueHandle cond, BlockHandle trueBranch,
                    ArrayRef<ValueHandle> trueOperands, BlockHandle falseBranch,
                    ArrayRef<ValueHandle> falseOperands);

/// Eagerly creates new mlir::Block* with argument types specified by
/// `trueOperands`/`falseOperands`.
/// Captures the new blocks in `trueBranch`/`falseBranch` and the arguments in
/// `trueCaptures/falseCaptures`.
/// To insert the new mlir::Block*, a local ScopedContext is constructed and
/// released. The branch instruction is then added in the original location and
/// targeting the eagerly constructed blocks.
///
/// Prerequisites:
///   `trueBranch`/`falseBranch` has not yet captured an mlir::Block*.
///   No `trueCaptures`/`falseCaptures` have captured any mlir::Value*.
///   All `trueOperands`/`trueOperands` have already captured an mlir::Value*
///   `trueCaptures`.size() == `trueOperands`.size()
///   `falseCaptures`.size() == `falseOperands`.size()
///   `trueCaptures` and `trueOperands` are pairwise of the same type
///   `falseCaptures` and `falseOperands` are pairwise of the same type.
ValueHandle COND_BR(ValueHandle cond, BlockHandle *trueBranch,
                    ArrayRef<ValueHandle *> trueCaptures,
                    ArrayRef<ValueHandle> trueOperands,
                    BlockHandle *falseBranch,
                    ArrayRef<ValueHandle *> falseCaptures,
                    ArrayRef<ValueHandle> falseOperands);

////////////////////////////////////////////////////////////////////////////////
// TODO(ntv): Intrinsics below this line should be TableGen'd.
////////////////////////////////////////////////////////////////////////////////
/// Builds an mlir::ReturnOp with the proper `operands` that each must have
/// captured an mlir::Value*.
/// Returns an empty ValueHandle.
ValueHandle RETURN(llvm::ArrayRef<ValueHandle> operands);

} // namespace intrinsics

} // namespace edsc

} // namespace mlir

#endif // MLIR_EDSC_INTRINSICS_H_
