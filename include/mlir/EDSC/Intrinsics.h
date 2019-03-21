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

#include "mlir/EDSC/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class MemRefType;
class Type;

namespace edsc {

class BlockHandle;
class InstructionHandle;
class ValueHandle;

/// Provides a set of first class intrinsics.
/// In the future, most of intrinsics reated to Instruction that don't contain
/// other instructions should be Tablegen'd.
namespace intrinsics {
/// Helper variadic abstraction to allow extending to any MLIR op without
/// boilerplate or Tablegen.
/// Arguably a builder is not a ValueHandle but in practice it is only used as
/// an alias to a notional ValueHandle<Op>.
/// Implementing it as a subclass allows it to compose all the way to Value*.
/// Without subclassing, implicit conversion to Value* would fail when composing
/// in patterns such as: `select(a, b, select(c, d, e))`.
template <typename Op> struct EDSCValueBuilder : public ValueHandle {
  template <typename... Args>
  EDSCValueBuilder(Args... args)
      : ValueHandle(ValueHandle::create<Op>(std::forward<Args>(args)...)) {}
  EDSCValueBuilder() : ValueHandle(ValueHandle::create<Op>()) {}
};

template <typename Op>
struct EDSCInstructionBuilder : public InstructionHandle {
  template <typename... Args>
  EDSCInstructionBuilder(Args... args)
      : InstructionHandle(
            InstructionHandle::create<Op>(std::forward<Args>(args)...)) {}
  EDSCInstructionBuilder()
      : InstructionHandle(InstructionHandle::create<Op>()) {}
};

using alloc = EDSCValueBuilder<AllocOp>;
using constant_float = EDSCValueBuilder<ConstantFloatOp>;
using constant_index = EDSCValueBuilder<ConstantIndexOp>;
using constant_int = EDSCValueBuilder<ConstantIntOp>;
using dealloc = EDSCInstructionBuilder<DeallocOp>;
using load = EDSCValueBuilder<LoadOp>;
using ret = EDSCInstructionBuilder<ReturnOp>;
using select = EDSCValueBuilder<SelectOp>;
using store = EDSCInstructionBuilder<StoreOp>;
using vector_type_cast = EDSCValueBuilder<VectorTypeCastOp>;

/// Branches into the mlir::Block* captured by BlockHandle `b` with `operands`.
///
/// Prerequisites:
///   All Handles have already captured previously constructed IR objects.
InstructionHandle br(BlockHandle bh, ArrayRef<ValueHandle> operands);

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
InstructionHandle br(BlockHandle *bh, ArrayRef<ValueHandle *> captures,
                     ArrayRef<ValueHandle> operands);

/// Branches into the mlir::Block* captured by BlockHandle `trueBranch` with
/// `trueOperands` if `cond` evaluates to `true` (resp. `falseBranch` and
/// `falseOperand` if `cond` evaluates to `false`).
///
/// Prerequisites:
///   All Handles have captured previouly constructed IR objects.
InstructionHandle cond_br(ValueHandle cond, BlockHandle trueBranch,
                          ArrayRef<ValueHandle> trueOperands,
                          BlockHandle falseBranch,
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
InstructionHandle cond_br(ValueHandle cond, BlockHandle *trueBranch,
                          ArrayRef<ValueHandle *> trueCaptures,
                          ArrayRef<ValueHandle> trueOperands,
                          BlockHandle *falseBranch,
                          ArrayRef<ValueHandle *> falseCaptures,
                          ArrayRef<ValueHandle> falseOperands);
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_INTRINSICS_H_
