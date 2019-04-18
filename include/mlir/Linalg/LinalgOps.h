//===- LinalgOps.h - Linalg Operations --------------------------*- C++ -*-===//
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

#ifndef MLIR_LINALG_LINALGOPS_H_
#define MLIR_LINALG_LINALGOPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// A BufferAllocOp is used to create a 1-D !linalg.buffer upon which a base
/// view can be laid out. The size argument is an `i64` (and not an index), so
/// that we can
class BufferAllocOp
    : public Op<BufferAllocOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.buffer_alloc"; }
  static void build(Builder *b, OperationState *result, Type type, Value *size);
  LogicalResult verify();
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *size() { return getOperand(); }
  BufferType getBufferType() { return getType().cast<BufferType>(); }
  Type getElementType() { return getBufferType().getElementType(); }
};

/// A BufferDeallocOp is used to free a !linalg.buffer.
class BufferDeallocOp
    : public Op<BufferDeallocOp, OpTrait::OneOperand, OpTrait::ZeroResult> {
public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.buffer_dealloc"; }
  static void build(Builder *b, OperationState *result, Value *buffer);
  LogicalResult verify();
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *getBuffer() { return getOperand(); }
  BufferType getBufferType() {
    return getOperand()->getType().cast<BufferType>();
  }
};

/// A RangeOp is used to create a value of RangeType from 3 values of type index
/// that represent the min, max and step values of the range.
class RangeOp : public Op<RangeOp, OpTrait::NOperands<3>::Impl,
                          OpTrait::OneResult, OpTrait::HasNoSideEffect> {
public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.range"; }
  static void build(Builder *b, OperationState *result, Value *min, Value *max,
                    Value *step);
  LogicalResult verify();
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *min() { return getOperand(0); }
  Value *max() { return getOperand(1); }
  Value *step() { return getOperand(2); }
};

} // namespace mlir

#endif // MLIR_LINALG_LINALGOPS_H_
