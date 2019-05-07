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
#include "mlir/Linalg/IR/LinalgTraits.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// The "buffer_alloc" op creates a 1-D linalg.buffer of the specified type,
/// upon which a base view can be laid out to give it indexing semantics.
/// "buffer_alloc" takes a single argument, the size of the buffer to allocate
/// (in number of elements).
///
/// ```{.mlir}
///     %0 = linalg.buffer_alloc %arg0 : !linalg.buffer<f32>
/// ```
class BufferAllocOp
    : public Op<BufferAllocOp, OpTrait::OneOperand, OpTrait::OneResult> {
public:
  friend Operation;
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.buffer_alloc"; }
  static void build(Builder *b, OperationState *result, Type type, Value *size);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *size() { return getOperand(); }
  BufferType getBufferType() { return getType().cast<BufferType>(); }
  Type getElementType() { return getBufferType().getElementType(); }
};

/// The "buffer_dealloc" op frees a 1-D linalg.buffer of the specified type.
///
/// ```{.mlir}
///     linalg.buffer_dealloc %0 : !linalg.buffer<f32>
/// ```
class BufferDeallocOp
    : public Op<BufferDeallocOp, OpTrait::OneOperand, OpTrait::ZeroResult> {
public:
  friend Operation;
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.buffer_dealloc"; }
  static void build(Builder *b, OperationState *result, Value *buffer);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *getBuffer() { return getOperand(); }
  BufferType getBufferType() {
    return getOperand()->getType().cast<BufferType>();
  }
};

/// The "linalg.range" op creates a linalg.range from 3 values of type `index`
/// that represent the min, max and step values of the range.
///
/// ```{.mlir}
///    %3 = linalg.range %0:%1:%2 : !linalg.range
/// ```
class RangeOp : public Op<RangeOp, OpTrait::NOperands<3>::Impl,
                          OpTrait::OneResult, OpTrait::HasNoSideEffect> {
public:
  friend Operation;
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.range"; }
  static void build(Builder *b, OperationState *result, Value *min, Value *max,
                    Value *step);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  Value *min() { return getOperand(0); }
  Value *max() { return getOperand(1); }
  Value *step() { return getOperand(2); }
};

/// The "linalg.slice" op produces a linalg.view which is a subview of a given
/// base view. This allows defining a subregion within the underlying buffer to
/// operate on only a subset of the buffer.
///
/// A "linalg.slice" op takes a base view and a variadic number of indexings and
/// produces a linalg.view of the same elemental type as the buffer. An indexing
/// is either:
///   1. a linalg.range, in which case it does not reduce the rank of the parent
///      view.
///   2. an index, in which case it reduces the rank of the parent view by one.
///
/// The parent view must be a base view (i.e. either a function argument or has
/// been produced by a linalg.view op). In other words, chains of
/// linalg.slice operations cannot be constructed in the IR. This defines away
/// problems related to keeping track of which dimensions of the base view have
/// been rank-reduced.
///
/// Examples:
///   1. rank-preserving slice:
///
/// ```{.mlir}
///    %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, !linalg.range,
///    !linalg.range, !linalg.view<?x?xf32>
/// ```
///
///   2. rank-reducing slice (from 2-D to 1-D):
///
/// ```{.mlir}
///    %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, index,
///    !linalg.range, !linalg.view<?xf32>
/// ```
///
///   3. rank-reducing slice (from 2-D to 0-D):
///
/// ```{.mlir}
///    %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, index, index,
///    !linalg.view<f32>
/// ```
class ViewOp;
class SliceOp : public mlir::Op<SliceOp, mlir::OpTrait::VariadicOperands,
                                mlir::OpTrait::OneResult,
                                mlir::OpTrait::HasNoSideEffect> {
  enum { FirstIndexingOperand = 1 };

public:
  friend Operation;
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.slice"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *base, llvm::ArrayRef<mlir::Value *> indexings);
  mlir::LogicalResult verify();
  static ParseResult parse(mlir::OpAsmParser *parser,
                           mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  mlir::Type getElementType() { return getViewType().getElementType(); }
  ViewType getViewType() { return getType().cast<ViewType>(); }
  Value *getBaseView() { return getOperand(0); }
  ViewOp getBaseViewOp();
  ViewType getBaseViewType();
  unsigned getBaseViewRank() { return getBaseViewType().getRank(); }
  // Get the underlying indexing at a given rank.
  mlir::Value *getIndexing(unsigned rank) {
    return *(getIndexings().begin() + rank);
  }
  // Get all the indexings in this view.
  mlir::Operation::operand_range getIndexings() {
    return {operand_begin() + SliceOp::FirstIndexingOperand, operand_end()};
  }
  // Get the subset of indexings that are of RangeType.
  SmallVector<Value *, 8> getRanges();
};

/// The "linalg.view" op produces a linalg.view which is a multi-dimensional
/// range abstraction on top of an underlying linalg.buffer. This gives an
/// indexing structure to an otherwise non-indexable linalg.buffer.
///
/// A "linalg.view" takes a buffer and a variadic number of ranges and produces
/// a `view` of the same elemental type as the buffer and of rank the number of
/// ranges:
///
/// ```{.mlir}
///    %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
///    %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
///    %3 = linalg.view %1[%2, %2] : !linalg.view<?x?xf32>
/// ```
class ViewOp : public mlir::Op<ViewOp, mlir::OpTrait::VariadicOperands,
                               mlir::OpTrait::OneResult,
                               mlir::OpTrait::HasNoSideEffect> {
  enum { FirstIndexingOperand = 1 };

public:
  friend Operation;
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.view"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *buffer,
                    llvm::ArrayRef<mlir::Value *> indexings);
  mlir::LogicalResult verify();
  static ParseResult parse(mlir::OpAsmParser *parser,
                           mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  mlir::Type getElementType() { return getViewType().getElementType(); }
  ViewType getViewType() { return getType().cast<ViewType>(); }
  mlir::Value *getSupportingBuffer() { return getOperand(0); }
  // Get the underlying indexing at a given rank.
  mlir::Value *getIndexing(unsigned rank) {
    return *(getIndexings().begin() + rank);
  }
  // Get all the indexings in this view.
  mlir::Operation::operand_range getIndexings() {
    return {operand_begin() + ViewOp::FirstIndexingOperand, operand_end()};
  }
};

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgOps.h.inc"

/// Returns the list of maps that map loops to operands of a Linalg op.
/// The i-th affine map identifies loop indices to subscripts that are used when
/// accessing the i-th operand.
/// For instance, a matmul that can be written in index notation as:
/// `A(i, k) * B(k, j) -> C(i, j)` will have the following, ordered, list of
/// affine maps:
///
/// ```{.mlir}
///    (
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    )
/// ```
///
/// Only permutation maps are currently supported. 
SmallVector<AffineMap, 4> loopToOperandRangesMaps(Operation *op);

} // namespace mlir

#endif // MLIR_LINALG_LINALGOPS_H_
