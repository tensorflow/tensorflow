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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Linalg/IR/LinalgTraits.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class OperationFolder;

namespace linalg {

/// A linalg.LoadOp is the counterpart of load but operating on ViewType
/// instead of MemRefType.
///
/// ```{.mlir}
///    %0 = linalg.load %V[%c0] : !linalg.view<?xf32>
/// ```
class LoadOp
    : public Op<LoadOp, OpTrait::VariadicOperands, OpTrait::OneResult> {
public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.load"; }
  static void build(Builder *b, OperationState *result, Value *view,
                    ArrayRef<Value *> indices = {});
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  ViewType getViewType() { return getView()->getType().cast<ViewType>(); }
  Value *getView() { return getOperand(0); }
  Operation::operand_range getIndices() {
    return {operand_begin() + 1, operand_end()};
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
class SliceOp : public Op<SliceOp, OpTrait::VariadicOperands,
                          OpTrait::OneResult, OpTrait::HasNoSideEffect> {
  enum { FirstIndexingOperand = 1 };

public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.slice"; }
  static void build(Builder *b, OperationState *result, Value *base,
                    llvm::ArrayRef<Value *> indexings);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  Type getElementType() { return getViewType().getElementType(); }
  ViewType getViewType() { return getType().cast<ViewType>(); }
  Value *getBaseView() { return getOperand(0); }
  ViewOp getBaseViewOp();
  ViewType getBaseViewType();
  unsigned getBaseViewRank() { return getBaseViewType().getRank(); }
  // Get the underlying indexing at a given rank.
  Value *getIndexing(unsigned rank) { return *(getIndexings().begin() + rank); }
  // Get all the indexings in this view.
  Operation::operand_range getIndexings() {
    return {operand_begin() + SliceOp::FirstIndexingOperand, operand_end()};
  }
  // Get the subset of indexings that are of RangeType.
  SmallVector<Value *, 8> getRanges();
};

/// A linalg.StoreOp is the counterpart of affine.store but operating on
/// ViewType instead of MemRefType.
///
/// ```{.mlir}
///    linalg.store %f, %V[%c0] : !linalg.view<?xf32>
/// ```
class StoreOp
    : public Op<StoreOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.store"; }
  static void build(Builder *b, OperationState *result, Value *valueToStore,
                    Value *view, ArrayRef<Value *> indices = {});
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  ViewType getViewType() { return getView()->getType().cast<ViewType>(); }
  Value *getValueToStore() { return getOperand(0); }
  Value *getView() { return getOperand(1); }
  Operation::operand_range getIndices() {
    return {operand_begin() + 2, operand_end()};
  }
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
class ViewOp : public Op<ViewOp, OpTrait::VariadicOperands, OpTrait::OneResult,
                         OpTrait::HasNoSideEffect> {
  enum { FirstIndexingOperand = 1 };

public:
  using Op::Op;

  // Hooks to customize the behavior of this op.
  static llvm::StringRef getOperationName() { return "linalg.view"; }
  static void build(Builder *b, OperationState *result, Value *buffer,
                    llvm::ArrayRef<Value *> indexings);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  // Op-specific functionality.
  unsigned getRank() { return getViewType().getRank(); }
  Type getElementType() { return getViewType().getElementType(); }
  ViewType getViewType() { return getType().cast<ViewType>(); }
  Value *getSupportingBuffer() { return getOperand(0); }
  // Get the underlying indexing at a given rank.
  Value *getIndexing(unsigned rank) { return *(getIndexings().begin() + rank); }
  // Get all the indexings in this view.
  Operation::operand_range getIndexings() {
    return {operand_begin() + ViewOp::FirstIndexingOperand, operand_end()};
  }
};

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgLibraryOps.h.inc"

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SubViewOp::Range &range);

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

/// A LinalgOp behaves like a base class for the Linalg operations that are
/// defined in LinalgLibraryOps.td. The implementation does not use inheritance
/// directly. Instead, a LinalgOp directly derives from Op, hides the `classof`
/// method and dispatches to the appropriate LinalgLibraryOp.
/// This allows writing generic passes, like tiling, for all current and future
/// LinalgOps without requiring templating and dispatch in multiple places.
class LinalgOp : public Op<LinalgOp> {
public:
  using Op::Op;

  LinalgOp(Operation *op) : Op<LinalgOp>(op) {
    impl = ModelDispatch<
#define GET_OP_LIST
#include "mlir/Linalg/IR/LinalgLibraryOps.cpp.inc"
        >::dispatch(op);
  }

  static bool classof(Operation *op) {
    return ModelDispatch<
#define GET_OP_LIST
#include "mlir/Linalg/IR/LinalgLibraryOps.cpp.inc"
        >::classof(op);
  }

  unsigned getNumParallelLoops() {
    return impl->getNumParallelLoops(getOperation());
  }
  unsigned getNumReductionLoops() {
    return impl->getNumReductionLoops(getOperation());
  }
  unsigned getNumWindowLoops() {
    return impl->getNumWindowLoops(getOperation());
  }
  unsigned getNumLoops() {
    return getNumParallelLoops() + getNumReductionLoops() + getNumWindowLoops();
  }
  unsigned getNumInputs() { return impl->getNumInputs(getOperation()); }
  unsigned getNumOutputs() { return impl->getNumOutputs(getOperation()); }
  unsigned getNumInputsAndOutputs() {
    return impl->getNumInputsAndOutputs(getOperation());
  }
  Value *getInput(unsigned i) { return impl->getInput(getOperation(), i); }
  llvm::Optional<unsigned> getIndexOfInput(Value *view) {
    return impl->getIndexOfInput(getOperation(), view);
  }
  ViewType getInputViewType(unsigned i) {
    return impl->getInputViewType(getOperation(), i);
  }
  Operation::operand_range getInputs() {
    return impl->getInputs(getOperation());
  }
  Value *getOutput(unsigned i) { return impl->getOutput(getOperation(), i); }
  llvm::Optional<unsigned> getIndexOfOutput(Value *view) {
    return impl->getIndexOfOutput(getOperation(), view);
  }
  ViewType getOutputViewType(unsigned i) {
    return impl->getOutputViewType(getOperation(), i);
  }
  Operation::operand_range getOutputs() {
    return impl->getOutputs(getOperation());
  }
  Operation::operand_range getInputsAndOutputs() {
    return impl->getInputsAndOutputs(getOperation());
  }
  LinalgOp create(OpBuilder &builder, Location loc, ArrayRef<Value *> operands,
                  ArrayRef<NamedAttribute> attributes) {
    return LinalgOp(impl->create(builder, loc, operands, attributes));
  }

private:
  struct Concept {
    virtual ~Concept() = default;
    virtual unsigned getNumInputs(Operation *op) = 0;
    virtual unsigned getNumOutputs(Operation *op) = 0;
    virtual unsigned getNumInputsAndOutputs(Operation *op) = 0;
    virtual unsigned getNumParallelLoops(Operation *op) = 0;
    virtual unsigned getNumReductionLoops(Operation *op) = 0;
    virtual unsigned getNumWindowLoops(Operation *op) = 0;
    virtual Value *getInput(Operation *op, unsigned i) = 0;
    virtual llvm::Optional<unsigned> getIndexOfInput(Operation *op,
                                                     Value *view) = 0;
    virtual ViewType getInputViewType(Operation *op, unsigned i) = 0;
    virtual Operation::operand_range getInputs(Operation *op) = 0;
    virtual Value *getOutput(Operation *op, unsigned i) = 0;
    virtual llvm::Optional<unsigned> getIndexOfOutput(Operation *op,
                                                      Value *view) = 0;
    virtual ViewType getOutputViewType(Operation *op, unsigned i) = 0;
    virtual Operation::operand_range getOutputs(Operation *op) = 0;
    virtual Operation::operand_range getInputsAndOutputs(Operation *op) = 0;
    virtual Operation *create(OpBuilder &builder, Location loc,
                              ArrayRef<Value *> operands,
                              ArrayRef<NamedAttribute> attributes) = 0;
  };

  /// The implementation is inspired from Sean Parent's concept-based
  /// polymorphism. A key difference is that the set of classes erased is
  /// statically known, which alleviates the need for using dynamic memory
  /// allocation.
  /// We use a zero-sized templated class `Model<ConcreteOp>` to emit the
  /// virtual table and generate a singleton object for each instantiation of
  /// this class.
  /// We pay the cost of initialization once on construction (find which class
  /// to dispatch to) and then a virtual dispatch on every call.
  template <typename ConcreteOp> struct Model : public Concept {
    static Model<ConcreteOp> &instance() {
      static Model<ConcreteOp> singleton;
      return singleton;
    }
    unsigned getNumInputs(Operation *op) override {
      return cast<ConcreteOp>(op).getNumInputs();
    }
    unsigned getNumOutputs(Operation *op) override {
      return cast<ConcreteOp>(op).getNumOutputs();
    }
    unsigned getNumInputsAndOutputs(Operation *op) override {
      return cast<ConcreteOp>(op).getNumInputsAndOutputs();
    }
    unsigned getNumParallelLoops(Operation *op) override {
      return cast<ConcreteOp>(op).getNumParallelLoops();
    }
    unsigned getNumReductionLoops(Operation *op) override {
      return cast<ConcreteOp>(op).getNumReductionLoops();
    }
    unsigned getNumWindowLoops(Operation *op) override {
      return cast<ConcreteOp>(op).getNumWindowLoops();
    }
    Value *getInput(Operation *op, unsigned i) override {
      return cast<ConcreteOp>(op).getInput(i);
    }
    llvm::Optional<unsigned> getIndexOfInput(Operation *op,
                                             Value *view) override {
      return cast<ConcreteOp>(op).getIndexOfInput(view);
    }
    ViewType getInputViewType(Operation *op, unsigned i) override {
      return cast<ConcreteOp>(op).getInputViewType(i);
    }
    Operation::operand_range getInputs(Operation *op) override {
      return cast<ConcreteOp>(op).getInputs();
    }
    Value *getOutput(Operation *op, unsigned i) override {
      return cast<ConcreteOp>(op).getOutput(i);
    }
    llvm::Optional<unsigned> getIndexOfOutput(Operation *op,
                                              Value *view) override {
      return cast<ConcreteOp>(op).getIndexOfOutput(view);
    }
    ViewType getOutputViewType(Operation *op, unsigned i) override {
      return cast<ConcreteOp>(op).getOutputViewType(i);
    }
    Operation::operand_range getOutputs(Operation *op) override {
      return cast<ConcreteOp>(op).getOutputs();
    }
    Operation::operand_range getInputsAndOutputs(Operation *op) override {
      return cast<ConcreteOp>(op).getInputsAndOutputs();
    }
    Operation *create(OpBuilder &builder, Location loc,
                      ArrayRef<Value *> operands,
                      ArrayRef<NamedAttribute> attributes) override {
      return builder.create<ConcreteOp>(loc, ArrayRef<Type>{}, operands,
                                        attributes);
    }
  };
  Concept *impl;

  template <typename... Types> struct ModelDispatch;

  template <typename First, typename... Rest>
  struct ModelDispatch<First, Rest...> {
    static bool classof(Operation *op) {
      return isa<First>(op) || ModelDispatch<Rest...>::classof(op);
    }
    static Concept *dispatch(Operation *op) {
      return isa<First>(op) ? &Model<First>::instance()
                            : ModelDispatch<Rest...>::dispatch(op);
    }
  };

  template <typename...> struct ModelDispatch {
    static bool classof(Operation *op) { return false; }
    static Concept *dispatch(Operation *op) {
      llvm_unreachable("Invalid LinalgOp");
    }
  };
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_LINALGOPS_H_
