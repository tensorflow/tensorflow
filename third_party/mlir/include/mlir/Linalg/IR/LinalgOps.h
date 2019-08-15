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
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Linalg/IR/LinalgTraits.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace linalg {

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%A, %f) : !linalg.view<f32>, f32
///   name mangles into `linalg_fill_viewf32_f32_impl`
///
/// 2. linalg.dot(%A, %B, %C) :
///      !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32_impl`
///
/// 3. linalg.matmul(...) :
///      !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32_impl`
std::string generateLibraryCallName(Operation *op);

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Linalg/IR/LinalgLibraryOps.h.inc"

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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SubViewOp::Range &range);

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
