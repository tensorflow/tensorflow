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

#ifndef MLIR_DIALECT_LINALG_LINALGOPS_H_
#define MLIR_DIALECT_LINALG_LINALGOPS_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
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

namespace detail {
struct LinalgOpInterfaceTraits {
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

  template <typename ConcreteOp> struct Model : public Concept {
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
};
} // namespace detail

/// A LinalgOp behaves like a base class for the Linalg operations that are
/// defined in LinalgLibraryOps.td. The implementation does not use inheritance
/// directly. Instead, a LinalgOp directly derives from Op, hides the `classof`
/// method and dispatches to the appropriate LinalgLibraryOp.
/// This allows writing generic passes, like tiling, for all current and future
/// LinalgOps without requiring templating and dispatch in multiple places.
class LinalgOp : public OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits> {
public:
  using OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::OpInterface;

  unsigned getNumParallelLoops() {
    return getImpl()->getNumParallelLoops(getOperation());
  }
  unsigned getNumReductionLoops() {
    return getImpl()->getNumReductionLoops(getOperation());
  }
  unsigned getNumWindowLoops() {
    return getImpl()->getNumWindowLoops(getOperation());
  }
  unsigned getNumLoops() {
    return getNumParallelLoops() + getNumReductionLoops() + getNumWindowLoops();
  }
  unsigned getNumInputs() { return getImpl()->getNumInputs(getOperation()); }
  unsigned getNumOutputs() { return getImpl()->getNumOutputs(getOperation()); }
  unsigned getNumInputsAndOutputs() {
    return getImpl()->getNumInputsAndOutputs(getOperation());
  }
  Value *getInput(unsigned i) { return getImpl()->getInput(getOperation(), i); }
  llvm::Optional<unsigned> getIndexOfInput(Value *view) {
    return getImpl()->getIndexOfInput(getOperation(), view);
  }
  ViewType getInputViewType(unsigned i) {
    return getImpl()->getInputViewType(getOperation(), i);
  }
  Operation::operand_range getInputs() {
    return getImpl()->getInputs(getOperation());
  }
  Value *getOutput(unsigned i) {
    return getImpl()->getOutput(getOperation(), i);
  }
  llvm::Optional<unsigned> getIndexOfOutput(Value *view) {
    return getImpl()->getIndexOfOutput(getOperation(), view);
  }
  ViewType getOutputViewType(unsigned i) {
    return getImpl()->getOutputViewType(getOperation(), i);
  }
  Operation::operand_range getOutputs() {
    return getImpl()->getOutputs(getOperation());
  }
  Operation::operand_range getInputsAndOutputs() {
    return getImpl()->getInputsAndOutputs(getOperation());
  }
  LinalgOp create(OpBuilder &builder, Location loc, ArrayRef<Value *> operands,
                  ArrayRef<NamedAttribute> attributes) {
    return LinalgOp(getImpl()->create(builder, loc, operands, attributes));
  }
};

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.h.inc"

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SubViewOp::Range &range);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGOPS_H_
