//===- LinalgTraits.h - Linalg Traits ---------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LINALGTRAITS_H_
#define MLIR_DIALECT_LINALG_LINALGTRAITS_H_

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace OpTrait {
namespace linalg {

/// This class provides the API for ops that are known to have a specified
/// number of inputs, all passed as operands. This is used as a trait like this:
///
///   class DotOp : public Op<DotOp, OpTrait::NInputs<2>::Impl> {
///
template <unsigned N> class NInputs {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, NInputs<N>::Impl> {
  public:
    static unsigned getNumInputs() { return N; }
  };
};

/// This class provides the API for ops that are known to have a specified
/// number of inputs, all passed as operands. This is used as a trait like this:
///
///   class DotOp : public Op<DotOp, OpTrait::NOutputs<2>::Impl> {
///
template <unsigned N> class NOutputs {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, NOutputs<N>::Impl> {
  public:
    static unsigned getNumOutputs() { return N; }
  };
};

/// This class provides the API for ops that are known to operate on views. This
/// trait must be used in conjunction with an op definition or a trait that
/// provides the methods `getNumInputs` and `getNumOutputs`. This is used as a
/// trait like this:
///
///   class DotOp : public Op<DotOp, OpTrait::ViewTrait> {
///
template <typename ConcreteType>
class ViewTraits : public OpTrait::TraitBase<ConcreteType, ViewTraits> {
private:
  /// Return the number of input views. For internal use only.
  unsigned nInputs() {
    return cast<ConcreteType>(this->getOperation()).getNumInputs();
  }
  /// Return the number of input views. For internal use only.
  unsigned nOutputs() {
    return cast<ConcreteType>(this->getOperation()).getNumOutputs();
  }

public:
  /// Return the `i`-th input view.
  Value getInput(unsigned i) {
    assert(i < nInputs());
    return this->getOperation()->getOperand(i);
  }
  /// Return the index of `view` in the list of input views if found, llvm::None
  /// otherwise.
  Optional<unsigned> getIndexOfInput(Value view) {
    auto it = llvm::find(getInputs(), view);
    if (it != getInputs().end())
      return it - getInputs().begin();
    return llvm::None;
  }
  /// Return the `i`-th input view type.
  MemRefType getInputViewType(unsigned i) {
    return getInput(i)->getType().template cast<MemRefType>();
  }
  /// Return the range over input views.
  Operation::operand_range getInputs() {
    auto range = this->getOperation()->getOperands();
    return {range.begin(), range.begin() + nInputs()};
  }
  /// Return the `i`-th output view.
  Value getOutput(unsigned i) {
    return this->getOperation()->getOperand(nInputs() + i);
  }
  /// Return the index of `view` in the list of output views if found,
  /// llvm::None otherwise.
  Optional<unsigned> getIndexOfOutput(Value view) {
    auto it = llvm::find(getOutputs(), view);
    if (it != getOutputs().end())
      return it - getOutputs().begin();
    return llvm::None;
  }
  /// Return the `i`-th output view type.
  MemRefType getOutputViewType(unsigned i) {
    return getOutput(i)->getType().template cast<MemRefType>();
  }
  /// Return the range over output views.
  Operation::operand_range getOutputs() {
    auto range = this->getOperation()->getOperands();
    return {range.begin() + nInputs(),
            range.begin() + getNumInputsAndOutputs()};
  }
  /// Return the number of input and output views.
  unsigned getNumInputsAndOutputs() { return nInputs() + nOutputs(); }
  /// Return the `i`-th view type.
  MemRefType getViewType(unsigned i) {
    return (i < nInputs()) ? getInputViewType(i)
                           : getOutputViewType(i - nInputs());
  }
  /// Return the range over input and output views.
  Operation::operand_range getInputsAndOutputs() {
    auto range = this->getOperation()->getOperands();
    return {range.begin(), range.begin() + getNumInputsAndOutputs()};
  }
  unsigned getNumParallelLoops() {
    return getNumIterators(
        getParallelIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumReductionLoops() {
    return getNumIterators(
        getReductionIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumWindowLoops() {
    return getNumIterators(
        getWindowIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumLoops() {
    return getNumIterators(
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  static LogicalResult verifyTrait(Operation *op) {
    auto nViews = cast<ConcreteType>(op).getNumInputsAndOutputs();
    if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nViews)))
      return failure();
    return success();
  }
};

} // namespace linalg
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTRAITS_H_
