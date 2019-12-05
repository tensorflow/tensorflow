//===- LinalgTraits.h - Linalg Traits ---------------------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_LINALG_LINALGTRAITS_H_
#define MLIR_DIALECT_LINALG_LINALGTRAITS_H_

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace OpTrait {
namespace linalg {

/// This class provides the API for ops that are known to have a specified
/// number of inputs and outputs, all passed as operands. This is used as a
/// trait like this:
///
///   class DotOp : public Op<DotOp, OpTrait::NInputsAndOutputs<2, 1>::Impl> {
///
template <unsigned NInputs, unsigned NOutputs> class NInputsAndOutputs {
public:
  template <typename ConcreteType>
  class Impl
      : public OpTrait::TraitBase<ConcreteType,
                                  NInputsAndOutputs<NInputs, NOutputs>::Impl> {
  public:
    static unsigned getNumInputs() { return NInputs; }
    static unsigned getNumOutputs() { return NOutputs; }
    static LogicalResult verifyTrait(Operation *op) {
      return OpTrait::impl::verifyAtLeastNOperands(op, NInputs + NOutputs);
    }
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
  Value *getInput(unsigned i) {
    assert(i < nInputs());
    return this->getOperation()->getOperand(i);
  }
  /// Return the index of `view` in the list of input views if found, llvm::None
  /// otherwise.
  llvm::Optional<unsigned> getIndexOfInput(Value *view) {
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
  Value *getOutput(unsigned i) {
    return this->getOperation()->getOperand(nInputs() + i);
  }
  /// Return the index of `view` in the list of output views if found,
  /// llvm::None otherwise.
  llvm::Optional<unsigned> getIndexOfOutput(Value *view) {
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
  static LogicalResult verifyTrait(Operation *op) {
    auto nViews = cast<ConcreteType>(op).getNumInputsAndOutputs();
    if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nViews)))
      return failure();
    return success();
  }
};

/// This class provides the API for ops that are known to have a specified
/// number of parallel, reduction and window loops. This is used as a trait like
/// this:
///
///   class MatmulOp : public Op<MatmulOp, OpTrait::NLoopTypes<2, 1, 0>::Impl> {
///
template <unsigned NParallel, unsigned NReduction, unsigned NWindow = 0>
class NLoopTypes {
public:
  template <typename ConcreteType>
  class Impl
      : public OpTrait::TraitBase<
            ConcreteType, NLoopTypes<NParallel, NReduction, NWindow>::Impl> {
  public:
    static unsigned getNumParallelLoops() { return NParallel; }
    static unsigned getNumReductionLoops() { return NReduction; }
    static unsigned getNumWindowLoops() { return NWindow; }
    static unsigned getNumLoops() { return NParallel + NReduction + NWindow; }
  };
};

} // namespace linalg
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTRAITS_H_
