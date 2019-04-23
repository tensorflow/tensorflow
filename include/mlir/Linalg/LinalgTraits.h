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

#ifndef MLIR_LINALG_LINALGTRAITS_H_
#define MLIR_LINALG_LINALGTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace OpTrait {

/// This class provides the API for ops that are known to have a specified
/// number of inputs and outputs, all passed as operands. This is used as a
/// trait like this:
///
///   class DotOp : public Op<DotOp, OpTrait::NInputsAndOutputs<2, 1>::Impl> {
///
template <unsigned NInputs, unsigned NOutputs> class NInputsAndOutputs {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::detail::MultiOperandTraitBase<
                   ConcreteType, NInputsAndOutputs<NInputs, NOutputs>::Impl> {
  public:
    static unsigned getNumInputs() { return NInputs; }
    static unsigned getNumOutputs() { return NOutputs; }
    static unsigned getNumInputsAndOutputs() { return NInputs + NOutputs; }
    Value *getInput(unsigned i) { return this->getOperand(i); }
    Value *getOutput(unsigned i) {
      return this->getOperand(getNumInputs() + i);
    }
    ViewType getInputViewType(unsigned i) {
      return this->getOperand(i)->getType().template cast<ViewType>();
    }
    ViewType getOutputViewType(unsigned i) {
      return this->getOperand(getNumInputs() + i)
          ->getType()
          .template cast<ViewType>();
    }
    ViewType getViewType(unsigned i) {
      return this->getOperand(i)->getType().template cast<ViewType>();
    }
    static LogicalResult verifyTrait(Operation *op) {
      return OpTrait::impl::verifyAtLeastNOperands(op, NInputs + NOutputs);
    }
  };
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

/// This class provides the API for ops that are known to have a specified
/// list of view ranks. This is used as a trait like this:
///
///   class MatvecOp : public Op<MatvecOp, OpTrait::ViewRanks<2, 1, 1>::Impl> {
///
template <unsigned... Ranks> class ViewRanks {
public:
  template <typename ConcreteType>
  class Impl
      : public OpTrait::TraitBase<ConcreteType, ViewRanks<Ranks...>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      ArrayRef<unsigned> ranks{Ranks...};
      if (op->getNumOperands() != ranks.size())
        return op->emitError("expected " + Twine(ranks.size()) + " operands");
      for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
        auto viewType = op->getOperand(i)->getType().dyn_cast<ViewType>();
        if (!viewType)
          return op->emitOpError("operand " + Twine(i) +
                                 " must have view type ");
        if (ranks[i] != viewType.getRank())
          return op->emitOpError("operand " + Twine(i) + " must have rank " +
                                 Twine(ranks[i]));
      }
      return success();
    }
  };
};

} // namespace OpTrait
} // namespace mlir

#endif // MLIR_LINALG_LINALGTRAITS_H_
