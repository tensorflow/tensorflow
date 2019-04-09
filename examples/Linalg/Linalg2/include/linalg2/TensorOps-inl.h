//===- TensorOps-inl.h - Linalg dialect TensorOps operation implementation ===//
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

/// The TensorOp-inl.h inclusion pattern is chosen to allow gradual extension of
/// TensorOps by adding implementations as they are needed in the appropriate
/// step in the tutorial.
#ifndef LINALG2_TENSOROPS_INL_H_
#define LINALG2_TENSOROPS_INL_H_

#include "linalg2/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace linalg {

template <class ConcreteOp>
mlir::Operation::operand_range
linalg::TensorContractionBase<ConcreteOp>::getInputs() {
  auto *op = static_cast<ConcreteOp *>(this)->getOperation();
  return {op->operand_begin(), op->operand_begin() + getNumInputs()};
}

template <class ConcreteOp>
mlir::Operation::operand_range
linalg::TensorContractionBase<ConcreteOp>::getOutputs() {
  auto *op = static_cast<ConcreteOp *>(this)->getOperation();
  return {op->operand_begin() + getNumInputs(),
          op->operand_begin() + getNumInputs() + getNumOutputs()};
}

template <class ConcreteOp>
mlir::Operation::operand_range
linalg::TensorContractionBase<ConcreteOp>::getInputsAndOutputs() {
  return {getInputs().begin(), getOutputs().end()};
}

template <class ConcreteOp>
mlir::LogicalResult linalg::TensorContractionBase<ConcreteOp>::verify() {
  auto *concreteOp = static_cast<ConcreteOp *>(this)->getOperation();
  if (getNumInputs() <= 0)
    concreteOp->emitOpError("expected at least one input");
  if (getNumOutputs() <= 0)
    concreteOp->emitOpError("expected at least one output");
  if (concreteOp->getNumOperands() != getNumInputs() + getNumOutputs()) {
    concreteOp->emitOpError("expected " +
                            llvm::Twine(getNumInputs() + getNumOutputs()) +
                            " operands");
  }
  for (unsigned i = 0, e = getNumInputs(); i < e; ++i) {
    if (!concreteOp->getOperand(i)->getType().template isa<ViewType>())
      return concreteOp->emitOpError("operand " + llvm::Twine(i) +
                                     " not a ViewType");
  }
  for (unsigned i = getNumInputs(), e = getNumInputs() + getNumOutputs(); i < e;
       ++i) {
    auto viewType =
        concreteOp->getOperand(i)->getType().template dyn_cast<ViewType>();
    if (!viewType)
      return concreteOp->emitOpError("operand " + llvm::Twine(i) +
                                     " not a ViewType");
    if (viewType.getRank() != getNumParallelDims())
      return concreteOp->emitOpError("operand " + llvm::Twine(i) +
                                     " must be of rank " +
                                     llvm::Twine(getNumParallelDims()));
  }
  return mlir::success();
}

template <class ConcreteOp>
bool linalg::TensorContractionBase<ConcreteOp>::parse(
    mlir::OpAsmParser *parser, mlir::OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

// A TensorContraction prints as:
//
// ```{.mlir}
//   concrete_op_name (ssa-inputs, ssa-outputs) : output-view-types
// ```
//
// for example:
//
// ```
//   linalg.matmul(%0, %1, %2) : view<?x?xf32>
// ```
//
// Where %0, %1 and %2 are ssa-values of type ViewType.
template <class ConcreteOp>
void linalg::TensorContractionBase<ConcreteOp>::print(mlir::OpAsmPrinter *p) {
  *p << static_cast<ConcreteOp *>(this)->getOperationName() << "(";
  auto *last = *std::prev(getInputsAndOutputs().end());
  for (auto *i : getInputsAndOutputs()) {
    *p << *i << ((i == last) ? "" : ", ");
  }
  *p << ") : ";
  auto *lastOutput = *std::prev(getOutputs().end());
  for (auto *o : getOutputs()) {
    *p << o->getType() << ((o == lastOutput) ? "" : ",");
  }
}

} // namespace linalg

#endif // LINALG2_TENSOROPS_INL_H_
