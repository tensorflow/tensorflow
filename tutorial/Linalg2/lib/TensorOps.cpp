//===- TensorOps.cpp - Implementation of the linalg TensorOps operation ---===//
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
// This file implements a simple IR operation to create new tensor computation
// operations in the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg2/Analysis.h"
#include "linalg2/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/STLExtras.h"

using llvm::ArrayRef;
using llvm::Twine;

using namespace mlir;
using namespace linalg;

template <class ConcreteOp>
Type linalg::TensorContractionBase<ConcreteOp>::getInputElementType(
    unsigned idx) {
  return getInputView(idx)
      ->getType()
      .template cast<ViewType>()
      .getElementType();
}

template <class ConcreteOp>
Type linalg::TensorContractionBase<ConcreteOp>::getOutputElementType(
    unsigned idx) {
  return getOutputView(idx)
      ->getType()
      .template cast<ViewType>()
      .getElementType();
}

template <class ConcreteOp>
Value *linalg::TensorContractionBase<ConcreteOp>::getInputView(unsigned idx) {
  return *(getInputs().begin() + idx);
}

template <class ConcreteOp>
Value *linalg::TensorContractionBase<ConcreteOp>::getOutputView(unsigned idx) {
  return *(getOutputs().begin() + idx);
}

template <class ConcreteOp>
Value *linalg::TensorContractionBase<ConcreteOp>::getInputMemRef(unsigned idx) {
  return getViewSupportingMemRef(*(getInputs().begin() + idx));
}

template <class ConcreteOp>
Value *
linalg::TensorContractionBase<ConcreteOp>::getOutputMemRef(unsigned idx) {
  return getViewSupportingMemRef(*(getOutputs().begin() + idx));
}

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
bool linalg::TensorContractionBase<ConcreteOp>::verify() {
  auto *concreteOp = static_cast<ConcreteOp *>(this)->getOperation();
  if (getNumInputs() <= 0)
    concreteOp->emitOpError("expected at least one input");
  if (getNumOutputs() <= 0)
    concreteOp->emitOpError("expected at least one output");
  if (concreteOp->getNumOperands() != getNumInputs() + getNumOutputs()) {
    concreteOp->emitOpError(
        "expected " + Twine(getNumInputs() + getNumOutputs()) + " operands");
  }
  for (unsigned i = 0, e = getNumInputs(); i < e; ++i) {
    if (!concreteOp->getOperand(i)->getType().template isa<ViewType>())
      return concreteOp->emitOpError("operand " + Twine(i) + " not a ViewType");
  }
  for (unsigned i = getNumInputs(), e = getNumInputs() + getNumOutputs(); i < e;
       ++i) {
    auto viewType =
        concreteOp->getOperand(i)->getType().template dyn_cast<ViewType>();
    if (!viewType)
      return concreteOp->emitOpError("operand " + Twine(i) + " not a ViewType");
    if (viewType.getRank() != getNumParallelDims())
      return concreteOp->emitOpError("operand " + Twine(i) +
                                     " must be of rank " +
                                     Twine(getNumParallelDims()));
  }
  return false;
}

template <class ConcreteOp>
bool linalg::TensorContractionBase<ConcreteOp>::parse(OpAsmParser *parser,
                                                      OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

// A TensorContraction prints as:
//
// ```{.mlir}
//   concrete_op_name {%0, %1} -> {%2}
// ```
//
// Where %0, %1 is an ssa-value holding a View, %2 is an ssa-value holding a
// view.
template <class ConcreteOp>
void linalg::TensorContractionBase<ConcreteOp>::print(OpAsmPrinter *p) {
  *p << static_cast<ConcreteOp *>(this)->getOperationName() << " {";
  auto *lastInput = *std::prev(getInputs().end());
  for (auto *i : getInputs()) {
    *p << *i << ((i == lastInput) ? "} -> {" : ", ");
  }
  auto *lastOutput = *std::prev(getOutputs().end());
  for (auto *o : getOutputs()) {
    *p << *o << ((o == lastOutput) ? "}" : ",");
  }
}

//////////////////////////////////////////////////////////////////////////////
// Op-specific Dot.
//////////////////////////////////////////////////////////////////////////////
void linalg::DotOp::build(Builder *b, OperationState *result,
                          ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

bool linalg::DotOp::verify() {
  if (TensorContractionBaseType::verify())
    return true;
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  unsigned index = 0;
  for (auto *v : {A, B}) {
    if (getViewRank(v) != 1)
      return emitOpError("operand " + Twine(index++) + " must be of rank 1");
  }
  if (getViewRank(C) != 0)
    return emitOpError("operand 2 must be of rank 0");
  // TODO(ntv): check ranges match.
  return false;
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::DotOp::parse(mlir::OpAsmParser *parser,
                          mlir::OperationState *result) {
  return TensorContractionBaseType::parse(parser, result);
}

void linalg::DotOp::print(mlir::OpAsmPrinter *p) {
  TensorContractionBaseType::print(p);
}

//////////////////////////////////////////////////////////////////////////////
// Op-specific Matvec.
//////////////////////////////////////////////////////////////////////////////
void linalg::MatvecOp::build(Builder *b, OperationState *result,
                             ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

bool linalg::MatvecOp::verify() {
  if (TensorContractionBaseType::verify())
    return true;
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  if (getViewRank(A) != 2)
    return emitOpError("operand 0 must be of rank 2");
  unsigned index = 0;
  for (auto *v : {B, C}) {
    if (getViewRank(v) != 1)
      return emitOpError("operand " + Twine(1 + index++) +
                         " must be of rank 1");
  }
  // TODO(ntv): check ranges match.
  return false;
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::MatvecOp::parse(mlir::OpAsmParser *parser,
                             mlir::OperationState *result) {
  return TensorContractionBaseType::parse(parser, result);
}

void linalg::MatvecOp::print(mlir::OpAsmPrinter *p) {
  TensorContractionBaseType::print(p);
}

//////////////////////////////////////////////////////////////////////////////
// Op-specific Matmul.
//////////////////////////////////////////////////////////////////////////////
void linalg::MatmulOp::build(Builder *b, OperationState *result,
                             ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

bool linalg::MatmulOp::verify() {
  if (TensorContractionBaseType::verify())
    return true;
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  unsigned index = 0;
  for (auto *v : {A, B, C}) {
    if (getViewRank(v) != 2)
      return emitOpError("operand " + Twine(index++) + " must be of rank 2");
  }
  // TODO(ntv): check ranges match.
  return false;
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::MatmulOp::parse(mlir::OpAsmParser *parser,
                             mlir::OperationState *result) {
  return TensorContractionBaseType::parse(parser, result);
}

void linalg::MatmulOp::print(mlir::OpAsmPrinter *p) {
  TensorContractionBaseType::print(p);
}
