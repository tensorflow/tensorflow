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

#include "linalg1/Utils.h"
#include "linalg2/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace linalg;

#define TENSOR_CONTRACTION_DISPATCH(FUNCTION_NAME)                             \
  if (getTensorContractionName() == MatmulOp::getOperationName()) {            \
    return FUNCTION_NAME(static_cast<MatmulOp &>(*this));                      \
  }                                                                            \
  if (getTensorContractionName() == MatvecOp::getOperationName()) {            \
    return FUNCTION_NAME(static_cast<MatvecOp &>(*this));                      \
  }                                                                            \
  if (getTensorContractionName() == DotOp::getOperationName()) {               \
    return FUNCTION_NAME(static_cast<DotOp &>(*this));                         \
  }                                                                            \
  llvm_unreachable("Missing linalg op");

template <typename ConcreteOp>
static mlir::Operation::operand_range getInputs(ConcreteOp &concreteOp) {
  return {concreteOp.operand_begin(),
          concreteOp.operand_begin() + concreteOp.getNumInputs()};
}

mlir::Operation::operand_range linalg::TensorContractionBase::getInputs() {
  TENSOR_CONTRACTION_DISPATCH(::getInputs);
}

template <typename ConcreteOp>
static mlir::Operation::operand_range getOutputs(ConcreteOp &concreteOp) {
  return {concreteOp.operand_begin() + concreteOp.getNumInputs(),
          concreteOp.operand_begin() + concreteOp.getNumInputs() +
              concreteOp.getNumOutputs()};
}

mlir::Operation::operand_range linalg::TensorContractionBase::getOutputs() {
  TENSOR_CONTRACTION_DISPATCH(::getOutputs);
}

template <typename LinalgOp>
static mlir::LogicalResult verifyLinalgOp(LinalgOp op) {
  if (op.getNumInputs() <= 0)
    op.emitOpError("expected at least one input");
  if (op.getNumOutputs() <= 0)
    op.emitOpError("expected at least one output");
  if (op.getNumOperands() != op.getNumInputs() + op.getNumOutputs()) {
    op.emitOpError("expected " +
                   llvm::Twine(op.getNumInputs() + op.getNumOutputs()) +
                   " operands");
  }
  for (unsigned i = 0, e = op.getNumInputs(); i < e; ++i) {
    if (!op.getOperand(i)->getType().template isa<ViewType>())
      return op.emitOpError("operand " + llvm::Twine(i) + " not a ViewType");
  }
  for (unsigned i = op.getNumInputs(),
                e = op.getNumInputs() + op.getNumOutputs();
       i < e; ++i) {
    auto viewType = op.getOperand(i)->getType().template dyn_cast<ViewType>();
    if (!viewType)
      return op.emitOpError("operand " + llvm::Twine(i) + " not a ViewType");
    if (viewType.getRank() != op.getNumParallelDims())
      return op.emitOpError("operand " + llvm::Twine(i) + " must be of rank " +
                            llvm::Twine(op.getNumParallelDims()));
  }
  return mlir::success();
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
template <typename LinalgOp>
static void printLinalgOp(mlir::OpAsmPrinter *p, LinalgOp op) {
  *p << op.getOperationName() << "(";
  auto *last = *std::prev(op.getInputsAndOutputs().end());
  for (auto *i : op.getInputsAndOutputs()) {
    *p << *i << ((i == last) ? "" : ", ");
  }
  *p << ") : ";
  auto *lastOutput = *std::prev(op.getOutputs().end());
  for (auto *o : op.getOutputs()) {
    *p << o->getType() << ((o == lastOutput) ? "" : ",");
  }
}

//////////////////////////////////////////////////////////////////////////////
// Op-specific Dot.
//////////////////////////////////////////////////////////////////////////////
void linalg::DotOp::build(Builder *b, OperationState *result,
                          ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

LogicalResult linalg::DotOp::verify() {
  if (failed(verifyLinalgOp(*this)))
    return failure();
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  unsigned index = 0;
  for (auto *v : {A, B}) {
    if (getViewRank(v) != 1)
      return emitOpError("operand " + Twine(index++) + " must be of rank 1");
  }
  if (getViewRank(C) != 0)
    return emitOpError("operand 2 must be of rank 0");
  // TODO(ntv): check ranges match.
  return success();
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::DotOp::parse(mlir::OpAsmParser *parser,
                          mlir::OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

void linalg::DotOp::print(mlir::OpAsmPrinter *p) { printLinalgOp(p, *this); }

//////////////////////////////////////////////////////////////////////////////
// Op-specific Matvec.
//////////////////////////////////////////////////////////////////////////////
void linalg::MatvecOp::build(Builder *b, OperationState *result,
                             ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

LogicalResult linalg::MatvecOp::verify() {
  if (failed(verifyLinalgOp(*this)))
    return failure();
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
  return success();
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::MatvecOp::parse(mlir::OpAsmParser *parser,
                             mlir::OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

void linalg::MatvecOp::print(mlir::OpAsmPrinter *p) { printLinalgOp(p, *this); }

//////////////////////////////////////////////////////////////////////////////
// Op-specific Matmul.
//////////////////////////////////////////////////////////////////////////////
void linalg::MatmulOp::build(Builder *b, OperationState *result,
                             ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

LogicalResult linalg::MatmulOp::verify() {
  if (failed(verifyLinalgOp(*this)))
    return failure();
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  unsigned index = 0;
  for (auto *v : {A, B, C}) {
    if (getViewRank(v) != 2)
      return emitOpError("operand " + Twine(index++) + " must be of rank 2");
  }
  // TODO(ntv): check ranges match.
  return success();
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::MatmulOp::parse(mlir::OpAsmParser *parser,
                             mlir::OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

void linalg::MatmulOp::print(mlir::OpAsmPrinter *p) { printLinalgOp(p, *this); }
