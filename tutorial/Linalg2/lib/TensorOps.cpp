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
#include "llvm/ADT/STLExtras.h"

using llvm::ArrayRef;
using llvm::Twine;

using namespace mlir;
using namespace linalg;

//////////////////////////////////////////////////////////////////////////////
// Op-specific Dot.
//////////////////////////////////////////////////////////////////////////////
void linalg::DotOp::build(Builder *b, OperationState *result,
                          ArrayRef<Value *> operands) {
  result->addOperands(operands);
}

LogicalResult linalg::DotOp::verify() {
  if (failed(TensorContractionBaseType::verify()))
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

LogicalResult linalg::MatvecOp::verify() {
  if (failed(TensorContractionBaseType::verify()))
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

LogicalResult linalg::MatmulOp::verify() {
  if (failed(TensorContractionBaseType::verify()))
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
  return TensorContractionBaseType::parse(parser, result);
}

void linalg::MatmulOp::print(mlir::OpAsmPrinter *p) {
  TensorContractionBaseType::print(p);
}
