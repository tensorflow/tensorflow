//===- SSAValue.cpp - MLIR SSAValue Classes ------------===//
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

#include "mlir/IR/SSAValue.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Instructions.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Statements.h"
using namespace mlir;

/// If this value is the result of an OperationInst, return the instruction
/// that defines it.
OperationInst *SSAValue::getDefiningInst() {
  if (auto *result = dyn_cast<InstResult>(this))
    return result->getOwner();
  return nullptr;
}

/// If this value is the result of an OperationStmt, return the statement
/// that defines it.
OperationStmt *SSAValue::getDefiningStmt() {
  if (auto *result = dyn_cast<StmtResult>(this))
    return result->getOwner();
  return nullptr;
}

Operation *SSAValue::getDefiningOperation() {
  if (auto *inst = getDefiningInst())
    return inst;
  if (auto *stmt = getDefiningStmt())
    return stmt;
  return nullptr;
}

/// Return the function that this SSAValue is defined in.
Function *SSAValue::getFunction() {
  switch (getKind()) {
  case SSAValueKind::BBArgument:
    return cast<BBArgument>(this)->getFunction();
  case SSAValueKind::InstResult:
    return getDefiningInst()->getFunction();
  case SSAValueKind::MLFuncArgument:
    return cast<MLFuncArgument>(this)->getFunction();
  case SSAValueKind::StmtResult:
    return getDefiningStmt()->findFunction();
  case SSAValueKind::ForStmt:
    return cast<ForStmt>(this)->findFunction();
  }
}

//===----------------------------------------------------------------------===//
// CFGValue implementation.
//===----------------------------------------------------------------------===//

/// Return the function that this CFGValue is defined in.
CFGFunction *CFGValue::getFunction() {
  return cast<CFGFunction>(static_cast<SSAValue *>(this)->getFunction());
}

//===----------------------------------------------------------------------===//
// BBArgument implementation.
//===----------------------------------------------------------------------===//

/// Return the function that this argument is defined in.
CFGFunction *BBArgument::getFunction() {
  if (auto *owner = getOwner())
    return owner->getFunction();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MLValue implementation.
//===----------------------------------------------------------------------===//

/// Return the function that this MLValue is defined in.
MLFunction *MLValue::getFunction() {
  return cast<MLFunction>(static_cast<SSAValue *>(this)->getFunction());
}

// MLValue can be used a a dimension id if it is valid as a symbol, or
// it is an induction variable, or it is a result of affine apply operation
// with dimension id arguments.
bool MLValue::isValidDim() const {
  if (auto *stmt = getDefiningStmt()) {
    // Top level statement or constant operation is ok.
    if (stmt->getParentStmt() == nullptr || stmt->is<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = stmt->getAs<AffineApplyOp>())
      return op->isValidDim();
    return false;
  }
  // This value is either a function argument or an induction variable. Both
  // are ok.
  return true;
}

// MLValue can be used as a symbol if it is a constant, or it is defined at
// the top level, or it is a result of affine apply operation with symbol
// arguments.
bool MLValue::isValidSymbol() const {
  if (auto *stmt = getDefiningStmt()) {
    // Top level statement or constant operation is ok.
    if (stmt->getParentStmt() == nullptr || stmt->is<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = stmt->getAs<AffineApplyOp>())
      return op->isValidSymbol();
    return false;
  }
  // This value is either a function argument or an induction variable.
  // Function argument is ok, induction variable is not.
  return isa<MLFuncArgument>(this);
}
