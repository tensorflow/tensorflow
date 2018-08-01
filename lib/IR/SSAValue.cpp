//===- Instructions.cpp - MLIR CFGFunction Instruction Classes ------------===//
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
#include "mlir/IR/Instructions.h"
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
