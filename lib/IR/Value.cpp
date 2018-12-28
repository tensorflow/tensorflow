//===- Value.cpp - MLIR Value Classes -------------------------------------===//
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

#include "mlir/IR/Value.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Statements.h"
using namespace mlir;

/// If this value is the result of an Instruction, return the instruction
/// that defines it.
OperationInst *Value::getDefiningInst() {
  if (auto *result = dyn_cast<InstResult>(this))
    return result->getOwner();
  return nullptr;
}

/// Return the function that this Value is defined in.
Function *Value::getFunction() {
  switch (getKind()) {
  case Value::Kind::BlockArgument:
    return cast<BlockArgument>(this)->getFunction();
  case Value::Kind::StmtResult:
    return getDefiningInst()->getFunction();
  case Value::Kind::ForStmt:
    return cast<ForStmt>(this)->getFunction();
  }
}

//===----------------------------------------------------------------------===//
// IROperandOwner implementation.
//===----------------------------------------------------------------------===//

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead.  When this returns
/// there are zero uses of 'this'.
void IRObjectWithUseList::replaceAllUsesWith(IRObjectWithUseList *newValue) {
  assert(this != newValue && "cannot RAUW a value with itself");
  while (!use_empty()) {
    use_begin()->set(newValue);
  }
}

/// Return the context this operation is associated with.
MLIRContext *IROperandOwner::getContext() const {
  switch (getKind()) {
  case Kind::OperationInst:
    return cast<OperationInst>(this)->getContext();
  case Kind::ForStmt:
    return cast<ForStmt>(this)->getContext();
  case Kind::IfStmt:
    return cast<IfStmt>(this)->getContext();
  }
}

//===----------------------------------------------------------------------===//
// BlockArgument implementation.
//===----------------------------------------------------------------------===//

/// Return the function that this argument is defined in.
MLFunction *BlockArgument::getFunction() {
  if (auto *owner = getOwner())
    return owner->getFunction();
  return nullptr;
}
