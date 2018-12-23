//===- StmtBlock.cpp - MLIR Statement Instruction Classes -----------------===//
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

#include "mlir/IR/StmtBlock.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Statement block
//===----------------------------------------------------------------------===//

Statement *StmtBlock::getContainingStmt() {
  switch (kind) {
  case StmtBlockKind::MLFunc:
    return nullptr;
  case StmtBlockKind::ForBody:
    return cast<ForStmtBody>(this)->getFor();
  case StmtBlockKind::IfClause:
    return cast<IfClause>(this)->getIf();
  }
}

MLFunction *StmtBlock::findFunction() const {
  // FIXME: const incorrect.
  StmtBlock *block = const_cast<StmtBlock *>(this);

  while (block->getContainingStmt()) {
    block = block->getContainingStmt()->getBlock();
    if (!block)
      return nullptr;
  }
  return dyn_cast<MLFunction>(block);
}

/// Returns 'stmt' if 'stmt' lies in this block, or otherwise finds the ancestor
/// statement of 'stmt' that lies in this block. Returns nullptr if the latter
/// fails.
const Statement *
StmtBlock::findAncestorStmtInBlock(const Statement &stmt) const {
  // Traverse up the statement hierarchy starting from the owner of operand to
  // find the ancestor statement that resides in the block of 'forStmt'.
  const auto *currStmt = &stmt;
  while (currStmt->getBlock() != this) {
    currStmt = currStmt->getParentStmt();
    if (!currStmt)
      return nullptr;
  }
  return currStmt;
}
