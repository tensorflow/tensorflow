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

StmtBlock::~StmtBlock() {
  clear();

  llvm::DeleteContainerPointers(arguments);
}

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

//===----------------------------------------------------------------------===//
// Argument list management.
//===----------------------------------------------------------------------===//

BlockArgument *StmtBlock::addArgument(Type type) {
  auto *arg = new BlockArgument(type, this);
  arguments.push_back(arg);
  return arg;
}

/// Add one argument to the argument list for each type specified in the list.
auto StmtBlock::addArguments(ArrayRef<Type> types)
    -> llvm::iterator_range<args_iterator> {
  arguments.reserve(arguments.size() + types.size());
  auto initialSize = arguments.size();
  for (auto type : types) {
    addArgument(type);
  }
  return {arguments.data() + initialSize, arguments.data() + arguments.size()};
}

void StmtBlock::eraseArgument(unsigned index) {
  assert(index < arguments.size());

  // Delete the argument.
  delete arguments[index];
  arguments.erase(arguments.begin() + index);

  // Erase this argument from each of the predecessor's terminator.
  for (auto predIt = pred_begin(), predE = pred_end(); predIt != predE;
       ++predIt) {
    auto *predTerminator = (*predIt)->getTerminator();
    predTerminator->eraseSuccessorOperand(predIt.getSuccessorIndex(), index);
  }
}

//===----------------------------------------------------------------------===//
// Terminator management
//===----------------------------------------------------------------------===//

OperationStmt *StmtBlock::getTerminator() {
  if (empty())
    return nullptr;

  // Check if the last instruction is a terminator.
  auto &backInst = statements.back();
  auto *opStmt = dyn_cast<OperationStmt>(&backInst);
  if (!opStmt || !opStmt->isTerminator())
    return nullptr;
  return opStmt;
}

/// Return true if this block has no predecessors.
bool StmtBlock::hasNoPredecessors() const { return pred_begin() == pred_end(); }

// Indexed successor access.
unsigned StmtBlock::getNumSuccessors() const {
  return getTerminator()->getNumSuccessors();
}

StmtBlock *StmtBlock::getSuccessor(unsigned i) {
  return getTerminator()->getSuccessor(i);
}

/// If this block has exactly one predecessor, return it.  Otherwise, return
/// null.
///
/// Note that multiple edges from a single block (e.g. if you have a cond
/// branch with the same block as the true/false destinations) is not
/// considered to be a single predecessor.
StmtBlock *StmtBlock::getSinglePredecessor() {
  auto it = pred_begin();
  if (it == pred_end())
    return nullptr;
  auto *firstPred = *it;
  ++it;
  return it == pred_end() ? firstPred : nullptr;
}
