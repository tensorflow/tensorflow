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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;

StmtBlock::~StmtBlock() {
  clear();

  llvm::DeleteContainerPointers(arguments);
}

/// Returns the closest surrounding statement that contains this block or
/// nullptr if this is a top-level statement block.
Statement *StmtBlock::getContainingStmt() {
  return parent ? parent->getContainingStmt() : nullptr;
}

MLFunction *StmtBlock::getFunction() {
  StmtBlock *block = this;
  while (auto *stmt = block->getContainingStmt()) {
    block = stmt->getBlock();
    if (!block)
      return nullptr;
  }
  if (auto *list = block->getParent())
    return list->getContainingFunction();
  return nullptr;
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

//===----------------------------------------------------------------------===//
// Other
//===----------------------------------------------------------------------===//

/// Unlink this BasicBlock from its CFGFunction and delete it.
void BasicBlock::eraseFromFunction() {
  assert(getFunction() && "BasicBlock has no parent");
  getFunction()->getBlocks().erase(this);
}

/// Split the basic block into two basic blocks before the specified
/// instruction or iterator.
///
/// Note that all instructions BEFORE the specified iterator stay as part of
/// the original basic block, an unconditional branch is added to the original
/// block (going to the new block), and the rest of the instructions in the
/// original block are moved to the new BB, including the old terminator.  The
/// newly formed BasicBlock is returned.
///
/// This function invalidates the specified iterator.
BasicBlock *BasicBlock::splitBasicBlock(iterator splitBefore) {
  // Start by creating a new basic block, and insert it immediate after this
  // one in the containing function.
  auto newBB = new BasicBlock();
  getFunction()->getBlocks().insert(++CFGFunction::iterator(this), newBB);
  auto branchLoc =
      splitBefore == end() ? getTerminator()->getLoc() : splitBefore->getLoc();

  // Move all of the operations from the split point to the end of the function
  // into the new block.
  newBB->getStatements().splice(newBB->end(), getStatements(), splitBefore,
                                end());

  // Create an unconditional branch to the new block, and move our terminator
  // to the new block.
  FuncBuilder(this).create<BranchOp>(branchLoc, newBB);
  return newBB;
}

//===----------------------------------------------------------------------===//
// StmtBlockList
//===----------------------------------------------------------------------===//

StmtBlockList::StmtBlockList(Function *container) : container(container) {}

StmtBlockList::StmtBlockList(Statement *container) : container(container) {}

CFGFunction *StmtBlockList::getFunction() {
  return dyn_cast_or_null<CFGFunction>(getContainingFunction());
}

Statement *StmtBlockList::getContainingStmt() {
  return container.dyn_cast<Statement *>();
}

Function *StmtBlockList::getContainingFunction() {
  return container.dyn_cast<Function *>();
}

StmtBlockList *llvm::ilist_traits<::mlir::StmtBlock>::getContainingBlockList() {
  size_t Offset(size_t(
      &((StmtBlockList *)nullptr->*StmtBlockList::getSublistAccess(nullptr))));
  iplist<StmtBlock> *Anchor(static_cast<iplist<StmtBlock> *>(this));
  return reinterpret_cast<StmtBlockList *>(reinterpret_cast<char *>(Anchor) -
                                           Offset);
}

/// This is a trait method invoked when a basic block is added to a function.
/// We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::StmtBlock>::addNodeToList(StmtBlock *block) {
  assert(!block->parent && "already in a function!");
  block->parent = getContainingBlockList();
}

/// This is a trait method invoked when an instruction is removed from a
/// function.  We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::StmtBlock>::removeNodeFromList(
    StmtBlock *block) {
  assert(block->parent && "not already in a function!");
  block->parent = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::StmtBlock>::transferNodesFromList(
    ilist_traits<StmtBlock> &otherList, block_iterator first,
    block_iterator last) {
  // If we are transferring instructions within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getContainingBlockList();
  if (curParent == otherList.getContainingBlockList())
    return;

  // Update the 'parent' member of each StmtBlock.
  for (; first != last; ++first)
    first->parent = curParent;
}
