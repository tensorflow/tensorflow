//===- BasicBlock.cpp - MLIR BasicBlock Class -----------------------------===//
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

#include "mlir/IR/BasicBlock.h"
#include "mlir/IR/CFGFunction.h"
using namespace mlir;

BasicBlock::BasicBlock() {}

BasicBlock::~BasicBlock() {
  if (terminator)
    terminator->erase();
  for (BBArgument *arg : arguments)
    delete arg;
  arguments.clear();
}

//===----------------------------------------------------------------------===//
// Argument list management.
//===----------------------------------------------------------------------===//

BBArgument *BasicBlock::addArgument(Type type) {
  auto *arg = new BBArgument(type, this);
  arguments.push_back(arg);
  return arg;
}

/// Add one argument to the argument list for each type specified in the list.
auto BasicBlock::addArguments(ArrayRef<Type> types)
    -> llvm::iterator_range<args_iterator> {
  arguments.reserve(arguments.size() + types.size());
  auto initialSize = arguments.size();
  for (auto type : types) {
    addArgument(type);
  }
  return {arguments.data() + initialSize, arguments.data() + arguments.size()};
}

//===----------------------------------------------------------------------===//
// Terminator management
//===----------------------------------------------------------------------===//

void BasicBlock::setTerminator(TerminatorInst *inst) {
  assert((!inst || !inst->block) && "terminator already inserted into a block");
  // If we already had a terminator, abandon it.
  if (terminator)
    terminator->block = nullptr;

  // Reset our terminator to the new instruction.
  terminator = inst;
  if (inst)
    inst->block = this;
}

/// Return true if this block has no predecessors.
bool BasicBlock::hasNoPredecessors() const {
  return pred_begin() == pred_end();
}

/// If this basic block has exactly one predecessor, return it.  Otherwise,
/// return null.
///
/// Note that multiple edges from a single block (e.g. if you have a cond
/// branch with the same block as the true/false destinations) is not
/// considered to be a single predecessor.
BasicBlock *BasicBlock::getSinglePredecessor() {
  auto it = pred_begin();
  if (it == pred_end())
    return nullptr;
  auto *firstPred = *it;
  ++it;
  return it == pred_end() ? firstPred : nullptr;
}

//===----------------------------------------------------------------------===//
// ilist_traits for BasicBlock
//===----------------------------------------------------------------------===//

mlir::CFGFunction *
llvm::ilist_traits<::mlir::BasicBlock>::getContainingFunction() {
  size_t Offset(
    size_t(&((CFGFunction *)nullptr->*CFGFunction::getSublistAccess(nullptr))));
  iplist<BasicBlock> *Anchor(static_cast<iplist<BasicBlock> *>(this));
  return reinterpret_cast<CFGFunction *>(reinterpret_cast<char *>(Anchor) -
                                           Offset);
}

/// This is a trait method invoked when a basic block is added to a function.
/// We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
addNodeToList(BasicBlock *block) {
  assert(!block->function && "already in a function!");
  block->function = getContainingFunction();
}

/// This is a trait method invoked when an instruction is removed from a
/// function.  We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
removeNodeFromList(BasicBlock *block) {
  assert(block->function && "not already in a function!");
  block->function = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::BasicBlock>::
transferNodesFromList(ilist_traits<BasicBlock> &otherList,
                      block_iterator first, block_iterator last) {
  // If we are transferring instructions within the same function, the parent
  // pointer doesn't need to be updated.
  CFGFunction *curParent = getContainingFunction();
  if (curParent == otherList.getContainingFunction())
    return;

  // Update the 'function' member of each BasicBlock.
  for (; first != last; ++first)
    first->function = curParent;
}

//===----------------------------------------------------------------------===//
// Manipulators
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

  // Create an unconditional branch to the new block, and move our terminator
  // to the new block.
  auto *branchLoc =
      splitBefore == end() ? getTerminator()->getLoc() : splitBefore->getLoc();
  auto oldTerm = getTerminator();
  setTerminator(BranchInst::create(branchLoc, newBB));
  newBB->setTerminator(oldTerm);

  // Move all of the operations from the split point to the end of the function
  // into the new block.
  newBB->getOperations().splice(newBB->end(), getOperations(), splitBefore,
                                end());
  return newBB;
}
