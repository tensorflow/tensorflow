//===- Block.cpp - MLIR Block and BlockList Classes -----------------------===//
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

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
using namespace mlir;

Block::~Block() {
  clear();

  llvm::DeleteContainerPointers(arguments);
}

/// Returns the closest surrounding instruction that contains this block or
/// nullptr if this is a top-level instruction block.
Instruction *Block::getContainingInst() {
  return parent ? parent->getContainingInst() : nullptr;
}

Function *Block::getFunction() {
  Block *block = this;
  while (auto *inst = block->getContainingInst()) {
    block = inst->getBlock();
    if (!block)
      return nullptr;
  }
  if (auto *list = block->getParent())
    return list->getContainingFunction();
  return nullptr;
}

/// Insert this block (which must not already be in a function) right before
/// the specified block.
void Block::insertBefore(Block *block) {
  assert(!getParent() && "already inserted into a block!");
  assert(block->getParent() && "cannot insert before a block without a parent");
  block->getParent()->getBlocks().insert(BlockList::iterator(block), this);
}

/// Unlink this Block from its Function and delete it.
void Block::eraseFromFunction() {
  assert(getFunction() && "Block has no parent");
  getFunction()->getBlocks().erase(this);
}

/// Returns 'inst' if 'inst' lies in this block, or otherwise finds the
/// ancestor instruction of 'inst' that lies in this block. Returns nullptr if
/// the latter fails.
Instruction *Block::findAncestorInstInBlock(Instruction *inst) {
  // Traverse up the instruction hierarchy starting from the owner of operand to
  // find the ancestor instruction that resides in the block of 'forInst'.
  auto *currInst = inst;
  while (currInst->getBlock() != this) {
    currInst = currInst->getParentInst();
    if (!currInst)
      return nullptr;
  }
  return currInst;
}

//===----------------------------------------------------------------------===//
// Argument list management.
//===----------------------------------------------------------------------===//

BlockArgument *Block::addArgument(Type type) {
  auto *arg = new BlockArgument(type, this);
  arguments.push_back(arg);
  return arg;
}

/// Add one argument to the argument list for each type specified in the list.
auto Block::addArguments(ArrayRef<Type> types)
    -> llvm::iterator_range<args_iterator> {
  arguments.reserve(arguments.size() + types.size());
  auto initialSize = arguments.size();
  for (auto type : types) {
    addArgument(type);
  }
  return {arguments.data() + initialSize, arguments.data() + arguments.size()};
}

void Block::eraseArgument(unsigned index) {
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

OperationInst *Block::getTerminator() {
  if (empty())
    return nullptr;

  // Check if the last instruction is a terminator.
  auto &backInst = back();
  auto *opInst = dyn_cast<OperationInst>(&backInst);
  if (!opInst || !opInst->isTerminator())
    return nullptr;
  return opInst;
}

/// Return true if this block has no predecessors.
bool Block::hasNoPredecessors() const { return pred_begin() == pred_end(); }

// Indexed successor access.
unsigned Block::getNumSuccessors() const {
  return getTerminator()->getNumSuccessors();
}

Block *Block::getSuccessor(unsigned i) {
  return getTerminator()->getSuccessor(i);
}

/// If this block has exactly one predecessor, return it.  Otherwise, return
/// null.
///
/// Note that multiple edges from a single block (e.g. if you have a cond
/// branch with the same block as the true/false destinations) is not
/// considered to be a single predecessor.
Block *Block::getSinglePredecessor() {
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

/// Split the block into two blocks before the specified instruction or
/// iterator.
///
/// Note that all instructions BEFORE the specified iterator stay as part of
/// the original basic block, and the rest of the instructions in the original
/// block are moved to the new block, including the old terminator.  The
/// original block is left without a terminator.
///
/// The newly formed Block is returned, and the specified iterator is
/// invalidated.
Block *Block::splitBlock(iterator splitBefore) {
  // Start by creating a new basic block, and insert it immediate after this
  // one in the containing function.
  auto newBB = new Block();
  getFunction()->getBlocks().insert(++Function::iterator(this), newBB);

  // Move all of the operations from the split point to the end of the function
  // into the new block.
  newBB->getInstructions().splice(newBB->end(), getInstructions(), splitBefore,
                                  end());
  return newBB;
}

//===----------------------------------------------------------------------===//
// BlockList
//===----------------------------------------------------------------------===//

BlockList::BlockList(Function *container) : container(container) {}

BlockList::BlockList(Instruction *container) : container(container) {}

Instruction *BlockList::getContainingInst() {
  return container.dyn_cast<Instruction *>();
}

Function *BlockList::getContainingFunction() {
  return container.dyn_cast<Function *>();
}

BlockList *llvm::ilist_traits<::mlir::Block>::getContainingBlockList() {
  size_t Offset(
      size_t(&((BlockList *)nullptr->*BlockList::getSublistAccess(nullptr))));
  iplist<Block> *Anchor(static_cast<iplist<Block> *>(this));
  return reinterpret_cast<BlockList *>(reinterpret_cast<char *>(Anchor) -
                                       Offset);
}

/// This is a trait method invoked when a basic block is added to a function.
/// We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::Block>::addNodeToList(Block *block) {
  assert(!block->parent && "already in a function!");
  block->parent = getContainingBlockList();
}

/// This is a trait method invoked when an instruction is removed from a
/// function.  We keep the function pointer up to date.
void llvm::ilist_traits<::mlir::Block>::removeNodeFromList(Block *block) {
  assert(block->parent && "not already in a function!");
  block->parent = nullptr;
}

/// This is a trait method invoked when an instruction is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Block>::transferNodesFromList(
    ilist_traits<Block> &otherList, block_iterator first, block_iterator last) {
  // If we are transferring instructions within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getContainingBlockList();
  if (curParent == otherList.getContainingBlockList())
    return;

  // Update the 'parent' member of each Block.
  for (; first != last; ++first)
    first->parent = curParent;
}
