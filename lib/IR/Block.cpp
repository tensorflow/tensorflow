//===- Block.cpp - MLIR Block and Region Classes --------------------------===//
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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// BlockArgument
//===----------------------------------------------------------------------===//

/// Returns the number of this argument.
unsigned BlockArgument::getArgNumber() {
  // Arguments are not stored in place, so we have to find it within the list.
  auto argList = getOwner()->getArguments();
  return std::distance(argList.begin(), llvm::find(argList, this));
}

//===----------------------------------------------------------------------===//
// Block
//===----------------------------------------------------------------------===//

Block::~Block() {
  assert(!verifyInstOrder() && "Expected valid operation ordering.");
  clear();

  llvm::DeleteContainerPointers(arguments);
}

/// Returns the closest surrounding operation that contains this block or
/// nullptr if this is a top-level operation block.
Operation *Block::getContainingOp() {
  return getParent() ? getParent()->getContainingOp() : nullptr;
}

Function *Block::getFunction() {
  Block *block = this;
  while (auto *op = block->getContainingOp()) {
    block = op->getBlock();
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
  block->getParent()->getBlocks().insert(Region::iterator(block), this);
}

/// Unlink this Block from its Function and delete it.
void Block::eraseFromFunction() {
  assert(getFunction() && "Block has no parent");
  getFunction()->getBlocks().erase(this);
}

/// Returns 'op' if 'op' lies in this block, or otherwise finds the
/// ancestor operation of 'op' that lies in this block. Returns nullptr if
/// the latter fails.
Operation *Block::findAncestorInstInBlock(Operation &op) {
  // Traverse up the operation hierarchy starting from the owner of operand to
  // find the ancestor operation that resides in the block of 'forInst'.
  auto *currInst = &op;
  while (currInst->getBlock() != this) {
    currInst = currInst->getParentOp();
    if (!currInst)
      return nullptr;
  }
  return currInst;
}

/// This drops all operand uses from operations within this block, which is
/// an essential step in breaking cyclic dependences between references when
/// they are to be deleted.
void Block::dropAllReferences() {
  for (Operation &i : *this)
    i.dropAllReferences();
}

void Block::dropAllDefinedValueUses() {
  for (auto *arg : getArguments())
    arg->dropAllUses();
  for (auto &op : *this)
    op.dropAllDefinedValueUses();
  dropAllUses();
}

/// Verifies the current ordering of child operations. Returns false if the
/// order is valid, true otherwise.
bool Block::verifyInstOrder() {
  // The order is already known to be invalid.
  if (!isInstOrderValid())
    return false;
  // The order is valid if there are less than 2 operations.
  if (operations.empty() || std::next(operations.begin()) == operations.end())
    return false;

  Operation *prev = nullptr;
  for (auto &i : *this) {
    // The previous operation must have a smaller order index than the next as
    // it appears earlier in the list.
    if (prev && prev->orderIndex >= i.orderIndex)
      return true;
    prev = &i;
  }
  return false;
}

/// Recomputes the ordering of child operations within the block.
void Block::recomputeInstOrder() {
  parentValidInstOrderPair.setInt(true);

  // TODO(riverriddle) Have non-congruent indices to reduce the number of times
  // an insert invalidates the list.
  unsigned orderIndex = 0;
  for (auto &op : *this)
    op.orderIndex = orderIndex++;
}

Block *PredecessorIterator::operator*() const {
  // The use iterator points to an operand of a terminator.  The predecessor
  // we return is the block that the terminator is embedded into.
  return bbUseIterator.getUser()->getBlock();
}

/// Get the successor number in the predecessor terminator.
unsigned PredecessorIterator::getSuccessorIndex() const {
  return bbUseIterator->getOperandNumber();
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

/// Get the terminator operation of this block. This function asserts that
/// the block has a valid terminator operation.
Operation *Block::getTerminator() {
  assert(!empty() && !back().isKnownNonTerminator());
  return &back();
}

/// Return true if this block has no predecessors.
bool Block::hasNoPredecessors() { return pred_begin() == pred_end(); }

// Indexed successor access.
unsigned Block::getNumSuccessors() {
  return empty() ? 0 : back().getNumSuccessors();
}

Block *Block::getSuccessor(unsigned i) {
  assert(i < getNumSuccessors());
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
// Operation Walkers
//===----------------------------------------------------------------------===//

void Block::walk(const std::function<void(Operation *)> &callback) {
  walk(begin(), end(), callback);
}

/// Walk the operations in the specified [begin, end) range of this block,
/// calling the callback for each operation.
void Block::walk(Block::iterator begin, Block::iterator end,
                 const std::function<void(Operation *)> &callback) {
  for (auto &op : llvm::make_early_inc_range(llvm::make_range(begin, end)))
    op.walk(callback);
}

//===----------------------------------------------------------------------===//
// Other
//===----------------------------------------------------------------------===//

/// Split the block into two blocks before the specified operation or
/// iterator.
///
/// Note that all operations BEFORE the specified iterator stay as part of
/// the original basic block, and the rest of the operations in the original
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
  newBB->getOperations().splice(newBB->end(), getOperations(), splitBefore,
                                end());
  return newBB;
}

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

Region::Region(Function *container) : container(container) {}

Region::Region(Operation *container) : container(container) {}

Region::~Region() {
  // Operations may have cyclic references, which need to be dropped before we
  // can start deleting them.
  for (auto &bb : *this)
    bb.dropAllReferences();
}

Region *Region::getContainingRegion() {
  if (auto *inst = getContainingOp())
    return inst->getContainingRegion();
  return nullptr;
}

Operation *Region::getContainingOp() {
  assert(!container.isNull() && "no container");
  return container.dyn_cast<Operation *>();
}

Function *Region::getContainingFunction() {
  assert(!container.isNull() && "no container");
  return container.dyn_cast<Function *>();
}

bool Region::isProperAncestor(Region *other) {
  if (this == other)
    return false;

  while ((other = other->getContainingRegion())) {
    if (this == other)
      return true;
  }
  return false;
}

/// Clone the internal blocks from this region into `dest`. Any
/// cloned blocks are appended to the back of dest.
void Region::cloneInto(Region *dest, BlockAndValueMapping &mapper,
                       MLIRContext *context) {
  assert(dest && "expected valid region to clone into");

  // If the list is empty there is nothing to clone.
  if (empty())
    return;

  iterator lastOldBlock = --dest->end();
  for (Block &block : *this) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto *arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg->getType()));

    // Clone and remap the operations within this block.
    for (auto &op : block)
      newBlock->push_back(op.clone(mapper, context));

    dest->push_back(newBlock);
  }

  // Now that each of the blocks have been cloned, go through and remap the
  // operands of each of the operations.
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto *mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op->getBlockOperands())
      if (auto *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (auto it = std::next(lastOldBlock), e = dest->end(); it != e; ++it)
    it->walk(remapOperands);
}

// Check that the given `region` does not use any value defined outside its
// ancestor region `limit`.  That is, given `A{B{C{}}}` with limit `B`, `C` is
// allowed to use values defined in `B` but not those defined in `A`.
// Emit errors if `emitOpNote` is provided; this callback is used to point to
// the operation containing the region, the actual error is reported at the
// operation with an offending use.
static bool
isRegionIsolatedAbove(Region &region, Region &limit,
                      llvm::function_ref<void(const Twine &)> emitOpNote = {}) {
  assert(limit.isAncestor(&region) &&
         "expected isolation limit to be an ancestor of the given region");

  // List of regions to analyze.  Each region is processed independently, with
  // respect to the common `limit` region, so we can look at them in any order.
  // Therefore, use a simple vector and push/pop back the current region.
  SmallVector<Region *, 8> pendingRegions;
  pendingRegions.push_back(&region);

  // Traverse all operations in the region.
  while (!pendingRegions.empty()) {
    for (Block &block : *pendingRegions.pop_back_val()) {
      for (Operation &op : block) {
        for (Value *operand : op.getOperands()) {
          // Check that any value that is used by an operation is defined in the
          // same region as either an operation result or a block argument.
          if (operand->getContainingRegion()->isProperAncestor(&limit)) {
            if (emitOpNote) {
              op.emitOpError("using value defined outside the region");
              emitOpNote("required by region isolation constraints");
            }
            return false;
          }
        }
        // Schedule any regions the operations contain for further checking.
        pendingRegions.reserve(pendingRegions.size() + op.getNumRegions());
        for (Region &subRegion : op.getRegions())
          pendingRegions.push_back(&subRegion);
      }
    }
  }

  return true;
}

bool Region::isIsolatedAbove(
    llvm::function_ref<void(const Twine &)> noteEmitter) {
  return isRegionIsolatedAbove(*this, *this, noteEmitter);
}

Region *llvm::ilist_traits<::mlir::Block>::getContainingRegion() {
  size_t Offset(
      size_t(&((Region *)nullptr->*Region::getSublistAccess(nullptr))));
  iplist<Block> *Anchor(static_cast<iplist<Block> *>(this));
  return reinterpret_cast<Region *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a basic block is added to a region.
/// We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::addNodeToList(Block *block) {
  assert(!block->getParent() && "already in a region!");
  block->parentValidInstOrderPair.setPointer(getContainingRegion());
}

/// This is a trait method invoked when an operation is removed from a
/// region.  We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::removeNodeFromList(Block *block) {
  assert(block->getParent() && "not already in a region!");
  block->parentValidInstOrderPair.setPointer(nullptr);
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Block>::transferNodesFromList(
    ilist_traits<Block> &otherList, block_iterator first, block_iterator last) {
  // If we are transferring operations within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getContainingRegion();
  if (curParent == otherList.getContainingRegion())
    return;

  // Update the 'parent' member of each Block.
  for (; first != last; ++first)
    first->parentValidInstOrderPair.setPointer(curParent);
}
