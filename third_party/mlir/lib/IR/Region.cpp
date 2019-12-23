//===- Region.cpp - MLIR Region Class -------------------------------------===//
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

#include "mlir/IR/Region.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

Region::Region(Operation *container) : container(container) {}

Region::~Region() {
  // Operations may have cyclic references, which need to be dropped before we
  // can start deleting them.
  dropAllReferences();
}

/// Return the context this region is inserted in. The region must have a valid
/// parent container.
MLIRContext *Region::getContext() {
  assert(container && "region is not attached to a container");
  return container->getContext();
}

/// Return a location for this region. This is the location attached to the
/// parent container. The region must have a valid parent container.
Location Region::getLoc() {
  assert(container && "region is not attached to a container");
  return container->getLoc();
}

Region *Region::getParentRegion() {
  assert(container && "region is not attached to a container");
  return container->getParentRegion();
}

Operation *Region::getParentOp() { return container; }

bool Region::isProperAncestor(Region *other) {
  if (this == other)
    return false;

  while ((other = other->getParentRegion())) {
    if (this == other)
      return true;
  }
  return false;
}

/// Return the number of this region in the parent operation.
unsigned Region::getRegionNumber() {
  // Regions are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getParentOp()->getRegions()[0];
}

/// Clone the internal blocks from this region into `dest`. Any
/// cloned blocks are appended to the back of dest.
void Region::cloneInto(Region *dest, BlockAndValueMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  cloneInto(dest, dest->end(), mapper);
}

/// Clone this region into 'dest' before the given position in 'dest'.
void Region::cloneInto(Region *dest, Region::iterator destPos,
                       BlockAndValueMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  assert(this != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (empty())
    return;

  for (Block &block : *this) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg->getType()));

    // Clone and remap the operations within this block.
    for (auto &op : block)
      newBlock->push_back(op.clone(mapper));

    dest->getBlocks().insert(destPos, newBlock);
  }

  // Now that each of the blocks have been cloned, go through and remap the
  // operands of each of the operations.
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op->getBlockOperands())
      if (auto *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (iterator it(mapper.lookup(&front())); it != destPos; ++it)
    it->walk(remapOperands);
}

void Region::dropAllReferences() {
  for (Block &b : *this)
    b.dropAllReferences();
}

/// Check if there are any values used by operations in `region` defined
/// outside its ancestor region `limit`.  That is, given `A{B{C{}}}` with region
/// `C` and limit `B`, the values defined in `B` can be used but the values
/// defined in `A` cannot.  Emit errors if `noteLoc` is provided; this location
/// is used to point to the operation containing the region, the actual error is
/// reported at the operation with an offending use.
static bool isIsolatedAbove(Region &region, Region &limit,
                            Optional<Location> noteLoc) {
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
        for (Value operand : op.getOperands()) {
          // operand should be non-null here if the IR is well-formed. But
          // we don't assert here as this function is called from the verifier
          // and so could be called on invalid IR.
          if (!operand) {
            if (noteLoc)
              op.emitOpError("block's operand not defined").attachNote(noteLoc);
            return false;
          }

          // Check that any value that is used by an operation is defined in the
          // same region as either an operation result or a block argument.
          if (operand->getParentRegion()->isProperAncestor(&limit)) {
            if (noteLoc) {
              op.emitOpError("using value defined outside the region")
                      .attachNote(noteLoc)
                  << "required by region isolation constraints";
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

bool Region::isIsolatedFromAbove(Optional<Location> noteLoc) {
  return isIsolatedAbove(*this, *this, noteLoc);
}

Region *llvm::ilist_traits<::mlir::Block>::getParentRegion() {
  size_t Offset(
      size_t(&((Region *)nullptr->*Region::getSublistAccess(nullptr))));
  iplist<Block> *Anchor(static_cast<iplist<Block> *>(this));
  return reinterpret_cast<Region *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a basic block is added to a region.
/// We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::addNodeToList(Block *block) {
  assert(!block->getParent() && "already in a region!");
  block->parentValidOpOrderPair.setPointer(getParentRegion());
}

/// This is a trait method invoked when an operation is removed from a
/// region.  We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::removeNodeFromList(Block *block) {
  assert(block->getParent() && "not already in a region!");
  block->parentValidOpOrderPair.setPointer(nullptr);
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Block>::transferNodesFromList(
    ilist_traits<Block> &otherList, block_iterator first, block_iterator last) {
  // If we are transferring operations within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getParentRegion();
  if (curParent == otherList.getParentRegion())
    return;

  // Update the 'parent' member of each Block.
  for (; first != last; ++first)
    first->parentValidOpOrderPair.setPointer(curParent);
}

//===----------------------------------------------------------------------===//
// RegionRange
//===----------------------------------------------------------------------===//

RegionRange::RegionRange(MutableArrayRef<Region> regions)
    : RegionRange(regions.data(), regions.size()) {}
RegionRange::RegionRange(ArrayRef<std::unique_ptr<Region>> regions)
    : RegionRange(regions.data(), regions.size()) {}

/// See `detail::indexed_accessor_range_base` for details.
RegionRange::OwnerT RegionRange::offset_base(const OwnerT &owner,
                                             ptrdiff_t index) {
  if (auto *operand = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return operand + index;
  return &owner.get<Region *>()[index];
}
/// See `detail::indexed_accessor_range_base` for details.
Region *RegionRange::dereference_iterator(const OwnerT &owner,
                                          ptrdiff_t index) {
  if (auto *operand = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return operand[index].get();
  return &owner.get<Region *>()[index];
}
