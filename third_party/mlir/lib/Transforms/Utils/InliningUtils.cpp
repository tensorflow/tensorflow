//===- InliningUtils.cpp ---- Misc utilities for inlining -----------------===//
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
//
// This file implements miscellaneous inlining utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "inlining"

using namespace mlir;

/// Remap locations from the inlined blocks with CallSiteLoc locations with the
/// provided caller location.
static void
remapInlinedLocations(llvm::iterator_range<Region::iterator> inlinedBlocks,
                      Location callerLoc) {
  DenseMap<Location, Location> mappedLocations;
  auto remapOpLoc = [&](Operation *op) {
    auto it = mappedLocations.find(op->getLoc());
    if (it == mappedLocations.end()) {
      auto newLoc = CallSiteLoc::get(op->getLoc(), callerLoc);
      it = mappedLocations.try_emplace(op->getLoc(), newLoc).first;
    }
    op->setLoc(it->second);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOpLoc);
}

static void
remapInlinedOperands(llvm::iterator_range<Region::iterator> inlinedBlocks,
                     BlockAndValueMapping &mapper) {
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto *mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOperands);
}

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

InlinerInterface::~InlinerInterface() {}

bool InlinerInterface::isLegalToInline(
    Region *dest, Region *src, BlockAndValueMapping &valueMapping) const {
  // Regions can always be inlined into functions.
  if (isa<FuncOp>(dest->getParentOp()))
    return true;

  auto *handler = getInterfaceFor(dest->getParentOp());
  return handler ? handler->isLegalToInline(src, dest, valueMapping) : false;
}

bool InlinerInterface::isLegalToInline(
    Operation *op, Region *dest, BlockAndValueMapping &valueMapping) const {
  auto *handler = getInterfaceFor(op);
  return handler ? handler->isLegalToInline(op, dest, valueMapping) : false;
}

bool InlinerInterface::shouldAnalyzeRecursively(Operation *op) const {
  auto *handler = getInterfaceFor(op);
  return handler ? handler->shouldAnalyzeRecursively(op) : true;
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op, Block *newDest) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, newDest);
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op,
                                        ArrayRef<Value *> valuesToRepl) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, valuesToRepl);
}

/// Utility to check that all of the operations within 'src' can be inlined.
static bool isLegalToInline(InlinerInterface &interface, Region *src,
                            Region *insertRegion,
                            BlockAndValueMapping &valueMapping) {
  for (auto &block : *src) {
    for (auto &op : block) {
      // Check this operation.
      if (!interface.isLegalToInline(&op, insertRegion, valueMapping))
        return false;
      // Check any nested regions.
      if (interface.shouldAnalyzeRecursively(&op) &&
          llvm::any_of(op.getRegions(), [&](Region &region) {
            return !isLegalToInline(interface, &region, insertRegion,
                                    valueMapping);
          }))
        return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Inline Methods
//===----------------------------------------------------------------------===//

LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 BlockAndValueMapping &mapper,
                                 ArrayRef<Value *> resultsToReplace,
                                 llvm::Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  // Check that all of the region arguments have been mapped.
  auto *srcEntryBlock = &src->front();
  if (llvm::any_of(srcEntryBlock->getArguments(),
                   [&](BlockArgument *arg) { return !mapper.contains(arg); }))
    return failure();

  // The insertion point must be within a block.
  Block *insertBlock = inlinePoint->getBlock();
  if (!insertBlock)
    return failure();
  Region *insertRegion = insertBlock->getParent();

  // Check that the operations within the source region are valid to inline.
  if (!interface.isLegalToInline(insertRegion, src, mapper) ||
      !isLegalToInline(interface, src, insertRegion, mapper))
    return failure();

  // Split the insertion block.
  Block *postInsertBlock =
      insertBlock->splitBlock(++inlinePoint->getIterator());

  // Check to see if the region is being cloned, or moved inline. In either
  // case, move the new blocks after the 'insertBlock' to improve IR
  // readability.
  if (shouldCloneInlinedRegion)
    src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
  else
    insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                     src->getBlocks(), src->begin(),
                                     src->end());

  // Get the range of newly inserted blocks.
  auto newBlocks = llvm::make_range(std::next(insertBlock->getIterator()),
                                    postInsertBlock->getIterator());
  Block *firstNewBlock = &*newBlocks.begin();

  // Remap the locations of the inlined operations if a valid source location
  // was provided.
  if (inlineLoc && !inlineLoc->isa<UnknownLoc>())
    remapInlinedLocations(newBlocks, *inlineLoc);

  // If the blocks were moved in-place, make sure to remap any necessary
  // operands.
  if (!shouldCloneInlinedRegion)
    remapInlinedOperands(newBlocks, mapper);

  // Handle the case where only a single block was inlined.
  if (std::next(newBlocks.begin()) == newBlocks.end()) {
    // Have the interface handle the terminator of this block.
    auto *firstBlockTerminator = firstNewBlock->getTerminator();
    interface.handleTerminator(firstBlockTerminator, resultsToReplace);
    firstBlockTerminator->erase();

    // Merge the post insert block into the cloned entry block.
    firstNewBlock->getOperations().splice(firstNewBlock->end(),
                                          postInsertBlock->getOperations());
    postInsertBlock->erase();
  } else {
    // Otherwise, there were multiple blocks inlined. Add arguments to the post
    // insertion block to represent the results to replace.
    for (Value *resultToRepl : resultsToReplace) {
      resultToRepl->replaceAllUsesWith(
          postInsertBlock->addArgument(resultToRepl->getType()));
    }

    /// Handle the terminators for each of the new blocks.
    for (auto &newBlock : newBlocks)
      interface.handleTerminator(newBlock.getTerminator(), postInsertBlock);
  }

  // Splice the instructions of the inlined entry block into the insert block.
  insertBlock->getOperations().splice(insertBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
  return success();
}

/// This function is an overload of the above 'inlineRegion' that allows for
/// providing the set of operands ('inlinedOperands') that should be used
/// in-favor of the region arguments when inlining.
LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 ArrayRef<Value *> inlinedOperands,
                                 ArrayRef<Value *> resultsToReplace,
                                 llvm::Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  auto *entryBlock = &src->front();
  if (inlinedOperands.size() != entryBlock->getNumArguments())
    return failure();

  // Map the provided call operands to the arguments of the region.
  BlockAndValueMapping mapper;
  for (unsigned i = 0, e = inlinedOperands.size(); i != e; ++i) {
    // Verify that the types of the provided values match the function argument
    // types.
    BlockArgument *regionArg = entryBlock->getArgument(i);
    if (inlinedOperands[i]->getType() != regionArg->getType())
      return failure();
    mapper.map(regionArg, inlinedOperands[i]);
  }

  // Call into the main region inliner function.
  return inlineRegion(interface, src, inlinePoint, mapper, resultsToReplace,
                      inlineLoc, shouldCloneInlinedRegion);
}

/// This function inlines a FuncOp into another. This function returns failure
/// if it is not possible to inline this FuncOp. If the function returned
/// failure, then no changes to the module have been made.
///
/// Note that this only does one level of inlining.  For example, if the
/// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now
/// exists in the instruction stream.  Similarly this will inline a recursive
/// FuncOp by one level.
///
LogicalResult mlir::inlineFunction(InlinerInterface &interface, FuncOp callee,
                                   Operation *inlinePoint,
                                   ArrayRef<Value *> callOperands,
                                   ArrayRef<Value *> callResults,
                                   Location inlineLoc) {
  // We don't inline if the provided callee function is a declaration.
  assert(callee && "expected valid function to inline");
  if (callee.isExternal())
    return failure();

  // Verify that the provided arguments match the function arguments.
  if (callOperands.size() != callee.getNumArguments())
    return failure();

  // Verify that the provided values to replace match the function results.
  auto funcResultTypes = callee.getType().getResults();
  if (callResults.size() != funcResultTypes.size())
    return failure();
  for (unsigned i = 0, e = callResults.size(); i != e; ++i)
    if (callResults[i]->getType() != funcResultTypes[i])
      return failure();

  // Call into the main region inliner function.
  return inlineRegion(interface, &callee.getBody(), inlinePoint, callOperands,
                      callResults, inlineLoc);
}
