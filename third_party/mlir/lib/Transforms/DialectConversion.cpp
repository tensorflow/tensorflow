//===- DialectConversion.cpp - MLIR dialect conversion generic pass -------===//
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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

#define DEBUG_TYPE "dialect-conversion"

/// Recursively collect all of the operations to convert from within 'region'.
/// If 'target' is nonnull, operations that are recursively legal have their
/// regions pre-filtered to avoid considering them for legalization.
static LogicalResult
computeConversionSet(llvm::iterator_range<Region::iterator> region,
                     Location regionLoc, std::vector<Operation *> &toConvert,
                     ConversionTarget *target = nullptr) {
  if (llvm::empty(region))
    return success();

  // Traverse starting from the entry block.
  SmallVector<Block *, 16> worklist(1, &*region.begin());
  DenseSet<Block *> visitedBlocks;
  visitedBlocks.insert(worklist.front());
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    // Compute the conversion set of each of the nested operations.
    for (Operation &op : *block) {
      toConvert.emplace_back(&op);

      // Don't check this operation's children for conversion if the operation
      // is recursively legal.
      auto legalityInfo = target ? target->isLegal(&op)
                                 : Optional<ConversionTarget::LegalOpDetails>();
      if (legalityInfo && legalityInfo->isRecursivelyLegal)
        continue;
      for (auto &region : op.getRegions())
        computeConversionSet(region.getBlocks(), region.getLoc(), toConvert,
                             target);
    }

    // Recurse to children that haven't been visited.
    for (Block *succ : block->getSuccessors())
      if (visitedBlocks.insert(succ).second)
        worklist.push_back(succ);
  }

  // Check that all blocks in the region were visited.
  if (llvm::any_of(llvm::drop_begin(region, 1),
                   [&](Block &block) { return !visitedBlocks.count(&block); }))
    return emitError(regionLoc, "unreachable blocks were not converted");
  return success();
}

//===----------------------------------------------------------------------===//
// Multi-Level Value Mapper
//===----------------------------------------------------------------------===//

namespace {
/// This class wraps a BlockAndValueMapping to provide recursive lookup
/// functionality, i.e. we will traverse if the mapped value also has a mapping.
struct ConversionValueMapping {
  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return the provided value.
  Value *lookupOrDefault(Value *from) const;

  /// Map a value to the one provided.
  void map(Value *oldVal, Value *newVal) { mapping.map(oldVal, newVal); }

  /// Drop the last mapping for the given value.
  void erase(Value *value) { mapping.erase(value); }

private:
  /// Current value mappings.
  BlockAndValueMapping mapping;
};
} // end anonymous namespace

/// Lookup a mapped value within the map. If a mapping for the provided value
/// does not exist then return the provided value.
Value *ConversionValueMapping::lookupOrDefault(Value *from) const {
  // If this value had a valid mapping, unmap that value as well in the case
  // that it was also replaced.
  while (auto *mappedValue = mapping.lookupOrNull(from))
    from = mappedValue;
  return from;
}

//===----------------------------------------------------------------------===//
// ArgConverter
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a simple interface for converting the types of block
/// arguments. This is done by creating a new block that contains the new legal
/// types and extracting the block that contains the old illegal types to allow
/// for undoing pending rewrites in the case of failure.
struct ArgConverter {
  ArgConverter(TypeConverter *typeConverter, PatternRewriter &rewriter)
      : loc(rewriter.getUnknownLoc()), typeConverter(typeConverter),
        rewriter(rewriter) {}

  /// This structure contains the information pertaining to an argument that has
  /// been converted.
  struct ConvertedArgInfo {
    ConvertedArgInfo(unsigned newArgIdx, unsigned newArgSize,
                     Value *castValue = nullptr)
        : newArgIdx(newArgIdx), newArgSize(newArgSize), castValue(castValue) {}

    /// The start index of in the new argument list that contains arguments that
    /// replace the original.
    unsigned newArgIdx;

    /// The number of arguments that replaced the original argument.
    unsigned newArgSize;

    /// The cast value that was created to cast from the new arguments to the
    /// old. This only used if 'newArgSize' > 1.
    Value *castValue;
  };

  /// This structure contains information pertaining to a block that has had its
  /// signature converted.
  struct ConvertedBlockInfo {
    ConvertedBlockInfo(Block *origBlock) : origBlock(origBlock) {}

    /// The original block that was requested to have its signature converted.
    Block *origBlock;

    /// The conversion information for each of the arguments. The information is
    /// None if the argument was dropped during conversion.
    SmallVector<Optional<ConvertedArgInfo>, 1> argInfo;
  };

  /// Return if the signature of the given block has already been converted.
  bool hasBeenConverted(Block *block) const {
    return conversionInfo.count(block);
  }

  //===--------------------------------------------------------------------===//
  // Rewrite Application
  //===--------------------------------------------------------------------===//

  /// Erase any rewrites registered for the current block that is about to be
  /// removed. This merely drops the rewrites without undoing them.
  void notifyBlockRemoved(Block *block);

  /// Cleanup and undo any generated conversions for the arguments of block.
  /// This method replaces the new block with the original, reverting the IR to
  /// its original state.
  void discardRewrites(Block *block);

  /// Fully replace uses of the old arguments with the new, materializing cast
  /// operations as necessary.
  // FIXME(riverriddle) The 'mapping' parameter is only necessary because the
  // implementation of replaceUsesOfBlockArgument is buggy.
  void applyRewrites(ConversionValueMapping &mapping);

  //===--------------------------------------------------------------------===//
  // Conversion
  //===--------------------------------------------------------------------===//

  /// Attempt to convert the signature of the given block, if successful a new
  /// block is returned containing the new arguments. On failure, nullptr is
  /// returned.
  Block *convertSignature(Block *block, ConversionValueMapping &mapping);

  /// Apply the given signature conversion on the given block. The new block
  /// containing the updated signature is returned.
  Block *applySignatureConversion(
      Block *block, TypeConverter::SignatureConversion &signatureConversion,
      ConversionValueMapping &mapping);

  /// A collection of blocks that have had their arguments converted.
  llvm::MapVector<Block *, ConvertedBlockInfo> conversionInfo;

  /// An instance of the unknown location that is used when materializing
  /// conversions.
  Location loc;

  /// The type converter to use when changing types.
  TypeConverter *typeConverter;

  /// The pattern rewriter to use when materializing conversions.
  PatternRewriter &rewriter;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Rewrite Application

void ArgConverter::notifyBlockRemoved(Block *block) {
  auto it = conversionInfo.find(block);
  if (it == conversionInfo.end())
    return;

  // Drop all uses of the original arguments and delete the original block.
  Block *origBlock = it->second.origBlock;
  for (BlockArgument *arg : origBlock->getArguments())
    arg->dropAllUses();
  delete origBlock;

  conversionInfo.erase(it);
}

void ArgConverter::discardRewrites(Block *block) {
  auto it = conversionInfo.find(block);
  if (it == conversionInfo.end())
    return;
  Block *origBlock = it->second.origBlock;

  // Drop all uses of the new block arguments and replace uses of the new block.
  for (int i = block->getNumArguments() - 1; i >= 0; --i)
    block->getArgument(i)->dropAllUses();
  block->replaceAllUsesWith(origBlock);

  // Move the operations back the original block and the delete the new block.
  origBlock->getOperations().splice(origBlock->end(), block->getOperations());
  origBlock->insertBefore(block);
  block->erase();

  conversionInfo.erase(it);
}

void ArgConverter::applyRewrites(ConversionValueMapping &mapping) {
  for (auto &info : conversionInfo) {
    Block *newBlock = info.first;
    ConvertedBlockInfo &blockInfo = info.second;
    Block *origBlock = blockInfo.origBlock;

    // Process the remapping for each of the original arguments.
    for (unsigned i = 0, e = origBlock->getNumArguments(); i != e; ++i) {
      Optional<ConvertedArgInfo> &argInfo = blockInfo.argInfo[i];
      BlockArgument *origArg = origBlock->getArgument(i);

      // Handle the case of a 1->0 value mapping.
      if (!argInfo) {
        // If a replacement value was given for this argument, use that to
        // replace all uses.
        auto argReplacementValue = mapping.lookupOrDefault(origArg);
        if (argReplacementValue != origArg) {
          origArg->replaceAllUsesWith(argReplacementValue);
          continue;
        }
        // If there are any dangling uses then replace the argument with one
        // generated by the type converter. This is necessary as the cast must
        // persist in the IR after conversion.
        if (!origArg->use_empty()) {
          rewriter.setInsertionPointToStart(newBlock);
          auto *newOp = typeConverter->materializeConversion(
              rewriter, origArg->getType(), llvm::None, loc);
          origArg->replaceAllUsesWith(newOp->getResult(0));
        }
        continue;
      }

      // If mapping is 1-1, replace the remaining uses and drop the cast
      // operation.
      // FIXME(riverriddle) This should check that the result type and operand
      // type are the same, otherwise it should force a conversion to be
      // materialized.
      if (argInfo->newArgSize == 1) {
        origArg->replaceAllUsesWith(
            mapping.lookupOrDefault(newBlock->getArgument(argInfo->newArgIdx)));
        continue;
      }

      // Otherwise this is a 1->N value mapping.
      Value *castValue = argInfo->castValue;
      assert(argInfo->newArgSize > 1 && castValue && "expected 1->N mapping");

      // If the argument is still used, replace it with the generated cast.
      if (!origArg->use_empty())
        origArg->replaceAllUsesWith(mapping.lookupOrDefault(castValue));

      // If all users of the cast were removed, we can drop it. Otherwise, keep
      // the operation alive and let the user handle any remaining usages.
      if (castValue->use_empty())
        castValue->getDefiningOp()->erase();
    }

    // Drop the original block now the rewrites were applied.
    delete origBlock;
  }
}

//===----------------------------------------------------------------------===//
// Conversion

Block *ArgConverter::convertSignature(Block *block,
                                      ConversionValueMapping &mapping) {
  if (auto conversion = typeConverter->convertBlockSignature(block))
    return applySignatureConversion(block, *conversion, mapping);
  return nullptr;
}

Block *ArgConverter::applySignatureConversion(
    Block *block, TypeConverter::SignatureConversion &signatureConversion,
    ConversionValueMapping &mapping) {
  // If no arguments are being changed or added, there is nothing to do.
  unsigned origArgCount = block->getNumArguments();
  auto convertedTypes = signatureConversion.getConvertedTypes();
  if (origArgCount == 0 && convertedTypes.empty())
    return block;

  // Split the block at the beginning to get a new block to use for the updated
  // signature.
  Block *newBlock = block->splitBlock(block->begin());
  block->replaceAllUsesWith(newBlock);

  SmallVector<Value *, 4> newArgRange(newBlock->addArguments(convertedTypes));
  ArrayRef<Value *> newArgs(newArgRange);

  // Remap each of the original arguments as determined by the signature
  // conversion.
  ConvertedBlockInfo info(block);
  info.argInfo.resize(origArgCount);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newBlock);
  for (unsigned i = 0; i != origArgCount; ++i) {
    auto inputMap = signatureConversion.getInputMapping(i);
    if (!inputMap)
      continue;
    BlockArgument *origArg = block->getArgument(i);

    // If inputMap->replacementValue is not nullptr, then the argument is
    // dropped and a replacement value is provided to be the remappedValue.
    if (inputMap->replacementValue) {
      assert(inputMap->size == 0 &&
             "invalid to provide a replacement value when the argument isn't "
             "dropped");
      mapping.map(origArg, inputMap->replacementValue);
      continue;
    }

    // If this is a 1->1 mapping, then map the argument directly.
    if (inputMap->size == 1) {
      mapping.map(origArg, newArgs[inputMap->inputNo]);
      info.argInfo[i] = ConvertedArgInfo(inputMap->inputNo, inputMap->size);
      continue;
    }

    // Otherwise, this is a 1->N mapping. Call into the provided type converter
    // to pack the new values.
    auto replArgs = newArgs.slice(inputMap->inputNo, inputMap->size);
    Operation *cast = typeConverter->materializeConversion(
        rewriter, origArg->getType(), replArgs, loc);
    assert(cast->getNumResults() == 1 &&
           cast->getNumOperands() == replArgs.size());
    mapping.map(origArg, cast->getResult(0));
    info.argInfo[i] =
        ConvertedArgInfo(inputMap->inputNo, inputMap->size, cast->getResult(0));
  }

  // Remove the original block from the region and return the new one.
  newBlock->getParent()->getBlocks().remove(block);
  conversionInfo.insert({newBlock, std::move(info)});
  return newBlock;
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriterImpl
//===----------------------------------------------------------------------===//
namespace {
/// This class contains a snapshot of the current conversion rewriter state.
/// This is useful when saving and undoing a set of rewrites.
struct RewriterState {
  RewriterState(unsigned numCreatedOperations, unsigned numReplacements,
                unsigned numBlockActions, unsigned numIgnoredOperations)
      : numCreatedOperations(numCreatedOperations),
        numReplacements(numReplacements), numBlockActions(numBlockActions),
        numIgnoredOperations(numIgnoredOperations) {}

  /// The current number of created operations.
  unsigned numCreatedOperations;

  /// The current number of replacements queued.
  unsigned numReplacements;

  /// The current number of block actions performed.
  unsigned numBlockActions;

  /// The current number of ignored operations.
  unsigned numIgnoredOperations;
};
} // end anonymous namespace

namespace mlir {
namespace detail {
struct ConversionPatternRewriterImpl {
  /// This class represents one requested operation replacement via 'replaceOp'.
  struct OpReplacement {
    OpReplacement() = default;
    OpReplacement(Operation *op, ValueRange newValues)
        : op(op), newValues(newValues.begin(), newValues.end()) {}

    Operation *op;
    SmallVector<Value *, 2> newValues;
  };

  /// The kind of the block action performed during the rewrite.  Actions can be
  /// undone if the conversion fails.
  enum class BlockActionKind { Create, Move, Split, TypeConversion };

  /// Original position of the given block in its parent region.  We cannot use
  /// a region iterator because it could have been invalidated by other region
  /// operations since the position was stored.
  struct BlockPosition {
    Region *region;
    Region::iterator::difference_type position;
  };

  /// The storage class for an undoable block action (one of BlockActionKind),
  /// contains the information necessary to undo this action.
  struct BlockAction {
    static BlockAction getCreate(Block *block) {
      return {BlockActionKind::Create, block, {}};
    }
    static BlockAction getMove(Block *block, BlockPosition originalPos) {
      return {BlockActionKind::Move, block, {originalPos}};
    }
    static BlockAction getSplit(Block *block, Block *originalBlock) {
      BlockAction action{BlockActionKind::Split, block, {}};
      action.originalBlock = originalBlock;
      return action;
    }
    static BlockAction getTypeConversion(Block *block) {
      return BlockAction{BlockActionKind::TypeConversion, block, {}};
    }

    // The action kind.
    BlockActionKind kind;

    // A pointer to the block that was created by the action.
    Block *block;

    union {
      // In use if kind == BlockActionKind::Move and contains a pointer to the
      // region that originally contained the block as well as the position of
      // the block in that region.
      BlockPosition originalPosition;
      // In use if kind == BlockActionKind::Split and contains a pointer to the
      // block that was split into two parts.
      Block *originalBlock;
    };
  };

  ConversionPatternRewriterImpl(PatternRewriter &rewriter,
                                TypeConverter *converter)
      : argConverter(converter, rewriter) {}

  /// Return the current state of the rewriter.
  RewriterState getCurrentState();

  /// Reset the state of the rewriter to a previously saved point.
  void resetState(RewriterState state);

  /// Undo the block actions (motions, splits) one by one in reverse order until
  /// "numActionsToKeep" actions remains.
  void undoBlockActions(unsigned numActionsToKeep = 0);

  /// Cleanup and destroy any generated rewrite operations. This method is
  /// invoked when the conversion process fails.
  void discardRewrites();

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites();

  /// Convert the signature of the given block.
  LogicalResult convertBlockSignature(Block *block);

  /// Apply a signature conversion on the given region.
  Block *
  applySignatureConversion(Region *region,
                           TypeConverter::SignatureConversion &conversion);

  /// PatternRewriter hook for replacing the results of an operation.
  void replaceOp(Operation *op, ValueRange newValues,
                 ValueRange valuesToRemoveIfDead);

  /// Notifies that a block was split.
  void notifySplitBlock(Block *block, Block *continuation);

  /// Notifies that the blocks of a region are about to be moved.
  void notifyRegionIsBeingInlinedBefore(Region &region, Region &parent,
                                        Region::iterator before);

  /// Notifies that the blocks of a region were cloned into another.
  void
  notifyRegionWasClonedBefore(llvm::iterator_range<Region::iterator> &blocks,
                              Location origRegionLoc);

  /// Remap the given operands to those with potentially different types.
  void remapValues(Operation::operand_range operands,
                   SmallVectorImpl<Value *> &remapped);

  /// Returns true if the given operation is ignored, and does not need to be
  /// converted.
  bool isOpIgnored(Operation *op) const;

  /// Recursively marks the nested operations under 'op' as ignored. This
  /// removes them from being considered for legalization.
  void markNestedOpsIgnored(Operation *op);

  // Mapping between replaced values that differ in type. This happens when
  // replacing a value with one of a different type.
  ConversionValueMapping mapping;

  /// Utility used to convert block arguments.
  ArgConverter argConverter;

  /// Ordered vector of all of the newly created operations during conversion.
  std::vector<Operation *> createdOps;

  /// Ordered vector of any requested operation replacements.
  SmallVector<OpReplacement, 4> replacements;

  /// Ordered list of block operations (creations, splits, motions).
  SmallVector<BlockAction, 4> blockActions;

  /// A set of operations that have been erased/replaced/etc that should no
  /// longer be considered for legalization. This is not meant to be an
  /// exhaustive list of all operations, but the minimal set that can be used to
  /// detect if a given operation should be `ignored`. For example, we may add
  /// the operations that define non-empty regions to the set, but not any of
  /// the others. This simplifies the amount of memory needed as we can query if
  /// the parent operation was ignored.
  llvm::SetVector<Operation *> ignoredOps;
};
} // end namespace detail
} // end namespace mlir

RewriterState ConversionPatternRewriterImpl::getCurrentState() {
  return RewriterState(createdOps.size(), replacements.size(),
                       blockActions.size(), ignoredOps.size());
}

void ConversionPatternRewriterImpl::resetState(RewriterState state) {
  // Undo any block actions.
  undoBlockActions(state.numBlockActions);

  // Reset any replaced operations and undo any saved mappings.
  for (auto &repl : llvm::drop_begin(replacements, state.numReplacements))
    for (auto *result : repl.op->getResults())
      mapping.erase(result);
  replacements.resize(state.numReplacements);

  // Pop all of the newly created operations.
  while (createdOps.size() != state.numCreatedOperations) {
    createdOps.back()->erase();
    createdOps.pop_back();
  }

  // Pop all of the recorded ignored operations that are no longer valid.
  while (ignoredOps.size() != state.numIgnoredOperations)
    ignoredOps.pop_back();
}

void ConversionPatternRewriterImpl::undoBlockActions(
    unsigned numActionsToKeep) {
  for (auto &action :
       llvm::reverse(llvm::drop_begin(blockActions, numActionsToKeep))) {
    switch (action.kind) {
    // Delete the created block.
    case BlockActionKind::Create: {
      // Unlink all of the operations within this block, they will be deleted
      // separately.
      auto &blockOps = action.block->getOperations();
      while (!blockOps.empty())
        blockOps.remove(blockOps.begin());
      action.block->dropAllDefinedValueUses();
      action.block->erase();
      break;
    }
    // Move the block back to its original position.
    case BlockActionKind::Move: {
      Region *originalRegion = action.originalPosition.region;
      originalRegion->getBlocks().splice(
          std::next(originalRegion->begin(), action.originalPosition.position),
          action.block->getParent()->getBlocks(), action.block);
      break;
    }
    // Merge back the block that was split out.
    case BlockActionKind::Split: {
      action.originalBlock->getOperations().splice(
          action.originalBlock->end(), action.block->getOperations());
      action.block->dropAllDefinedValueUses();
      action.block->erase();
      break;
    }
    // Undo the type conversion.
    case BlockActionKind::TypeConversion: {
      argConverter.discardRewrites(action.block);
      break;
    }
    }
  }
  blockActions.resize(numActionsToKeep);
}

void ConversionPatternRewriterImpl::discardRewrites() {
  undoBlockActions();

  // Remove any newly created ops.
  for (auto *op : llvm::reverse(createdOps))
    op->erase();
}

void ConversionPatternRewriterImpl::applyRewrites() {
  // Apply all of the rewrites replacements requested during conversion.
  for (auto &repl : replacements) {
    for (unsigned i = 0, e = repl.newValues.size(); i != e; ++i) {
      if (auto *newValue = repl.newValues[i])
        repl.op->getResult(i)->replaceAllUsesWith(
            mapping.lookupOrDefault(newValue));
    }

    // If this operation defines any regions, drop any pending argument
    // rewrites.
    if (argConverter.typeConverter && repl.op->getNumRegions()) {
      for (auto &region : repl.op->getRegions())
        for (auto &block : region)
          argConverter.notifyBlockRemoved(&block);
    }
  }

  // In a second pass, erase all of the replaced operations in reverse. This
  // allows processing nested operations before their parent region is
  // destroyed.
  for (auto &repl : llvm::reverse(replacements))
    repl.op->erase();

  argConverter.applyRewrites(mapping);
}

LogicalResult
ConversionPatternRewriterImpl::convertBlockSignature(Block *block) {
  // Check to see if this block should not be converted:
  // * There is no type converter.
  // * The block has already been converted.
  // * This is an entry block, these are converted explicitly via patterns.
  if (!argConverter.typeConverter || argConverter.hasBeenConverted(block) ||
      !block->getParent() || block->isEntryBlock())
    return success();

  // Otherwise, try to convert the block signature.
  Block *newBlock = argConverter.convertSignature(block, mapping);
  if (newBlock)
    blockActions.push_back(BlockAction::getTypeConversion(newBlock));
  return success(newBlock);
}

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion) {
  if (!region->empty()) {
    Block *newEntry = argConverter.applySignatureConversion(
        &region->front(), conversion, mapping);
    blockActions.push_back(BlockAction::getTypeConversion(newEntry));
    return newEntry;
  }
  return nullptr;
}

void ConversionPatternRewriterImpl::replaceOp(Operation *op,
                                              ValueRange newValues,
                                              ValueRange valuesToRemoveIfDead) {
  assert(newValues.size() == op->getNumResults());

  // Create mappings for each of the new result values.
  for (unsigned i = 0, e = newValues.size(); i < e; ++i)
    if (auto *repl = newValues[i])
      mapping.map(op->getResult(i), repl);

  // Record the requested operation replacement.
  replacements.emplace_back(op, newValues);

  /// Mark this operation as recursively ignored so that we don't need to
  /// convert any nested operations.
  markNestedOpsIgnored(op);
}

void ConversionPatternRewriterImpl::notifySplitBlock(Block *block,
                                                     Block *continuation) {
  blockActions.push_back(BlockAction::getSplit(continuation, block));
}

void ConversionPatternRewriterImpl::notifyRegionIsBeingInlinedBefore(
    Region &region, Region &parent, Region::iterator before) {
  for (auto &pair : llvm::enumerate(region)) {
    Block &block = pair.value();
    unsigned position = pair.index();
    blockActions.push_back(BlockAction::getMove(&block, {&region, position}));
  }
}

void ConversionPatternRewriterImpl::notifyRegionWasClonedBefore(
    llvm::iterator_range<Region::iterator> &blocks, Location origRegionLoc) {
  for (Block &block : blocks)
    blockActions.push_back(BlockAction::getCreate(&block));

  // Compute the conversion set for the inlined region.
  auto result = computeConversionSet(blocks, origRegionLoc, createdOps);

  // This original region has already had its conversion set computed, so there
  // shouldn't be any new failures.
  (void)result;
  assert(succeeded(result) && "expected region to have no unreachable blocks");
}

void ConversionPatternRewriterImpl::remapValues(
    Operation::operand_range operands, SmallVectorImpl<Value *> &remapped) {
  remapped.reserve(llvm::size(operands));
  for (Value *operand : operands)
    remapped.push_back(mapping.lookupOrDefault(operand));
}

bool ConversionPatternRewriterImpl::isOpIgnored(Operation *op) const {
  // Check to see if this operation or its parent were ignored.
  return ignoredOps.count(op) || ignoredOps.count(op->getParentOp());
}

void ConversionPatternRewriterImpl::markNestedOpsIgnored(Operation *op) {
  // Walk this operation and collect nested operations that define non-empty
  // regions. We mark such operations as 'ignored' so that we know we don't have
  // to convert them, or their nested ops.
  if (op->getNumRegions() == 0)
    return;
  op->walk([&](Operation *op) {
    if (llvm::any_of(op->getRegions(),
                     [](Region &region) { return !region.empty(); }))
      ignoredOps.insert(op);
  });
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriter
//===----------------------------------------------------------------------===//

ConversionPatternRewriter::ConversionPatternRewriter(MLIRContext *ctx,
                                                     TypeConverter *converter)
    : PatternRewriter(ctx),
      impl(new detail::ConversionPatternRewriterImpl(*this, converter)) {}
ConversionPatternRewriter::~ConversionPatternRewriter() {}

/// PatternRewriter hook for replacing the results of an operation.
void ConversionPatternRewriter::replaceOp(Operation *op, ValueRange newValues,
                                          ValueRange valuesToRemoveIfDead) {
  LLVM_DEBUG(llvm::dbgs() << "** Replacing operation : " << op->getName()
                          << "\n");
  impl->replaceOp(op, newValues, valuesToRemoveIfDead);
}

/// PatternRewriter hook for erasing a dead operation. The uses of this
/// operation *must* be made dead by the end of the conversion process,
/// otherwise an assert will be issued.
void ConversionPatternRewriter::eraseOp(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "** Erasing operation : " << op->getName()
                          << "\n");
  SmallVector<Value *, 1> nullRepls(op->getNumResults(), nullptr);
  impl->replaceOp(op, nullRepls, /*valuesToRemoveIfDead=*/llvm::None);
}

/// Apply a signature conversion to the entry block of the given region.
Block *ConversionPatternRewriter::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion) {
  return impl->applySignatureConversion(region, conversion);
}

void ConversionPatternRewriter::replaceUsesOfBlockArgument(BlockArgument *from,
                                                           Value *to) {
  for (auto &u : from->getUses()) {
    if (u.getOwner() == to->getDefiningOp())
      continue;
    u.getOwner()->replaceUsesOfWith(from, to);
  }
  impl->mapping.map(impl->mapping.lookupOrDefault(from), to);
}

/// Return the converted value that replaces 'key'. Return 'key' if there is
/// no such a converted value.
Value *ConversionPatternRewriter::getRemappedValue(Value *key) {
  return impl->mapping.lookupOrDefault(key);
}

/// PatternRewriter hook for splitting a block into two parts.
Block *ConversionPatternRewriter::splitBlock(Block *block,
                                             Block::iterator before) {
  auto *continuation = PatternRewriter::splitBlock(block, before);
  impl->notifySplitBlock(block, continuation);
  return continuation;
}

/// PatternRewriter hook for merging a block into another.
void ConversionPatternRewriter::mergeBlocks(Block *source, Block *dest,
                                            ValueRange argValues) {
  // TODO(riverriddle) This requires fixing the implementation of
  // 'replaceUsesOfBlockArgument', which currently isn't undoable.
  llvm_unreachable("block merging updates are currently not supported");
}

/// PatternRewriter hook for moving blocks out of a region.
void ConversionPatternRewriter::inlineRegionBefore(Region &region,
                                                   Region &parent,
                                                   Region::iterator before) {
  impl->notifyRegionIsBeingInlinedBefore(region, parent, before);
  PatternRewriter::inlineRegionBefore(region, parent, before);
}

/// PatternRewriter hook for cloning blocks of one region into another.
void ConversionPatternRewriter::cloneRegionBefore(
    Region &region, Region &parent, Region::iterator before,
    BlockAndValueMapping &mapping) {
  if (region.empty())
    return;
  PatternRewriter::cloneRegionBefore(region, parent, before, mapping);

  // Collect the range of the cloned blocks.
  auto clonedBeginIt = mapping.lookup(&region.front())->getIterator();
  auto clonedBlocks = llvm::make_range(clonedBeginIt, before);
  impl->notifyRegionWasClonedBefore(clonedBlocks, region.getLoc());
}

/// PatternRewriter hook for creating a new operation.
Operation *ConversionPatternRewriter::insert(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "** Inserting operation : " << op->getName()
                          << "\n");
  impl->createdOps.push_back(op);
  return OpBuilder::insert(op);
}

/// PatternRewriter hook for updating the root operation in-place.
void ConversionPatternRewriter::notifyRootUpdated(Operation *op) {
  // The rewriter caches changes to the IR to allow for operating in-place and
  // backtracking. The rewriter is currently not capable of backtracking
  // in-place modifications.
  llvm_unreachable("in-place operation updates are not supported");
}

/// Return a reference to the internal implementation.
detail::ConversionPatternRewriterImpl &ConversionPatternRewriter::getImpl() {
  return *impl;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Attempt to match and rewrite the IR root at the specified operation.
PatternMatchResult
ConversionPattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  SmallVector<Value *, 4> operands;
  auto &dialectRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
  dialectRewriter.getImpl().remapValues(op->getOperands(), operands);

  // If this operation has no successors, invoke the rewrite directly.
  if (op->getNumSuccessors() == 0)
    return matchAndRewrite(op, operands, dialectRewriter);

  // Otherwise, we need to remap the successors.
  SmallVector<Block *, 2> destinations;
  destinations.reserve(op->getNumSuccessors());

  SmallVector<ArrayRef<Value *>, 2> operandsPerDestination;
  unsigned firstSuccessorOperand = op->getSuccessorOperandIndex(0);
  for (unsigned i = 0, seen = 0, e = op->getNumSuccessors(); i < e; ++i) {
    destinations.push_back(op->getSuccessor(i));

    // Lookup the successors operands.
    unsigned n = op->getNumSuccessorOperands(i);
    operandsPerDestination.push_back(
        llvm::makeArrayRef(operands.data() + firstSuccessorOperand + seen, n));
    seen += n;
  }

  // Rewrite the operation.
  return matchAndRewrite(
      op,
      llvm::makeArrayRef(operands.data(),
                         operands.data() + firstSuccessorOperand),
      destinations, operandsPerDestination, dialectRewriter);
}

//===----------------------------------------------------------------------===//
// OperationLegalizer
//===----------------------------------------------------------------------===//

namespace {
/// A set of rewrite patterns that can be used to legalize a given operation.
using LegalizationPatterns = SmallVector<RewritePattern *, 1>;

/// This class defines a recursive operation legalizer.
class OperationLegalizer {
public:
  using LegalizationAction = ConversionTarget::LegalizationAction;

  OperationLegalizer(ConversionTarget &targetInfo,
                     const OwningRewritePatternList &patterns)
      : target(targetInfo) {
    buildLegalizationGraph(patterns);
    computeLegalizationGraphBenefit();
  }

  /// Returns if the given operation is known to be illegal on the target.
  bool isIllegal(Operation *op) const;

  /// Attempt to legalize the given operation. Returns success if the operation
  /// was legalized, failure otherwise.
  LogicalResult legalize(Operation *op, ConversionPatternRewriter &rewriter);

  /// Returns the conversion target in use by the legalizer.
  ConversionTarget &getTarget() { return target; }

private:
  /// Attempt to legalize the given operation by applying the provided pattern.
  /// Returns success if the operation was legalized, failure otherwise.
  LogicalResult legalizePattern(Operation *op, RewritePattern *pattern,
                                ConversionPatternRewriter &rewriter);

  /// Build an optimistic legalization graph given the provided patterns. This
  /// function populates 'legalizerPatterns' with the operations that are not
  /// directly legal, but may be transitively legal for the current target given
  /// the provided patterns.
  void buildLegalizationGraph(const OwningRewritePatternList &patterns);

  /// Compute the benefit of each node within the computed legalization graph.
  /// This orders the patterns within 'legalizerPatterns' based upon two
  /// criteria:
  ///  1) Prefer patterns that have the lowest legalization depth, i.e.
  ///     represent the more direct mapping to the target.
  ///  2) When comparing patterns with the same legalization depth, prefer the
  ///     pattern with the highest PatternBenefit. This allows for users to
  ///     prefer specific legalizations over others.
  void computeLegalizationGraphBenefit();

  /// The current set of patterns that have been applied.
  llvm::SmallPtrSet<RewritePattern *, 8> appliedPatterns;

  /// The set of legality information for operations transitively supported by
  /// the target.
  DenseMap<OperationName, LegalizationPatterns> legalizerPatterns;

  /// The legalization information provided by the target.
  ConversionTarget &target;
};
} // namespace

bool OperationLegalizer::isIllegal(Operation *op) const {
  // Check if the target explicitly marked this operation as illegal.
  return target.getOpAction(op->getName()) == LegalizationAction::Illegal;
}

LogicalResult
OperationLegalizer::legalize(Operation *op,
                             ConversionPatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Legalizing operation : " << op->getName()
                          << "\n");

  // Check if this operation is legal on the target.
  if (auto legalityInfo = target.isLegal(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << "-- Success : Operation marked legal by the target\n");
    // If this operation is recursively legal, mark its children as ignored so
    // that we don't consider them for legalization.
    if (legalityInfo->isRecursivelyLegal) {
      LLVM_DEBUG(llvm::dbgs() << "-- Success : Operation is recursively legal; "
                                 "Skipping internals\n");
      rewriter.getImpl().markNestedOpsIgnored(op);
    }
    return success();
  }

  // Check to see if the operation is ignored and doesn't need to be converted.
  if (rewriter.getImpl().isOpIgnored(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << "-- Success : Operation marked ignored during conversion\n");
    return success();
  }

  // Otherwise, we need to apply a legalization pattern to this operation.
  auto it = legalizerPatterns.find(op->getName());
  if (it == legalizerPatterns.end()) {
    LLVM_DEBUG(llvm::dbgs() << "-- FAIL : no known legalization path.\n");
    return failure();
  }

  // The patterns are sorted by expected benefit, so try to apply each in-order.
  for (auto *pattern : it->second)
    if (succeeded(legalizePattern(op, pattern, rewriter)))
      return success();

  LLVM_DEBUG(llvm::dbgs() << "-- FAIL : no matched legalization pattern.\n");
  return failure();
}

LogicalResult
OperationLegalizer::legalizePattern(Operation *op, RewritePattern *pattern,
                                    ConversionPatternRewriter &rewriter) {
  LLVM_DEBUG({
    llvm::dbgs() << "-* Applying rewrite pattern '" << op->getName() << " -> (";
    interleaveComma(pattern->getGeneratedOps(), llvm::dbgs());
    llvm::dbgs() << ")'.\n";
  });

  // Ensure that we don't cycle by not allowing the same pattern to be
  // applied twice in the same recursion stack.
  // TODO(riverriddle) We could eventually converge, but that requires more
  // complicated analysis.
  if (!appliedPatterns.insert(pattern).second) {
    LLVM_DEBUG(llvm::dbgs() << "-- FAIL: Pattern was already applied.\n");
    return failure();
  }

  auto &rewriterImpl = rewriter.getImpl();
  RewriterState curState = rewriterImpl.getCurrentState();
  auto cleanupFailure = [&] {
    // Reset the rewriter state and pop this pattern.
    rewriterImpl.resetState(curState);
    appliedPatterns.erase(pattern);
    return failure();
  };

  // Try to rewrite with the given pattern.
  rewriter.setInsertionPoint(op);
  if (!pattern->matchAndRewrite(op, rewriter)) {
    LLVM_DEBUG(llvm::dbgs() << "-- FAIL: Pattern failed to match.\n");
    return cleanupFailure();
  }

  // If the pattern moved or created any blocks, try to legalize their types.
  // This ensures that the types of the block arguments are legal for the region
  // they were moved into.
  for (unsigned i = curState.numBlockActions,
                e = rewriterImpl.blockActions.size();
       i != e; ++i) {
    auto &action = rewriterImpl.blockActions[i];
    if (action.kind ==
        ConversionPatternRewriterImpl::BlockActionKind::TypeConversion)
      continue;

    // Convert the block signature.
    if (failed(rewriterImpl.convertBlockSignature(action.block))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "-- FAIL: failed to convert types of moved block.\n");
      return cleanupFailure();
    }
  }

  // Check all of the replacements to ensure that the pattern actually replaced
  // the root operation. We also mark any other replaced ops as 'dead' so that
  // we don't try to legalize them later.
  bool replacedRoot = false;
  for (unsigned i = curState.numReplacements,
                e = rewriterImpl.replacements.size();
       i != e; ++i) {
    Operation *replacedOp = rewriterImpl.replacements[i].op;
    if (replacedOp == op)
      replacedRoot = true;
    else
      rewriterImpl.ignoredOps.insert(replacedOp);
  }
  assert(replacedRoot && "expected pattern to replace the root operation");
  (void)replacedRoot;

  // Recursively legalize each of the new operations.
  for (unsigned i = curState.numCreatedOperations,
                e = rewriterImpl.createdOps.size();
       i != e; ++i) {
    Operation *op = rewriterImpl.createdOps[i];
    if (failed(legalize(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs() << "-- FAIL: Generated operation '"
                              << op->getName() << "' was illegal.\n");
      return cleanupFailure();
    }
  }

  appliedPatterns.erase(pattern);
  return success();
}

void OperationLegalizer::buildLegalizationGraph(
    const OwningRewritePatternList &patterns) {
  // A mapping between an operation and a set of operations that can be used to
  // generate it.
  DenseMap<OperationName, SmallPtrSet<OperationName, 2>> parentOps;
  // A mapping between an operation and any currently invalid patterns it has.
  DenseMap<OperationName, SmallPtrSet<RewritePattern *, 2>> invalidPatterns;
  // A worklist of patterns to consider for legality.
  llvm::SetVector<RewritePattern *> patternWorklist;

  // Build the mapping from operations to the parent ops that may generate them.
  for (auto &pattern : patterns) {
    auto root = pattern->getRootKind();

    // Skip operations that are always known to be legal.
    if (target.getOpAction(root) == LegalizationAction::Legal)
      continue;

    // Add this pattern to the invalid set for the root op and record this root
    // as a parent for any generated operations.
    invalidPatterns[root].insert(pattern.get());
    for (auto op : pattern->getGeneratedOps())
      parentOps[op].insert(root);

    // Add this pattern to the worklist.
    patternWorklist.insert(pattern.get());
  }

  while (!patternWorklist.empty()) {
    auto *pattern = patternWorklist.pop_back_val();

    // Check to see if any of the generated operations are invalid.
    if (llvm::any_of(pattern->getGeneratedOps(), [&](OperationName op) {
          Optional<LegalizationAction> action = target.getOpAction(op);
          return !legalizerPatterns.count(op) &&
                 (!action || action == LegalizationAction::Illegal);
        }))
      continue;

    // Otherwise, if all of the generated operation are valid, this op is now
    // legal so add all of the child patterns to the worklist.
    legalizerPatterns[pattern->getRootKind()].push_back(pattern);
    invalidPatterns[pattern->getRootKind()].erase(pattern);

    // Add any invalid patterns of the parent operations to see if they have now
    // become legal.
    for (auto op : parentOps[pattern->getRootKind()])
      patternWorklist.set_union(invalidPatterns[op]);
  }
}

void OperationLegalizer::computeLegalizationGraphBenefit() {
  // The smallest pattern depth, when legalizing an operation.
  DenseMap<OperationName, unsigned> minPatternDepth;

  // Compute the minimum legalization depth for a given operation.
  std::function<unsigned(OperationName)> computeDepth = [&](OperationName op) {
    // Check for existing depth.
    auto depthIt = minPatternDepth.find(op);
    if (depthIt != minPatternDepth.end())
      return depthIt->second;

    // If a mapping for this operation does not exist, then this operation
    // is always legal. Return 0 as the depth for a directly legal operation.
    auto opPatternsIt = legalizerPatterns.find(op);
    if (opPatternsIt == legalizerPatterns.end() || opPatternsIt->second.empty())
      return 0u;

    // Initialize the depth to the maximum value.
    unsigned minDepth = std::numeric_limits<unsigned>::max();

    // Record this initial depth in case we encounter this op again when
    // recursively computing the depth.
    minPatternDepth.try_emplace(op, minDepth);

    // Compute the depth for each pattern used to legalize this operation.
    SmallVector<std::pair<RewritePattern *, unsigned>, 4> patternsByDepth;
    patternsByDepth.reserve(opPatternsIt->second.size());
    for (RewritePattern *pattern : opPatternsIt->second) {
      unsigned depth = 0;
      for (auto generatedOp : pattern->getGeneratedOps())
        depth = std::max(depth, computeDepth(generatedOp) + 1);
      patternsByDepth.emplace_back(pattern, depth);

      // Update the min depth for this operation.
      minDepth = std::min(minDepth, depth);
    }

    // Update the pattern depth.
    minPatternDepth[op] = minDepth;

    // If the operation only has one legalization pattern, there is no need to
    // sort them.
    if (patternsByDepth.size() == 1)
      return minDepth;

    // Sort the patterns by those likely to be the most beneficial.
    llvm::array_pod_sort(
        patternsByDepth.begin(), patternsByDepth.end(),
        [](const std::pair<RewritePattern *, unsigned> *lhs,
           const std::pair<RewritePattern *, unsigned> *rhs) {
          // First sort by the smaller pattern legalization depth.
          if (lhs->second != rhs->second)
            return llvm::array_pod_sort_comparator<unsigned>(&lhs->second,
                                                             &rhs->second);

          // Then sort by the larger pattern benefit.
          auto lhsBenefit = lhs->first->getBenefit();
          auto rhsBenefit = rhs->first->getBenefit();
          return llvm::array_pod_sort_comparator<PatternBenefit>(&rhsBenefit,
                                                                 &lhsBenefit);
        });

    // Update the legalization pattern to use the new sorted list.
    opPatternsIt->second.clear();
    for (auto &patternIt : patternsByDepth)
      opPatternsIt->second.push_back(patternIt.first);

    return minDepth;
  };

  // For each operation that is transitively legal, compute a cost for it.
  for (auto &opIt : legalizerPatterns)
    if (!minPatternDepth.count(opIt.first))
      computeDepth(opIt.first);
}

//===----------------------------------------------------------------------===//
// OperationConverter
//===----------------------------------------------------------------------===//
namespace {
enum OpConversionMode {
  // In this mode, the conversion will ignore failed conversions to allow
  // illegal operations to co-exist in the IR.
  Partial,

  // In this mode, all operations must be legal for the given target for the
  // conversion to succeed.
  Full,

  // In this mode, operations are analyzed for legality. No actual rewrites are
  // applied to the operations on success.
  Analysis,
};

// This class converts operations to a given conversion target via a set of
// rewrite patterns. The conversion behaves differently depending on the
// conversion mode.
struct OperationConverter {
  explicit OperationConverter(ConversionTarget &target,
                              const OwningRewritePatternList &patterns,
                              OpConversionMode mode,
                              DenseSet<Operation *> *legalizableOps = nullptr)
      : opLegalizer(target, patterns), mode(mode),
        legalizableOps(legalizableOps) {}

  /// Converts the given operations to the conversion target.
  LogicalResult convertOperations(ArrayRef<Operation *> ops,
                                  TypeConverter *typeConverter);

private:
  /// Converts an operation with the given rewriter.
  LogicalResult convert(ConversionPatternRewriter &rewriter, Operation *op);

  /// Converts the type signatures of the blocks nested within 'op'.
  LogicalResult convertBlockSignatures(ConversionPatternRewriter &rewriter,
                                       Operation *op);

  /// The legalizer to use when converting operations.
  OperationLegalizer opLegalizer;

  /// The conversion mode to use when legalizing operations.
  OpConversionMode mode;

  /// A set of pre-existing operations that were found to be legalizable to the
  /// target. This field is only used when mode == OpConversionMode::Analysis.
  DenseSet<Operation *> *legalizableOps;
};
} // end anonymous namespace

LogicalResult
OperationConverter::convertBlockSignatures(ConversionPatternRewriter &rewriter,
                                           Operation *op) {
  // Check to see if type signatures need to be converted.
  if (!rewriter.getImpl().argConverter.typeConverter)
    return success();

  for (auto &region : op->getRegions()) {
    for (auto &block : llvm::make_early_inc_range(region))
      if (failed(rewriter.getImpl().convertBlockSignature(&block)))
        return failure();
  }
  return success();
}

LogicalResult OperationConverter::convert(ConversionPatternRewriter &rewriter,
                                          Operation *op) {
  // Legalize the given operation.
  if (failed(opLegalizer.legalize(op, rewriter))) {
    // Handle the case of a failed conversion for each of the different modes.
    /// Full conversions expect all operations to be converted.
    if (mode == OpConversionMode::Full)
      return op->emitError()
             << "failed to legalize operation '" << op->getName() << "'";
    /// Partial conversions allow conversions to fail iff the operation was not
    /// explicitly marked as illegal.
    if (mode == OpConversionMode::Partial && opLegalizer.isIllegal(op))
      return op->emitError()
             << "failed to legalize operation '" << op->getName()
             << "' that was explicitly marked illegal";
  } else {
    /// Analysis conversions don't fail if any operations fail to legalize,
    /// they are only interested in the operations that were successfully
    /// legalized.
    if (mode == OpConversionMode::Analysis)
      legalizableOps->insert(op);

    // If legalization succeeded, convert the types any of the blocks within
    // this operation.
    if (failed(convertBlockSignatures(rewriter, op)))
      return failure();
  }
  return success();
}

LogicalResult
OperationConverter::convertOperations(ArrayRef<Operation *> ops,
                                      TypeConverter *typeConverter) {
  if (ops.empty())
    return success();
  ConversionTarget &target = opLegalizer.getTarget();

  /// Compute the set of operations and blocks to convert.
  std::vector<Operation *> toConvert;
  for (auto *op : ops) {
    toConvert.emplace_back(op);
    for (auto &region : op->getRegions())
      if (failed(computeConversionSet(region.getBlocks(), region.getLoc(),
                                      toConvert, &target)))
        return failure();
  }

  // Convert each operation and discard rewrites on failure.
  ConversionPatternRewriter rewriter(ops.front()->getContext(), typeConverter);
  for (auto *op : toConvert)
    if (failed(convert(rewriter, op)))
      return rewriter.getImpl().discardRewrites(), failure();

  // Otherwise, the body conversion succeeded. Apply rewrites if this is not an
  // analysis conversion.
  if (mode == OpConversionMode::Analysis)
    rewriter.getImpl().discardRewrites();
  else
    rewriter.getImpl().applyRewrites();
  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Remap an input of the original signature with a new set of types. The
/// new types are appended to the new signature conversion.
void TypeConverter::SignatureConversion::addInputs(unsigned origInputNo,
                                                   ArrayRef<Type> types) {
  assert(!types.empty() && "expected valid types");
  remapInput(origInputNo, /*newInputNo=*/argTypes.size(), types.size());
  addInputs(types);
}

/// Append new input types to the signature conversion, this should only be
/// used if the new types are not intended to remap an existing input.
void TypeConverter::SignatureConversion::addInputs(ArrayRef<Type> types) {
  assert(!types.empty() &&
         "1->0 type remappings don't need to be added explicitly");
  argTypes.append(types.begin(), types.end());
}

/// Remap an input of the original signature with a range of types in the
/// new signature.
void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    unsigned newInputNo,
                                                    unsigned newInputCount) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  assert(newInputCount != 0 && "expected valid input count");
  remappedInputs[origInputNo] =
      InputMapping{newInputNo, newInputCount, /*replacementValue=*/nullptr};
}

/// Remap an input of the original signature to another `replacementValue`
/// value. This would make the signature converter drop this argument.
void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    Value *replacementValue) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  remappedInputs[origInputNo] =
      InputMapping{origInputNo, /*size=*/0, replacementValue};
}

/// This hooks allows for converting a type.
LogicalResult TypeConverter::convertType(Type t,
                                         SmallVectorImpl<Type> &results) {
  if (auto newT = convertType(t)) {
    results.push_back(newT);
    return success();
  }
  return failure();
}

/// Convert the given set of types, filling 'results' as necessary. This
/// returns failure if the conversion of any of the types fails, success
/// otherwise.
LogicalResult TypeConverter::convertTypes(ArrayRef<Type> types,
                                          SmallVectorImpl<Type> &results) {
  for (auto type : types)
    if (failed(convertType(type, results)))
      return failure();
  return success();
}

/// Return true if the given type is legal for this type converter, i.e. the
/// type converts to itself.
bool TypeConverter::isLegal(Type type) {
  SmallVector<Type, 1> results;
  return succeeded(convertType(type, results)) && results.size() == 1 &&
         results.front() == type;
}

/// Return true if the inputs and outputs of the given function type are
/// legal.
bool TypeConverter::isSignatureLegal(FunctionType funcType) {
  return llvm::all_of(
      llvm::concat<const Type>(funcType.getInputs(), funcType.getResults()),
      [this](Type type) { return isLegal(type); });
}

/// This hook allows for converting a specific argument of a signature.
LogicalResult TypeConverter::convertSignatureArg(unsigned inputNo, Type type,
                                                 SignatureConversion &result) {
  // Try to convert the given input type.
  SmallVector<Type, 1> convertedTypes;
  if (failed(convertType(type, convertedTypes)))
    return failure();

  // If this argument is being dropped, there is nothing left to do.
  if (convertedTypes.empty())
    return success();

  // Otherwise, add the new inputs.
  result.addInputs(inputNo, convertedTypes);
  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FuncOp.
namespace {
struct FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
  FuncOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  /// Hook for derived classes to implement combined matching and rewriting.
  PatternMatchResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = funcOp.getType();

    // Convert the original function arguments.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
      if (failed(converter.convertSignatureArg(i, type.getInput(i), result)))
        return matchFailure();

    // Convert the original function results.
    SmallVector<Type, 1> convertedResults;
    if (failed(converter.convertTypes(type.getResults(), convertedResults)))
      return matchFailure();

    // Create a new function with an updated signature.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.setType(FunctionType::get(result.getConvertedTypes(),
                                        convertedResults, funcOp.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
    rewriter.eraseOp(funcOp);
    return matchSuccess();
  }

  /// The type converter to use when rewriting the signature.
  TypeConverter &converter;
};
} // end anonymous namespace

void mlir::populateFuncOpTypeConversionPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx,
    TypeConverter &converter) {
  patterns.insert<FuncOpSignatureConversion>(ctx, converter);
}

/// This function converts the type signature of the given block, by invoking
/// 'convertSignatureArg' for each argument. This function should return a valid
/// conversion for the signature on success, None otherwise.
auto TypeConverter::convertBlockSignature(Block *block)
    -> llvm::Optional<SignatureConversion> {
  SignatureConversion conversion(block->getNumArguments());
  for (unsigned i = 0, e = block->getNumArguments(); i != e; ++i)
    if (failed(convertSignatureArg(i, block->getArgument(i)->getType(),
                                   conversion)))
      return llvm::None;
  return conversion;
}

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

/// Register a legality action for the given operation.
void ConversionTarget::setOpAction(OperationName op,
                                   LegalizationAction action) {
  legalOperations[op] = {action, /*isRecursivelyLegal=*/false};
}

/// Register a legality action for the given dialects.
void ConversionTarget::setDialectAction(ArrayRef<StringRef> dialectNames,
                                        LegalizationAction action) {
  for (StringRef dialect : dialectNames)
    legalDialects[dialect] = action;
}

/// Get the legality action for the given operation.
auto ConversionTarget::getOpAction(OperationName op) const
    -> Optional<LegalizationAction> {
  Optional<LegalizationInfo> info = getOpInfo(op);
  return info ? info->action : Optional<LegalizationAction>();
}

/// If the given operation instance is legal on this target, a structure
/// containing legality information is returned. If the operation is not legal,
/// None is returned.
auto ConversionTarget::isLegal(Operation *op) const
    -> Optional<LegalOpDetails> {
  Optional<LegalizationInfo> info = getOpInfo(op->getName());
  if (!info)
    return llvm::None;

  // Returns true if this operation instance is known to be legal.
  auto isOpLegal = [&] {
    // Handle dynamic legality.
    if (info->action == LegalizationAction::Dynamic) {
      // Check for callbacks on the operation or dialect.
      auto opFn = opLegalityFns.find(op->getName());
      if (opFn != opLegalityFns.end())
        return opFn->second(op);
      auto dialectFn = dialectLegalityFns.find(op->getName().getDialect());
      if (dialectFn != dialectLegalityFns.end())
        return dialectFn->second(op);

      // Otherwise, invoke the hook on the derived instance.
      return isDynamicallyLegal(op);
    }

    // Otherwise, the operation is only legal if it was marked 'Legal'.
    return info->action == LegalizationAction::Legal;
  };
  if (!isOpLegal())
    return llvm::None;

  // This operation is legal, compute any additional legality information.
  LegalOpDetails legalityDetails;

  if (info->isRecursivelyLegal) {
    auto legalityFnIt = opRecursiveLegalityFns.find(op->getName());
    if (legalityFnIt != opRecursiveLegalityFns.end())
      legalityDetails.isRecursivelyLegal = legalityFnIt->second(op);
    else
      legalityDetails.isRecursivelyLegal = true;
  }
  return legalityDetails;
}

/// Set the dynamic legality callback for the given operation.
void ConversionTarget::setLegalityCallback(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  opLegalityFns[name] = callback;
}

/// Set the recursive legality callback for the given operation and mark the
/// operation as recursively legal.
void ConversionTarget::markOpRecursivelyLegal(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  auto infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action != LegalizationAction::Illegal &&
         "expected operation to already be marked as legal");
  infoIt->second.isRecursivelyLegal = true;
  if (callback)
    opRecursiveLegalityFns[name] = callback;
  else
    opRecursiveLegalityFns.erase(name);
}

/// Set the dynamic legality callback for the given dialects.
void ConversionTarget::setLegalityCallback(
    ArrayRef<StringRef> dialects, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  for (StringRef dialect : dialects)
    dialectLegalityFns[dialect] = callback;
}

/// Get the legalization information for the given operation.
auto ConversionTarget::getOpInfo(OperationName op) const
    -> Optional<LegalizationInfo> {
  // Check for info for this specific operation.
  auto it = legalOperations.find(op);
  if (it != legalOperations.end())
    return it->second;
  // Otherwise, default to checking on the parent dialect.
  auto dialectIt = legalDialects.find(op.getDialect());
  if (dialectIt != legalDialects.end())
    return LegalizationInfo{dialectIt->second, /*isRecursivelyLegal=*/false};
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Apply a partial conversion on the given operations, and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize.
LogicalResult mlir::applyPartialConversion(
    ArrayRef<Operation *> ops, ConversionTarget &target,
    const OwningRewritePatternList &patterns, TypeConverter *converter) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Partial);
  return opConverter.convertOperations(ops, converter);
}
LogicalResult
mlir::applyPartialConversion(Operation *op, ConversionTarget &target,
                             const OwningRewritePatternList &patterns,
                             TypeConverter *converter) {
  return applyPartialConversion(llvm::makeArrayRef(op), target, patterns,
                                converter);
}

/// Apply a complete conversion on the given operations, and all nested
/// operations. This method will return failure if the conversion of any
/// operation fails.
LogicalResult
mlir::applyFullConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                          const OwningRewritePatternList &patterns,
                          TypeConverter *converter) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Full);
  return opConverter.convertOperations(ops, converter);
}
LogicalResult
mlir::applyFullConversion(Operation *op, ConversionTarget &target,
                          const OwningRewritePatternList &patterns,
                          TypeConverter *converter) {
  return applyFullConversion(llvm::makeArrayRef(op), target, patterns,
                             converter);
}

/// Apply an analysis conversion on the given operations, and all nested
/// operations. This method analyzes which operations would be successfully
/// converted to the target if a conversion was applied. All operations that
/// were found to be legalizable to the given 'target' are placed within the
/// provided 'convertedOps' set; note that no actual rewrites are applied to the
/// operations on success and only pre-existing operations are added to the set.
LogicalResult mlir::applyAnalysisConversion(
    ArrayRef<Operation *> ops, ConversionTarget &target,
    const OwningRewritePatternList &patterns,
    DenseSet<Operation *> &convertedOps, TypeConverter *converter) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Analysis,
                                 &convertedOps);
  return opConverter.convertOperations(ops, converter);
}
LogicalResult
mlir::applyAnalysisConversion(Operation *op, ConversionTarget &target,
                              const OwningRewritePatternList &patterns,
                              DenseSet<Operation *> &convertedOps,
                              TypeConverter *converter) {
  return applyAnalysisConversion(llvm::makeArrayRef(op), target, patterns,
                                 convertedOps, converter);
}
