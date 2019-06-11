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

#define DEBUG_TYPE "dialect-conversion"

//===----------------------------------------------------------------------===//
// ArgConverter
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a simple interface for converting the types of block
/// arguments. This is done by inserting fake cast operations for the illegal
/// type that allow for updating the real type to return the correct type.
struct ArgConverter {
  ArgConverter(MLIRContext *ctx)
      : castOpName(kCastName, ctx), loc(UnknownLoc::get(ctx)) {}

  /// Cleanup and undo any generated conversion values.
  void discardRewrites() {
    // On failure drop all uses of the cast operation and destroy it.
    for (auto *op : castOps) {
      op->getResult(0)->dropAllUses();
      op->destroy();
    }
    castOps.clear();
  }

  /// Replace usages of the cast operations with the argument directly.
  void applyRewrites() {
    // On success, we update the type of the block argument and replace uses of
    // the cast.
    for (auto *op : castOps) {
      op->getOperand(0)->setType(op->getResult(0)->getType());
      op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
      op->destroy();
    }
  }

  /// Generate a cast operation for 'arg' that produces the new, legal, type.
  void castArgument(BlockArgument *arg, Type newType,
                    BlockAndValueMapping &mapping) {
    // Otherwise, generate a new cast operation for the given value type.
    auto *cast = Operation::create(loc, castOpName, arg, newType, llvm::None,
                                   llvm::None, 0, false, arg->getContext());

    // Replace the uses of the argument and record the mapping.
    mapping.map(arg, cast->getResult(0));
    castOps.push_back(cast);
  }

  /// This is an operation name for a fake operation that is inserted during the
  /// conversion process. Operations of this type are guaranteed to never escape
  /// the converter.
  static constexpr StringLiteral kCastName = "__mlir_conversion.cast";
  OperationName castOpName;

  /// This is a collection of cast values that were generated during the
  /// conversion process.
  std::vector<Operation *> castOps;

  /// An instance of the unknown location that is used when generating
  /// producers.
  UnknownLoc loc;
};

constexpr StringLiteral ArgConverter::kCastName;

//===----------------------------------------------------------------------===//
// DialectConversionRewriter
//===----------------------------------------------------------------------===//

/// This class contains a snapshot of the current conversion rewriter state.
/// This is useful when saving and undoing a set of rewrites.
struct RewriterState {
  RewriterState(unsigned numCreatedOperations, unsigned numReplacements,
                unsigned numBlockActions)
      : numCreatedOperations(numCreatedOperations),
        numReplacements(numReplacements), numBlockActions(numBlockActions) {}

  /// The current number of created operations.
  unsigned numCreatedOperations;

  /// The current number of replacements queued.
  unsigned numReplacements;

  /// The current number of block actions performed.
  unsigned numBlockActions;
};

/// This class implements a pattern rewriter for ConversionPattern
/// patterns. It automatically performs remapping of replaced operation values.
struct DialectConversionRewriter final : public PatternRewriter {
  /// This class represents one requested operation replacement via 'replaceOp'.
  struct OpReplacement {
    OpReplacement() = default;
    OpReplacement(Operation *op, ArrayRef<Value *> newValues)
        : op(op), newValues(newValues.begin(), newValues.end()) {}

    Operation *op;
    SmallVector<Value *, 2> newValues;
  };

  /// The kind of the block action performed during the rewrite.  Actions can be
  /// undone if the conversion fails.
  enum class BlockActionKind { Split, Move };

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

    BlockActionKind kind;
  };

  DialectConversionRewriter(Region &region)
      : PatternRewriter(region), argConverter(region.getContext()) {}
  ~DialectConversionRewriter() = default;

  /// Return the current state of the rewriter.
  RewriterState getCurrentState() {
    return RewriterState(createdOps.size(), replacements.size(),
                         blockActions.size());
  }

  /// Reset the state of the rewriter to a previously saved point.
  void resetState(RewriterState state) {
    // Reset any replaced operations and undo any saved mappings.
    for (auto &repl : llvm::drop_begin(replacements, state.numReplacements))
      for (auto *result : repl.op->getResults())
        mapping.erase(result);
    replacements.resize(state.numReplacements);

    // Pop all of the newly created operations.
    while (createdOps.size() != state.numCreatedOperations)
      createdOps.pop_back_val()->erase();

    // Undo any block operations.
    undoBlockActions(state.numBlockActions);
  }

  /// Undo the block actions (motions, splits) one by one in reverse order until
  /// "numActionsToKeep" actions remains.
  void undoBlockActions(unsigned numActionsToKeep = 0) {
    for (auto &action :
         llvm::reverse(llvm::drop_begin(blockActions, numActionsToKeep))) {
      switch (action.kind) {
      // Merge back the block that was split out.
      case BlockActionKind::Split: {
        action.originalBlock->getOperations().splice(
            action.originalBlock->end(), action.block->getOperations());
        action.block->erase();
        break;
      }
      // Move the block back to its original position.
      case BlockActionKind::Move: {
        Region *originalRegion = action.originalPosition.region;
        originalRegion->getBlocks().splice(
            std::next(originalRegion->begin(),
                      action.originalPosition.position),
            action.block->getParent()->getBlocks(), action.block);
        break;
      }
      }
    }
  }

  /// Cleanup and destroy any generated rewrite operations. This method is
  /// invoked when the conversion process fails.
  void discardRewrites() {
    argConverter.discardRewrites();

    // Remove any newly created ops.
    for (auto *op : createdOps) {
      op->dropAllDefinedValueUses();
      op->erase();
    }

    undoBlockActions();
  }

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites() {
    // Apply all of the rewrites replacements requested during conversion.
    for (auto &repl : replacements) {
      for (unsigned i = 0, e = repl.newValues.size(); i != e; ++i)
        repl.op->getResult(i)->replaceAllUsesWith(repl.newValues[i]);
      repl.op->erase();
    }

    argConverter.applyRewrites();
  }

  /// PatternRewriter hook for replacing the results of an operation.
  void replaceOp(Operation *op, ArrayRef<Value *> newValues,
                 ArrayRef<Value *> valuesToRemoveIfDead) override {
    assert(newValues.size() == op->getNumResults());

    // Create mappings for each of the new result values.
    for (unsigned i = 0, e = newValues.size(); i < e; ++i) {
      assert((newValues[i] || op->getResult(i)->use_empty()) &&
             "result value has remaining uses that must be replaced");
      if (newValues[i])
        mapping.map(op->getResult(i), newValues[i]);
    }

    // Record the requested operation replacement.
    replacements.emplace_back(op, newValues);
  }

  /// PatternRewriter hook for splitting a block into two parts.
  Block *splitBlock(Block *block, Block::iterator before) override {
    auto *continuation = PatternRewriter::splitBlock(block, before);
    BlockAction action;
    action.kind = BlockActionKind::Split;
    action.block = continuation;
    action.originalBlock = block;
    blockActions.push_back(action);
    return continuation;
  }

  /// PatternRewriter hook for moving blocks out of a region.
  void inlineRegionBefore(Region &region, Region::iterator before) override {
    for (auto &pair : llvm::enumerate(region)) {
      Block &block = pair.value();
      unsigned position = pair.index();
      BlockAction action;
      action.kind = BlockActionKind::Move;
      action.block = &block;
      action.originalPosition = {&region, position};
      blockActions.push_back(action);
    }
    PatternRewriter::inlineRegionBefore(region, before);
  }

  /// PatternRewriter hook for creating a new operation.
  Operation *createOperation(const OperationState &state) override {
    auto *result = OpBuilder::createOperation(state);
    createdOps.push_back(result);
    return result;
  }

  /// PatternRewriter hook for updating the root operation in-place.
  void notifyRootUpdated(Operation *op) override {
    // The rewriter caches changes to the IR to allow for operating in-place and
    // backtracking. The rewrite is currently not capable of backtracking
    // in-place modifications.
    llvm_unreachable("in-place operation updates are not supported");
  }

  /// Remap the given operands to those with potentially different types.
  void remapValues(Operation::operand_range operands,
                   SmallVectorImpl<Value *> &remapped) {
    remapped.reserve(llvm::size(operands));
    for (Value *operand : operands)
      remapped.push_back(mapping.lookupOrDefault(operand));
  }

  // Mapping between replaced values that differ in type. This happens when
  // replacing a value with one of a different type.
  BlockAndValueMapping mapping;

  /// Utility used to convert block arguments.
  ArgConverter argConverter;

  /// Ordered vector of all of the newly created operations during conversion.
  SmallVector<Operation *, 4> createdOps;

  /// Ordered vector of any requested operation replacements.
  SmallVector<OpReplacement, 4> replacements;

  /// Ordered list of block operations (creations, splits, motions).
  SmallVector<BlockAction, 4> blockActions;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConversionPattern
//===----------------------------------------------------------------------===//

/// Attempt to match and rewrite the IR root at the specified operation.
PatternMatchResult
ConversionPattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  SmallVector<Value *, 4> operands;
  auto &dialectRewriter = static_cast<DialectConversionRewriter &>(rewriter);
  dialectRewriter.remapValues(op->getOperands(), operands);

  // If this operation has no successors, invoke the rewrite directly.
  if (op->getNumSuccessors() == 0)
    return matchAndRewrite(op, operands, rewriter);

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
      destinations, operandsPerDestination, rewriter);
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
  OperationLegalizer(ConversionTarget &targetInfo,
                     OwningRewritePatternList &patterns)
      : target(targetInfo) {
    buildLegalizationGraph(patterns);
    computeLegalizationGraphBenefit();
  }

  /// Attempt to legalize the given operation. Returns success if the operation
  /// was legalized, failure otherwise.
  LogicalResult legalize(Operation *op, DialectConversionRewriter &rewriter);

private:
  /// Attempt to legalize the given operation by applying the provided pattern.
  /// Returns success if the operation was legalized, failure otherwise.
  LogicalResult legalizePattern(Operation *op, RewritePattern *pattern,
                                DialectConversionRewriter &rewriter);

  /// Build an optimistic legalization graph given the provided patterns. This
  /// function populates 'legalizerPatterns' with the operations that are not
  /// directly legal, but may be transitively legal for the current target given
  /// the provided patterns.
  void buildLegalizationGraph(OwningRewritePatternList &patterns);

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

LogicalResult
OperationLegalizer::legalize(Operation *op,
                             DialectConversionRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "Legalizing operation : " << op->getName()
                          << "\n");

  // Check if this was marked legal by the target.
  if (auto action = target.getOpAction(op->getName())) {
    // Check if this operation is always legal.
    if (*action == ConversionTarget::LegalizationAction::Legal)
      return success();

    // Otherwise, handle dynamic legalization.
    LLVM_DEBUG(llvm::dbgs() << "- Trying dynamic legalization.\n");
    if (target.isLegal(op))
      return success();

    // Fallthough to see if a pattern can convert this into a legal operation.
  }

  // Otherwise, we need to apply a legalization pattern to this operation.
  auto it = legalizerPatterns.find(op->getName());
  if (it == legalizerPatterns.end()) {
    LLVM_DEBUG(llvm::dbgs() << "-- FAIL : no known legalization path.\n");
    return failure();
  }

  // TODO(riverriddle) This currently has no cost model and doesn't prioritize
  // specific patterns in any way.
  for (auto *pattern : it->second)
    if (succeeded(legalizePattern(op, pattern, rewriter)))
      return success();

  LLVM_DEBUG(llvm::dbgs() << "-- FAIL : no matched legalization pattern.\n");
  return failure();
}

LogicalResult
OperationLegalizer::legalizePattern(Operation *op, RewritePattern *pattern,
                                    DialectConversionRewriter &rewriter) {
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

  RewriterState curState = rewriter.getCurrentState();
  auto cleanupFailure = [&] {
    // Reset the rewriter state and pop this pattern.
    rewriter.resetState(curState);
    appliedPatterns.erase(pattern);
    return failure();
  };

  // Try to rewrite with the given pattern.
  rewriter.setInsertionPoint(op);
  if (!pattern->matchAndRewrite(op, rewriter)) {
    LLVM_DEBUG(llvm::dbgs() << "-- FAIL: Pattern failed to match.\n");
    return cleanupFailure();
  }

  // Recursively legalize each of the new operations.
  for (unsigned i = curState.numCreatedOperations,
                e = rewriter.createdOps.size();
       i != e; ++i) {
    if (failed(legalize(rewriter.createdOps[i], rewriter))) {
      LLVM_DEBUG(llvm::dbgs() << "-- FAIL: Generated operation was illegal.\n");
      return cleanupFailure();
    }
  }

  appliedPatterns.erase(pattern);
  return success();
}

void OperationLegalizer::buildLegalizationGraph(
    OwningRewritePatternList &patterns) {
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
    if (target.getOpAction(root) == ConversionTarget::LegalizationAction::Legal)
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
          return !legalizerPatterns.count(op) && !target.getOpAction(op);
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
    if (opPatternsIt == legalizerPatterns.end())
      return 0u;

    auto &minDepth = minPatternDepth[op];
    if (opPatternsIt->second.empty())
      return minDepth;

    // Initialize the depth to the maximum value.
    minDepth = std::numeric_limits<unsigned>::max();

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
// FunctionConverter
//===----------------------------------------------------------------------===//
namespace {
// This class converts a single function using the given pattern matcher. If a
// TypeConverter object is provided, then the types of block arguments will be
// converted using the appropriate 'convertType' calls.
struct FunctionConverter {
  explicit FunctionConverter(MLIRContext *ctx, ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter *conversion = nullptr)
      : typeConverter(conversion), opLegalizer(target, patterns) {}

  /// Converts the given function to the dialect using hooks defined in
  /// `typeConverter`. Returns failure on error, success otherwise.
  LogicalResult convertFunction(Function *f);

  /// Converts the given region starting from the entry block and following the
  /// block successors. Returns failure on error, success otherwise. Prints
  /// error messages at `loc`.
  LogicalResult convertRegion(DialectConversionRewriter &rewriter,
                              Region &region, Location loc);

  /// Converts a block by traversing its operations sequentially, attempting to
  /// match a pattern. If there is no match, recurses the operations regions if
  /// it has any.
  //
  /// After converting operations, traverses the successor blocks unless they
  /// have been visited already as indicated in `visitedBlocks`.
  LogicalResult convertBlock(DialectConversionRewriter &rewriter, Block *block,
                             DenseSet<Block *> &visitedBlocks);

  /// Converts the type of the given block argument. Returns success if the
  /// argument type could be successfully converted, failure otherwise.
  LogicalResult convertArgument(DialectConversionRewriter &rewriter,
                                BlockArgument *arg, Location loc);

  /// Pointer to a specific dialect conversion info.
  TypeConverter *typeConverter;

  /// The legalizer to use when converting operations.
  OperationLegalizer opLegalizer;
};
} // end anonymous namespace

LogicalResult
FunctionConverter::convertArgument(DialectConversionRewriter &rewriter,
                                   BlockArgument *arg, Location loc) {
  auto convertedType = typeConverter->convertType(arg->getType());
  if (!convertedType)
    return arg->getContext()->emitError(loc)
           << "could not convert block argument of type : " << arg->getType();

  // Generate a replacement value, with the new type, for this argument.
  if (convertedType != arg->getType())
    rewriter.argConverter.castArgument(arg, convertedType, rewriter.mapping);
  return success();
}

LogicalResult
FunctionConverter::convertBlock(DialectConversionRewriter &rewriter,
                                Block *block,
                                DenseSet<Block *> &visitedBlocks) {
  // First, add the current block to the list of visited blocks.
  visitedBlocks.insert(block);

  if (block->empty())
    return success();

  // Preserve the successors before rewriting the operations.
  SmallVector<Block *, 4> successors(block->getSuccessors());

  // Iterate over ops and convert them.  Since the conversion may split the
  // block, we eagerly take the pointer to the next operation in it.  Splitting
  // moves the operations from one block to another, so this will keep
  // considering the original list of operations independently of the block
  // within which they are currently located.  This relies on iplist node API
  // to get the next node in the list witout knowing which list it is, iterators
  // are unsuitable because block splitting invalidates all iterators following
  // the current one. Any operation inserted by the conversion, independently of
  // its parent block, will be recursively legalized independently of this
  // function.
  Operation *current = &block->front();
  Operation *next = nullptr;
  do {
    next = current->getNextNode();
    // Traverse any held regions.
    for (auto &region : current->getRegions())
      if (!region.empty() &&
          failed(convertRegion(rewriter, region, current->getLoc())))
        return failure();

    // Legalize the current operation.
    (void)opLegalizer.legalize(current, rewriter);
  } while ((current = next));

  // Recurse to children that haven't been visited.
  for (Block *succ : successors) {
    if (visitedBlocks.count(succ))
      continue;
    if (failed(convertBlock(rewriter, succ, visitedBlocks)))
      return failure();
  }
  return success();
}

LogicalResult
FunctionConverter::convertRegion(DialectConversionRewriter &rewriter,
                                 Region &region, Location loc) {
  assert(!region.empty() && "expected non-empty region");

  // Create the arguments of each of the blocks in the region. If a type
  // converter was not provided, then we don't need to change any of the block
  // types.
  if (typeConverter) {
    for (Block &block : region)
      for (auto *arg : block.getArguments())
        if (failed(convertArgument(rewriter, arg, loc)))
          return failure();
  }

  // Store the number of blocks before conversion (new blocks may be added due
  // to splits or moves, but the operations in them will be processed
  // elsewhere).
  unsigned numBlocks = std::distance(region.begin(), region.end());

  // Start a DFS-order traversal of the CFG to make sure defs are converted
  // before uses in dominated blocks.
  llvm::DenseSet<Block *> visitedBlocks;
  if (failed(convertBlock(rewriter, &region.front(), visitedBlocks)))
    return failure();

  // If some blocks are not reachable through successor chains, they should have
  // been removed by the DCE before this.
  if (visitedBlocks.size() != numBlocks)
    return rewriter.getContext()->emitError(loc)
           << "unreachable blocks were not converted";
  return success();
}

LogicalResult FunctionConverter::convertFunction(Function *f) {
  // If this is an external function, there is nothing else to do.
  if (f->isExternal())
    return success();

  // Rewrite the function body.
  DialectConversionRewriter rewriter(f->getBody());
  if (failed(convertRegion(rewriter, f->getBody(), f->getLoc()))) {
    // Reset any of the generated rewrites.
    rewriter.discardRewrites();
    return failure();
  }

  // Otherwise the conversion succeeded, so apply all rewrites.
  rewriter.applyRewrites();
  return success();
}

//===----------------------------------------------------------------------===//
// TypeConverter
//===----------------------------------------------------------------------===//

// Create a function type with arguments and results converted, and argument
// attributes passed through.
FunctionType TypeConverter::convertFunctionSignatureType(
    FunctionType type, ArrayRef<NamedAttributeList> argAttrs,
    SmallVectorImpl<NamedAttributeList> &convertedArgAttrs) {
  SmallVector<Type, 8> arguments;
  SmallVector<Type, 4> results;

  arguments.reserve(type.getNumInputs());
  for (auto t : type.getInputs())
    arguments.push_back(convertType(t));

  results.reserve(type.getNumResults());
  for (auto t : type.getResults())
    results.push_back(convertType(t));

  // Note this will cause an extra allocation only if we need
  // to grow the caller-provided resulting attribute vector.
  convertedArgAttrs.reserve(arguments.size());
  for (auto attr : argAttrs)
    convertedArgAttrs.push_back(attr);

  return FunctionType::get(arguments, results, type.getContext());
}

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

/// Register a legality action for the given operation.
void ConversionTarget::setOpAction(OperationName op,
                                   LegalizationAction action) {
  legalOperations[op] = action;
}

/// Register a legality action for the given dialects.
void ConversionTarget::setDialectAction(ArrayRef<StringRef> dialectNames,
                                        LegalizationAction action) {
  for (StringRef dialect : dialectNames)
    legalDialects[dialect] = action;
}

/// Get the legality action for the given operation.
auto ConversionTarget::getOpAction(OperationName op) const
    -> llvm::Optional<LegalizationAction> {
  // Check for an action for this specific operation.
  auto it = legalOperations.find(op);
  if (it != legalOperations.end())
    return it->second;
  // Otherwise, default to checking for an action on the parent dialect.
  auto dialectIt = legalDialects.find(op.getDialect());
  if (dialectIt != legalDialects.end())
    return dialectIt->second;
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// applyConversionPatterns
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a function to be converted. It allows for converting
/// the body of functions and the signature in two phases.
struct ConvertedFunction {
  ConvertedFunction(Function *fn, FunctionType newType,
                    ArrayRef<NamedAttributeList> newFunctionArgAttrs)
      : fn(fn), newType(newType),
        newFunctionArgAttrs(newFunctionArgAttrs.begin(),
                            newFunctionArgAttrs.end()) {}

  /// The function to convert.
  Function *fn;
  /// The new type and argument attributes for the function.
  FunctionType newType;
  SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
};
} // end anonymous namespace

/// Convert the given module with the provided conversion patterns and type
/// conversion object. If conversion fails for specific functions, those
/// functions remains unmodified.
LogicalResult
mlir::applyConversionPatterns(Module &module, ConversionTarget &target,
                              TypeConverter &converter,
                              OwningRewritePatternList &&patterns) {
  std::vector<Function *> allFunctions;
  allFunctions.reserve(module.getFunctions().size());
  for (auto &func : module)
    allFunctions.push_back(&func);
  return applyConversionPatterns(allFunctions, target, converter,
                                 std::move(patterns));
}

/// Convert the given functions with the provided conversion patterns. This will
/// convert as many of the operations within each function as possible given the
/// set of patterns. If conversion fails for specific functions, those functions
// remains unmodified.
LogicalResult mlir::applyConversionPatterns(
    ArrayRef<Function *> fns, ConversionTarget &target,
    TypeConverter &converter, OwningRewritePatternList &&patterns) {
  if (fns.empty())
    return success();

  // Build the function converter.
  FunctionConverter funcConverter(fns.front()->getContext(), target, patterns,
                                  &converter);

  // Try to convert each of the functions within the module. Defer updating the
  // signatures of the functions until after all of the bodies have been
  // converted. This allows for the conversion patterns to still rely on the
  // public signatures of the functions within the module before they are
  // updated.
  std::vector<ConvertedFunction> toConvert;
  toConvert.reserve(fns.size());
  for (auto *func : fns) {
    // Convert the function type using the dialect converter.
    SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
    FunctionType newType = converter.convertFunctionSignatureType(
        func->getType(), func->getAllArgAttrs(), newFunctionArgAttrs);
    if (!newType || !newType.isa<FunctionType>())
      return func->emitError("could not convert function type");

    // Convert the body of this function.
    if (failed(funcConverter.convertFunction(func)))
      return failure();

    // Add function signature to be updated.
    toConvert.emplace_back(func, newType.cast<FunctionType>(),
                           newFunctionArgAttrs);
  }

  // Finally, update the signatures of all of the converted functions.
  for (auto &it : toConvert) {
    it.fn->setType(it.newType);
    it.fn->setAllArgAttrs(it.newFunctionArgAttrs);
  }

  return success();
}

/// Convert the given function with the provided conversion patterns. This will
/// convert as many of the operations within 'fn' as possible given the set of
/// patterns.
LogicalResult
mlir::applyConversionPatterns(Function &fn, ConversionTarget &target,
                              OwningRewritePatternList &&patterns) {
  // Convert the body of this function.
  FunctionConverter converter(fn.getContext(), target, patterns);
  return converter.convertFunction(&fn);
}
