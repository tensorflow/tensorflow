//===- PatternMatch.cpp - Base classes for pattern match ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
using namespace mlir;

PatternBenefit::PatternBenefit(unsigned benefit) : representation(benefit) {
  assert(representation == benefit && benefit != ImpossibleToMatchSentinel &&
         "This pattern match benefit is too large to represent");
}

unsigned short PatternBenefit::getBenefit() const {
  assert(representation != ImpossibleToMatchSentinel &&
         "Pattern doesn't match");
  return representation;
}

//===----------------------------------------------------------------------===//
// Pattern implementation
//===----------------------------------------------------------------------===//

Pattern::Pattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context)
    : rootKind(OperationName(rootName, context)), benefit(benefit) {}

// Out-of-line vtable anchor.
void Pattern::anchor() {}

//===----------------------------------------------------------------------===//
// RewritePattern and PatternRewriter implementation
//===----------------------------------------------------------------------===//

void RewritePattern::rewrite(Operation *op, std::unique_ptr<PatternState> state,
                             PatternRewriter &rewriter) const {
  rewrite(op, rewriter);
}

void RewritePattern::rewrite(Operation *op, PatternRewriter &rewriter) const {
  llvm_unreachable("need to implement either matchAndRewrite or one of the "
                   "rewrite functions!");
}

PatternMatchResult RewritePattern::match(Operation *op) const {
  llvm_unreachable("need to implement either match or matchAndRewrite!");
}

/// Patterns must specify the root operation name they match against, and can
/// also specify the benefit of the pattern matching. They can also specify the
/// names of operations that may be generated during a successful rewrite.
RewritePattern::RewritePattern(StringRef rootName,
                               ArrayRef<StringRef> generatedNames,
                               PatternBenefit benefit, MLIRContext *context)
    : Pattern(rootName, benefit, context) {
  generatedOps.reserve(generatedNames.size());
  std::transform(generatedNames.begin(), generatedNames.end(),
                 std::back_inserter(generatedOps), [context](StringRef name) {
                   return OperationName(name, context);
                 });
}

PatternRewriter::~PatternRewriter() {
  // Out of line to provide a vtable anchor for the class.
}

/// This method performs the final replacement for a pattern, where the
/// results of the operation are updated to use the specified list of SSA
/// values.  In addition to replacing and removing the specified operation,
/// clients can specify a list of other nodes that this replacement may make
/// (perhaps transitively) dead.  If any of those ops are dead, this will
/// remove them as well.
void PatternRewriter::replaceOp(Operation *op, ValueRange newValues,
                                ValueRange valuesToRemoveIfDead) {
  // Notify the rewriter subclass that we're about to replace this root.
  notifyRootReplaced(op);

  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  op->replaceAllUsesWith(newValues);

  notifyOperationRemoved(op);
  op->erase();

  // TODO: Process the valuesToRemoveIfDead list, removing things and calling
  // the notifyOperationRemoved hook in the process.
}

/// This method erases an operation that is known to have no uses. The uses of
/// the given operation *must* be known to be dead.
void PatternRewriter::eraseOp(Operation *op) {
  assert(op->use_empty() && "expected 'op' to have no uses");
  notifyOperationRemoved(op);
  op->erase();
}

/// Merge the operations of block 'source' into the end of block 'dest'.
/// 'source's predecessors must be empty or only contain 'dest`.
/// 'argValues' is used to replace the block arguments of 'source' after
/// merging.
void PatternRewriter::mergeBlocks(Block *source, Block *dest,
                                  ValueRange argValues) {
  assert(llvm::all_of(source->getPredecessors(),
                      [dest](Block *succ) { return succ == dest; }) &&
         "expected 'source' to have no predecessors or only 'dest'");
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");

  // Replace all of the successor arguments with the provided values.
  for (auto it : llvm::zip(source->getArguments(), argValues))
    std::get<0>(it)->replaceAllUsesWith(std::get<1>(it));

  // Splice the operations of the 'source' block into the 'dest' block and erase
  // it.
  dest->getOperations().splice(dest->end(), source->getOperations());
  source->dropAllUses();
  source->erase();
}

/// Split the operations starting at "before" (inclusive) out of the given
/// block into a new block, and return it.
Block *PatternRewriter::splitBlock(Block *block, Block::iterator before) {
  return block->splitBlock(before);
}

/// op and newOp are known to have the same number of results, replace the
/// uses of op with uses of newOp
void PatternRewriter::replaceOpWithResultsOfAnotherOp(
    Operation *op, Operation *newOp, ValueRange valuesToRemoveIfDead) {
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  if (op->getNumResults() == 1)
    return replaceOp(op, newOp->getResult(0), valuesToRemoveIfDead);
  return replaceOp(op, newOp->getResults(), valuesToRemoveIfDead);
}

/// Move the blocks that belong to "region" before the given position in
/// another region.  The two regions must be different.  The caller is in
/// charge to update create the operation transferring the control flow to the
/// region and pass it the correct block arguments.
void PatternRewriter::inlineRegionBefore(Region &region, Region &parent,
                                         Region::iterator before) {
  parent.getBlocks().splice(before, region.getBlocks());
}
void PatternRewriter::inlineRegionBefore(Region &region, Block *before) {
  inlineRegionBefore(region, *before->getParent(), before->getIterator());
}

/// Clone the blocks that belong to "region" before the given position in
/// another region "parent". The two regions must be different. The caller is
/// responsible for creating or updating the operation transferring flow of
/// control to the region and passing it the correct block arguments.
void PatternRewriter::cloneRegionBefore(Region &region, Region &parent,
                                        Region::iterator before,
                                        BlockAndValueMapping &mapping) {
  region.cloneInto(&parent, before, mapping);
}
void PatternRewriter::cloneRegionBefore(Region &region, Region &parent,
                                        Region::iterator before) {
  BlockAndValueMapping mapping;
  cloneRegionBefore(region, parent, before, mapping);
}
void PatternRewriter::cloneRegionBefore(Region &region, Block *before) {
  cloneRegionBefore(region, *before->getParent(), before->getIterator());
}

//===----------------------------------------------------------------------===//
// PatternMatcher implementation
//===----------------------------------------------------------------------===//

RewritePatternMatcher::RewritePatternMatcher(
    const OwningRewritePatternList &patterns) {
  for (auto &pattern : patterns)
    this->patterns.push_back(pattern.get());

  // Sort the patterns by benefit to simplify the matching logic.
  std::stable_sort(this->patterns.begin(), this->patterns.end(),
                   [](RewritePattern *l, RewritePattern *r) {
                     return r->getBenefit() < l->getBenefit();
                   });
}

/// Try to match the given operation to a pattern and rewrite it.
bool RewritePatternMatcher::matchAndRewrite(Operation *op,
                                            PatternRewriter &rewriter) {
  for (auto *pattern : patterns) {
    // Ignore patterns that are for the wrong root or are impossible to match.
    if (pattern->getRootKind() != op->getName() ||
        pattern->getBenefit().isImpossibleToMatch())
      continue;

    // Try to match and rewrite this pattern. The patterns are sorted by
    // benefit, so if we match we can immediately rewrite and return.
    if (pattern->matchAndRewrite(op, rewriter))
      return true;
  }
  return false;
}
