//===- PatternMatch.cpp - Base classes for pattern match ------------------===//
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

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Statements.h"
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

bool PatternBenefit::operator==(const PatternBenefit& other) {
  if (isImpossibleToMatch())
    return other.isImpossibleToMatch();
  if (other.isImpossibleToMatch())
    return false;
  return getBenefit() == other.getBenefit();
}

bool PatternBenefit::operator!=(const PatternBenefit& other) {
  return !(*this == other);
}

//===----------------------------------------------------------------------===//
// Pattern implementation
//===----------------------------------------------------------------------===//

Pattern::Pattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context)
    : rootKind(OperationName(rootName, context)), benefit(benefit) {}

void Pattern::rewrite(Operation *op, std::unique_ptr<PatternState> state,
                      PatternRewriter &rewriter) const {
  rewrite(op, rewriter);
}

void Pattern::rewrite(Operation *op, PatternRewriter &rewriter) const {
  llvm_unreachable("need to implement one of the rewrite functions!");
}

//===----------------------------------------------------------------------===//
// PatternRewriter implementation
//===----------------------------------------------------------------------===//

PatternRewriter::~PatternRewriter() {
  // Out of line to provide a vtable anchor for the class.
}

/// This method performs the final replacement for a pattern, where the
/// results of the operation are updated to use the specified list of SSA
/// values.  In addition to replacing and removing the specified operation,
/// clients can specify a list of other nodes that this replacement may make
/// (perhaps transitively) dead.  If any of those ops are dead, this will
/// remove them as well.
void PatternRewriter::replaceOp(Operation *op, ArrayRef<SSAValue *> newValues,
                                ArrayRef<SSAValue *> opsToRemoveIfDead) {
  // Notify the rewriter subclass that we're about to replace this root.
  notifyRootReplaced(op);

  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  for (unsigned i = 0, e = newValues.size(); i != e; ++i)
    op->getResult(i)->replaceAllUsesWith(newValues[i]);

  notifyOperationRemoved(op);
  op->erase();

  // TODO: Process the opsToRemoveIfDead list, removing things and calling the
  // notifyOperationRemoved hook in the process.
}

/// This method is used as the final replacement hook for patterns that match
/// a single result value.  In addition to replacing and removing the
/// specified operation, clients can specify a list of other nodes that this
/// replacement may make (perhaps transitively) dead.  If any of those ops are
/// dead, this will remove them as well.
void PatternRewriter::replaceSingleResultOp(
    Operation *op, SSAValue *newValue, ArrayRef<SSAValue *> opsToRemoveIfDead) {
  // Notify the rewriter subclass that we're about to replace this root.
  notifyRootReplaced(op);

  assert(op->getNumResults() == 1 && "op isn't a SingleResultOp!");
  op->getResult(0)->replaceAllUsesWith(newValue);

  notifyOperationRemoved(op);
  op->erase();

  // TODO: Process the opsToRemoveIfDead list, removing things and calling the
  // notifyOperationRemoved hook in the process.
}

/// This method is used as the final notification hook for patterns that end
/// up modifying the pattern root in place, by changing its operands.  This is
/// a minor efficiency win (it avoids creating a new instruction and removing
/// the old one) but also often allows simpler code in the client.
///
/// The opsToRemoveIfDead list is an optional list of nodes that the rewriter
/// should remove if they are dead at this point.
///
void PatternRewriter::updatedRootInPlace(
    Operation *op, ArrayRef<SSAValue *> opsToRemoveIfDead) {
  // Notify the rewriter subclass that we're about to replace this root.
  notifyRootUpdated(op);

  // TODO: Process the opsToRemoveIfDead list, removing things and calling the
  // notifyOperationRemoved hook in the process.
}

//===----------------------------------------------------------------------===//
// PatternMatcher implementation
//===----------------------------------------------------------------------===//

/// Find the highest benefit pattern available in the pattern set for the DAG
/// rooted at the specified node.  This returns the pattern if found, or null
/// if there are no matches.
auto PatternMatcher::findMatch(Operation *op) -> MatchResult {
  // TODO: This is a completely trivial implementation, expand this in the
  // future.

  // Keep track of the best match, the benefit of it, and any matcher specific
  // state it is maintaining.
  MatchResult bestMatch = {nullptr, nullptr};
  Optional<PatternBenefit> bestBenefit;

  for (auto &pattern : patterns) {
    // Ignore patterns that are for the wrong root.
    if (pattern->getRootKind() != op->getName())
      continue;

    auto benefit = pattern->getBenefit();
    if (benefit.isImpossibleToMatch())
      continue;

    // If the benefit of the pattern is worse than what we've already found then
    // don't run it.
    if (bestBenefit.hasValue() &&
        benefit.getBenefit() < bestBenefit.getValue().getBenefit())
      continue;

    // Check to see if this pattern matches this node.
    auto result = pattern->match(op);

    // If this pattern failed to match, ignore it.
    if (!result)
      continue;

    // Okay we found a match that is better than our previous one, remember it.
    bestBenefit = benefit;
    bestMatch = {pattern.get(), std::move(result.getValue())};
  }

  // If we found any match, return it.
  return bestMatch;
}
