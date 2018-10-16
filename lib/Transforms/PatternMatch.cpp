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

#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Statements.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/PatternMatch.h"

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

Pattern::Pattern(OperationName rootKind, Optional<PatternBenefit> staticBenefit)
    : rootKind(rootKind), staticBenefit(staticBenefit) {}

Pattern::Pattern(OperationName rootKind, unsigned staticBenefit)
    : rootKind(rootKind), staticBenefit(staticBenefit) {}

Optional<PatternBenefit> Pattern::getStaticBenefit() const {
  return staticBenefit;
}

OperationName Pattern::getRootKind() const { return rootKind; }

void Pattern::rewrite(Operation *op, std::unique_ptr<PatternState> state,
                      // TODO: Need a generic builder.
                      MLFuncBuilder &builder) const {
  rewrite(op, builder);
}

void Pattern::rewrite(Operation *op,
                      // TODO: Need a generic builder.
                      MLFuncBuilder &builder) const {
  llvm_unreachable("need to implement one of the rewrite functions!");
}

/// This method indicates that no match was found.
PatternMatchResult Pattern::matchFailure() {
  // TODO: Use a proper sentinel / discriminated union instad of -1 magic
  // number.
  return {-1, std::unique_ptr<PatternState>()};
}

/// This method indicates that a match was found and has the specified cost.
PatternMatchResult
Pattern::matchSuccess(PatternBenefit benefit,
                      std::unique_ptr<PatternState> state) const {
  assert((!getStaticBenefit().hasValue() ||
          getStaticBenefit().getValue() == benefit) &&
         "This version of matchSuccess must be called with a benefit that "
         "matches the static benefit if set!");

  return {benefit, std::move(state)};
}

/// This method indicates that a match was found for patterns that have a
/// known static benefit.
PatternMatchResult
Pattern::matchSuccess(std::unique_ptr<PatternState> state) const {
  auto benefit = getStaticBenefit();
  assert(benefit.hasValue() && "Pattern doesn't have a static benefit");
  return matchSuccess(benefit.getValue(), std::move(state));
}

/// This method is used as the final replacement hook for patterns that match
/// a single result value.  In addition to replacing and removing the
/// specified operation, clients can specify a list of other nodes that this
/// replacement may make (perhaps transitively) dead.  If any of those ops are
/// dead, this will remove them as well.
void Pattern::replaceSingleResultOp(
    Operation *op, SSAValue *newValue,
    ArrayRef<SSAValue *> opsToRemoveIfDead) const {
  assert(op->getNumResults() == 1 && "op isn't a SingleResultOp!");
  op->getResult(0)->replaceAllUsesWith(newValue);

  // TODO: This shouldn't be statement specific.
  cast<OperationStmt>(op)->eraseFromBlock();

  // TODO: Process the opsToRemoveIfDead list once we have side-effect
  // information.  Be careful about notifying clients that this is happening
  // so they can be removed from worklists etc (needs a callback of some
  // sort).
}

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

  for (auto *pattern : patterns) {
    // Ignore patterns that are for the wrong root.
    if (pattern->getRootKind() != op->getName())
      continue;

    // If we know the static cost of the pattern is worse than what we've
    // already found then don't run it.
    auto staticBenefit = pattern->getStaticBenefit();
    if (staticBenefit.hasValue() && bestBenefit.hasValue() &&
        staticBenefit.getValue().getBenefit() <
            bestBenefit.getValue().getBenefit())
      continue;

    // Check to see if this pattern matches this node.
    auto result = pattern->match(op);
    auto benefit = result.first;

    // If this pattern failed to match, ignore it.
    if (benefit.isImpossibleToMatch())
      continue;

    // If it matched but had lower benefit than our best match so far, then
    // ignore it.
    if (bestBenefit.hasValue() &&
        benefit.getBenefit() < bestBenefit.getValue().getBenefit())
      continue;

    // Okay we found a match that is better than our previous one, remember it.
    bestBenefit = benefit;
    bestMatch = {pattern, std::move(result.second)};
  }

  // If we found any match, return it.
  return bestMatch;
}
