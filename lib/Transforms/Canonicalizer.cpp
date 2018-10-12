//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
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
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
using namespace mlir;

//===----------------------------------------------------------------------===//
// Definition of Pattern and related types.
//===----------------------------------------------------------------------===//

// TODO(clattner): Move this out of this file when it is ready.

// TODO(clattner): Define this as a tagged union with proper sentinels.
typedef int PatternBenefit;

/// Pattern state is used by patterns that want to maintain state between their
/// match and rewrite phases.  Patterns can define a pattern-specific subclass
/// of this.
class PatternState {
public:
  virtual ~PatternState() {}
};

/// This is the type returned by a pattern match.  The first field indicates the
/// benefit of the match, the second is a state token that can optionally be
/// produced by a pattern match to maintain state between the match and rewrite
/// phases.
typedef std::pair<PatternBenefit, std::unique_ptr<PatternState>>
    PatternMatchResult;

class Pattern {
public:
  // Return the benefit (the inverse of “cost”) of matching this pattern,
  // if it is statically determinable.  The result is an integer if known,
  // a sentinel if dynamically computed, and another sentinel if the
  // pattern can never be matched.
  PatternBenefit getStaticBenefit() const { return staticBenefit; }

  // Return the root node that this pattern matches.  Patterns that can
  // match multiple root types are instantiated once per root.
  OperationName getRootKind() const { return rootKind; }

  //===--------------------------------------------------------------------===//
  // Implementation hooks for patterns to implement.
  //===--------------------------------------------------------------------===//

  // Attempt to match against code rooted at the specified operation,
  // which is the same operation code as getRootKind().  On success it
  // returns the benefit of the match along with an (optional)
  // pattern-specific state which is passed back into its rewrite
  // function if this match is selected.  On failure, this returns a
  // sentinel indicating that it didn’t match.
  virtual PatternMatchResult match(Operation *op) const = 0;

  // Rewrite the IR rooted at the specified operation with the result of
  // this pattern, generating any new operations with the specified
  // builder.  If an unexpected error is encountered (an internal
  // compiler error), it is emitted through the normal MLIR diagnostic
  // hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, std::unique_ptr<PatternState> state,
                       // TODO: Need a generic builder.
                       MLFuncBuilder &builder) const {
    rewrite(op, builder);
  }

  // Rewrite the IR rooted at the specified operation with the result of
  // this pattern, generating any new operations with the specified
  // builder.  If an unexpected error is encountered (an internal
  // compiler error), it is emitted through the normal MLIR diagnostic
  // hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op,
                       // TODO: Need a generic builder.
                       MLFuncBuilder &builder) const {
    llvm_unreachable("need to implement one of the rewrite functions!");
  }

  virtual ~Pattern();

  //===--------------------------------------------------------------------===//
  // Helper methods to simplify pattern implementations
  //===--------------------------------------------------------------------===//

  /// This method indicates that no match was found.
  static PatternMatchResult matchFailure() {
    // TODO: Use a proper sentinel / discriminated union instad of -1 magic
    // number.
    return {-1, std::unique_ptr<PatternState>()};
  }

  static PatternMatchResult matchSuccess(
      PatternBenefit benefit,
      std::unique_ptr<PatternState> state = std::unique_ptr<PatternState>()) {
    return {benefit, std::move(state)};
  }

protected:
  Pattern(PatternBenefit staticBenefit, OperationName rootKind)
      : staticBenefit(staticBenefit), rootKind(rootKind) {}

private:
  const PatternBenefit staticBenefit;
  const OperationName rootKind;
};

Pattern::~Pattern() {}

//===----------------------------------------------------------------------===//
// PatternMatcher class
//===----------------------------------------------------------------------===//

/// This class manages optimization an execution of a group of patterns, and
/// provides an API for finding the best match against a given node.
///
class PatternMatcher {
public:
  /// Create a PatternMatch with the specified set of patterns.  This takes
  /// ownership of the patterns in question.
  explicit PatternMatcher(ArrayRef<Pattern *> patterns)
      : patterns(patterns.begin(), patterns.end()) {}

  typedef std::pair<Pattern *, std::unique_ptr<PatternState>> MatchResult;

  /// Find the highest benefit pattern available in the pattern set for the DAG
  /// rooted at the specified node.  This returns the pattern (and any state it
  /// needs) if found, or null if there are no matches.
  MatchResult findMatch(Operation *op);

  ~PatternMatcher() {
    for (auto *p : patterns)
      delete p;
  }

private:
  PatternMatcher(const PatternMatcher &) = delete;
  void operator=(const PatternMatcher &) = delete;

  std::vector<Pattern *> patterns;
};

/// Find the highest benefit pattern available in the pattern set for the DAG
/// rooted at the specified node.  This returns the pattern if found, or null
/// if there are no matches.
auto PatternMatcher::findMatch(Operation *op) -> MatchResult {
  // TODO: This is a completely trivial implementation, expand this in the
  // future.

  // Keep track of the best match, the benefit of it, and any matcher specific
  // state it is maintaining.
  MatchResult bestMatch = {nullptr, nullptr};
  // TODO: eliminate magic numbers.
  PatternBenefit bestBenefit = -1;

  for (auto *pattern : patterns) {
    // Ignore patterns that are for the wrong root.
    if (pattern->getRootKind() != op->getName())
      continue;

    // If we know the static cost of the pattern is worse than what we've
    // already found then don't run it.
    auto staticBenefit = pattern->getStaticBenefit();
    if (staticBenefit < 0 || staticBenefit < bestBenefit)
      continue;

    // Check to see if this pattern matches this node.
    auto result = pattern->match(op);
    // TODO: magic numbers.
    if (result.first < 0 || result.first < bestBenefit)
      continue;

    // Okay we found a match that is better than our previous one, remember it.
    bestBenefit = result.first;
    bestMatch = {pattern, std::move(result.second)};
  }

  // If we found any match, return it.
  return bestMatch;
}

//===----------------------------------------------------------------------===//
// Definition of a few patterns for canonicalizing operations.
//===----------------------------------------------------------------------===//

namespace {
/// subi(x,x) -> 0
///
struct SimplifyXMinusX : public Pattern {
  SimplifyXMinusX(MLIRContext *context)
      // FIXME: rename getOperationName and add a proper one.
      : Pattern(1, OperationName(SubIOp::getOperationName(), context)) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    // TODO: Rename getAs -> dyn_cast, and add a cast<> method.
    auto subi = op->getAs<SubIOp>();
    assert(subi && "Matcher should have produced this");

    if (subi->getOperand(0) == subi->getOperand(1))
      return matchSuccess(1);

    return matchFailure();
  }

  // Rewrite the IR rooted at the specified operation with the result of
  // this pattern, generating any new operations with the specified
  // builder.  If an unexpected error is encountered (an internal
  // compiler error), it is emitted through the normal MLIR diagnostic
  // hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, MLFuncBuilder &builder) const override {
    // TODO: Rename getAs -> dyn_cast, and add a cast<> method.
    auto subi = op->getAs<SubIOp>();
    assert(subi && "Matcher should have produced this");

    // TODO: Better "replace and remove" API on Pattern.
    auto result =
        builder.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());
    op->getResult(0)->replaceAllUsesWith(result->getResult());

    cast<OperationStmt>(op)->eraseFromBlock();
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// The actual Canonicalizer Pass.
//===----------------------------------------------------------------------===//

// TODO: Canonicalize and unique all constant operations into the entry of the
// function.

namespace {
/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;

  void simplifyFunction(std::vector<Operation *> &worklist,
                        MLFuncBuilder &builder);
};
} // end anonymous namespace

PassResult Canonicalizer::runOnCFGFunction(CFGFunction *f) {
  // TODO: Add this.
  return success();
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *f) {
  std::vector<Operation *> worklist;
  worklist.reserve(64);

  f->walk([&](OperationStmt *stmt) { worklist.push_back(stmt); });

  MLFuncBuilder builder(f);
  simplifyFunction(worklist, builder);
  return success();
}

// TODO: This should work on both ML and CFG functions.
void Canonicalizer::simplifyFunction(std::vector<Operation *> &worklist,
                                     MLFuncBuilder &builder) {
  // TODO: Instead of a hard coded list of patterns, ask the registered dialects
  // for their canonicalization patterns.

  PatternMatcher matcher({new SimplifyXMinusX(builder.getContext())});

  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();

    // TODO: If no side effects, and operation has no users, then it is
    // trivially dead - remove it.

    // TODO: Call the constant folding hook on this operation, and canonicalize
    // constants into the entry node.

    // Check to see if we have any patterns that match this node.
    auto match = matcher.findMatch(op);
    if (!match.first)
      continue;

    // TODO: Need to be a bit trickier to make sure new instructions get into
    // the worklist.
    match.first->rewrite(op, std::move(match.second), builder);
  }
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
