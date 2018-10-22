//===- PatternMatch.h - Base classes for pattern match ----------*- C++ -*-===//
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

#ifndef MLIR_PATTERN_MATCH_H
#define MLIR_PATTERN_MATCH_H

#include "mlir/IR/Builders.h"

namespace mlir {

class PatternRewriter;

//===----------------------------------------------------------------------===//
// Definition of Pattern and related types.
//===----------------------------------------------------------------------===//

/// This class represents the benefit of a pattern match in a unitless scheme
/// that ranges from 0 (very little benefit) to 65K.  The most common unit to
/// use here is the "number of operations matched" by the pattern.
///
/// This also has a sentinel representation that can be used for patterns that
/// fail to match.
///
class PatternBenefit {
  enum { ImpossibleToMatchSentinel = 65535 };

public:
  /*implicit*/ PatternBenefit(unsigned benefit);
  PatternBenefit(const PatternBenefit &) = default;
  PatternBenefit &operator=(const PatternBenefit &) = default;

  static PatternBenefit impossibleToMatch() { return PatternBenefit(); }

  bool isImpossibleToMatch() const {
    return representation == ImpossibleToMatchSentinel;
  }

  /// If the corresponding pattern can match, return its benefit.  If the
  // corresponding pattern isImpossibleToMatch() then this aborts.
  unsigned short getBenefit() const;

  inline bool operator==(const PatternBenefit& other);
  inline bool operator!=(const PatternBenefit& other);

private:
  PatternBenefit() : representation(ImpossibleToMatchSentinel) {}
  unsigned short representation;
};

/// Pattern state is used by patterns that want to maintain state between their
/// match and rewrite phases.  Patterns can define a pattern-specific subclass
/// of this.
class PatternState {
public:
  virtual ~PatternState() {}

protected:
  // Must be subclassed.
  PatternState() {}
};

/// This is the type returned by a pattern match.  The first field indicates
/// the benefit of the match, the second is a state token that can optionally
/// be produced by a pattern match to maintain state between the match and
/// rewrite phases.
typedef std::pair<PatternBenefit, std::unique_ptr<PatternState>>
    PatternMatchResult;

class Pattern {
public:
  // Return the benefit (the inverse of "cost") of matching this pattern,
  // if it is statically determinable.  The result is a PatternBenefit if known,
  // or 'None' if the cost is dynamically computed.
  Optional<PatternBenefit> getStaticBenefit() const;

  // Return the root node that this pattern matches.  Patterns that can
  // match multiple root types are instantiated once per root.
  OperationName getRootKind() const;

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
                       PatternRewriter &rewriter) const;

  // Rewrite the IR rooted at the specified operation with the result of
  // this pattern, generating any new operations with the specified
  // builder.  If an unexpected error is encountered (an internal
  // compiler error), it is emitted through the normal MLIR diagnostic
  // hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, PatternRewriter &rewriter) const;

  virtual ~Pattern() {}

  //===--------------------------------------------------------------------===//
  // Helper methods to simplify pattern implementations
  //===--------------------------------------------------------------------===//

  /// This method indicates that no match was found.
  static PatternMatchResult matchFailure();

  /// This method indicates that a match was found and has the specified cost.
  PatternMatchResult
  matchSuccess(PatternBenefit benefit,
               std::unique_ptr<PatternState> state = {}) const;

  /// This method indicates that a match was found for patterns that have a
  /// known static benefit.
  PatternMatchResult
  matchSuccess(std::unique_ptr<PatternState> state = {}) const;

protected:
  /// Patterns must specify the root operation name they match against, and can
  /// also optionally specify a static benefit of matching.
  Pattern(OperationName rootKind,
          Optional<PatternBenefit> staticBenefit = llvm::None);

  Pattern(OperationName rootKind, unsigned staticBenefit);

private:
  const OperationName rootKind;
  const Optional<PatternBenefit> staticBenefit;
};

//===----------------------------------------------------------------------===//
// PatternRewriter class
//===----------------------------------------------------------------------===//

/// This class coordinates the application of a pattern to the current function,
/// providing a way to create operations and keep track of what gets deleted.
///
/// These class serves two purposes:
///  1) it is the interface that patterns interact with to make mutations to the
///     IR they are being applied to.
///  2) It is a base class that clients of the PatternMatcher use when they want
///     to apply patterns and observe their effects (e.g. to keep worklists or
///     other data structures up to date).
///
class PatternRewriter : public Builder {
public:
  /// Create operation of specific op type at the current insertion point
  /// without verifying to see if it is valid.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> create(Location *location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *op = createOperation(state);
    auto result = op->dyn_cast<OpTy>();
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Creates an operation of specific op type at the current insertion point.
  /// If the result is an invalid op (the verifier hook fails), emit an error
  /// and return null.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> createChecked(Location *location, Args... args) {
    OperationState state(context, location, OpTy::getOperationName());
    auto result = this->create(location, args...);

    // If the OperationInst we produce is valid, return it.
    if (!OpTy::verifyInvariants(result))
      return result;

    // Otherwise, the error message got emitted.  Just remove the instruction
    // we made.
    if (auto *inst = dyn_cast<OperationInst>(result.getOperation()))
      inst->eraseFromParent();
    else
      cast<OperationStmt>(result.getOperation())->eraseFromBlock();

    result->eraseFromBlock();
    return result;
  }

  /// This method is used as the final replacement hook for patterns that match
  /// a single result value.  In addition to replacing and removing the
  /// specified operation, clients can specify a list of other nodes that this
  /// replacement may make (perhaps transitively) dead.  If any of those ops are
  /// dead, this will remove them as well.
  void replaceSingleResultOp(Operation *op, SSAValue *newValue,
                             ArrayRef<SSAValue *> opsToRemoveIfDead = {});

protected:
  PatternRewriter(MLIRContext *context) : Builder(context) {}
  virtual ~PatternRewriter();

  // These are the callback methods that subclasses can choose to implement if
  // they would like to be notified about certain types of mutations.

  /// This is implemented to create the specified operations and serves as a
  /// notification hook for rewriters that want to know about new operations.
  virtual Operation *createOperation(const OperationState &state) = 0;

  /// Notify the pattern rewriter that the specified operation is about to be
  /// replaced with another set of operations.  This is called before the uses
  /// of the operation have been changed.
  virtual void notifyRootReplaced(Operation *op) {}

  /// This is called on an operation that a pattern match is removing, right
  /// before the operation is deleted.  At this point, the operation has zero
  /// uses.
  virtual void notifyOperationRemoved(Operation *op) {}
};

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

  ~PatternMatcher() { llvm::DeleteContainerPointers(patterns); }

private:
  PatternMatcher(const PatternMatcher &) = delete;
  void operator=(const PatternMatcher &) = delete;

  std::vector<Pattern *> patterns;
};
} // end namespace mlir

#endif // MLIR_PATTERN_MATCH_H
