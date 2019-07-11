//===- PatternMatch.h - PatternMatcher classes -------==---------*- C++ -*-===//
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

#ifndef MLIR_PATTERNMATCHER_H
#define MLIR_PATTERNMATCHER_H

#include "mlir/IR/Builders.h"

namespace mlir {

class FuncOp;
class PatternRewriter;

//===----------------------------------------------------------------------===//
// PatternBenefit class
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
  bool isImpossibleToMatch() const { return *this == impossibleToMatch(); }

  /// If the corresponding pattern can match, return its benefit.  If the
  // corresponding pattern isImpossibleToMatch() then this aborts.
  unsigned short getBenefit() const;

  bool operator==(const PatternBenefit &rhs) const {
    return representation == rhs.representation;
  }
  bool operator!=(const PatternBenefit &rhs) const { return !(*this == rhs); }
  bool operator<(const PatternBenefit &rhs) const {
    return representation < rhs.representation;
  }

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

/// This is the type returned by a pattern match.  A match failure returns a
/// None value.  A match success returns a Some value with any state the pattern
/// may need to maintain (but may also be null).
using PatternMatchResult = Optional<std::unique_ptr<PatternState>>;

//===----------------------------------------------------------------------===//
// Pattern class
//===----------------------------------------------------------------------===//

/// Instances of Pattern can be matched against SSA IR.  These matches get used
/// in ways dependent on their subclasses and the driver doing the matching.
/// For example, RewritePatterns implement a rewrite from one matched pattern
/// to a replacement DAG tile.
class Pattern {
public:
  /// Return the benefit (the inverse of "cost") of matching this pattern.  The
  /// benefit of a Pattern is always static - rewrites that may have dynamic
  /// benefit can be instantiated multiple times (different Pattern instances)
  /// for each benefit that they may return, and be guarded by different match
  /// condition predicates.
  PatternBenefit getBenefit() const { return benefit; }

  /// Return the root node that this pattern matches.  Patterns that can
  /// match multiple root types are instantiated once per root.
  OperationName getRootKind() const { return rootKind; }

  //===--------------------------------------------------------------------===//
  // Implementation hooks for patterns to implement.
  //===--------------------------------------------------------------------===//

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().  On failure, this
  /// returns a None value.  On success it returns a (possibly null)
  /// pattern-specific state wrapped in an Optional.
  virtual PatternMatchResult match(Operation *op) const = 0;

  virtual ~Pattern() {}

  //===--------------------------------------------------------------------===//
  // Helper methods to simplify pattern implementations
  //===--------------------------------------------------------------------===//

  /// This method indicates that no match was found.
  static PatternMatchResult matchFailure() { return None; }

  /// This method indicates that a match was found and has the specified cost.
  PatternMatchResult
  matchSuccess(std::unique_ptr<PatternState> state = {}) const {
    return PatternMatchResult(std::move(state));
  }

protected:
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  Pattern(StringRef rootName, PatternBenefit benefit, MLIRContext *context);

private:
  const OperationName rootKind;
  const PatternBenefit benefit;

  virtual void anchor();
};

/// RewritePattern is the common base class for all DAG to DAG replacements.
/// There are two possible usages of this class:
///   * Multi-step RewritePattern with "match" and "rewrite"
///     - By overloading the "match" and "rewrite" functions, the user can
///       separate the concerns of matching and rewriting.
///   * Single-step RewritePattern with "matchAndRewrite"
///     - By overloading the "matchAndRewrite" function, the user can perform
///       the rewrite in the same call as the match. This removes the need for
///       any PatternState.
///
class RewritePattern : public Pattern {
public:
  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// rewriter.  If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, std::unique_ptr<PatternState> state,
                       PatternRewriter &rewriter) const;

  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// builder.  If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, PatternRewriter &rewriter) const;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().  On failure, this
  /// returns a None value.  On success, it returns a (possibly null)
  /// pattern-specific state wrapped in an Optional.  This state is passed back
  /// into the rewrite function if this match is selected.
  PatternMatchResult match(Operation *op) const override;

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind(). If successful, this
  /// function will automatically perform the rewrite.
  virtual PatternMatchResult matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
    if (auto matchResult = match(op)) {
      rewrite(op, std::move(*matchResult), rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }

  /// Return a list of operations that may be generated when rewriting an
  /// operation instance with this pattern.
  ArrayRef<OperationName> getGeneratedOps() const { return generatedOps; }

protected:
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  RewritePattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context)
      : Pattern(rootName, benefit, context) {}
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching. They can also specify
  /// the names of operations that may be generated during a successful rewrite.
  RewritePattern(StringRef rootName, ArrayRef<StringRef> generatedNames,
                 PatternBenefit benefit, MLIRContext *context);

  /// A list of the potential operations that may be generated when rewriting
  /// an op with this pattern.
  llvm::SmallVector<OperationName, 2> generatedOps;
};

/// OpRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp> struct OpRewritePattern : public RewritePattern {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching.
  OpRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(SourceOp::getOperationName(), benefit, context) {}

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  void rewrite(Operation *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const final {
    rewrite(llvm::cast<SourceOp>(op), std::move(state), rewriter);
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewrite(llvm::cast<SourceOp>(op), rewriter);
  }
  PatternMatchResult match(Operation *op) const final {
    return match(llvm::cast<SourceOp>(op));
  }
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const final {
    return matchAndRewrite(llvm::cast<SourceOp>(op), rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, std::unique_ptr<PatternState> state,
                       PatternRewriter &rewriter) const {
    rewrite(op, rewriter);
  }
  virtual void rewrite(SourceOp op, PatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
  virtual PatternMatchResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual PatternMatchResult matchAndRewrite(SourceOp op,
                                             PatternRewriter &rewriter) const {
    if (auto matchResult = match(op)) {
      rewrite(op, std::move(*matchResult), rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }
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
class PatternRewriter : public OpBuilder {
public:
  /// Create operation of specific op type at the current insertion point
  /// without verifying to see if it is valid.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args... args) {
    OperationState state(location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *op = createOperation(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Creates an operation of specific op type at the current insertion point.
  /// If the result is an invalid op (the verifier hook fails), emit an error
  /// and return null.
  template <typename OpTy, typename... Args>
  OpTy createChecked(Location location, Args... args) {
    OperationState state(location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *op = createOperation(state);

    // If the Operation we produce is valid, return it.
    if (!OpTy::verifyInvariants(op)) {
      auto result = dyn_cast<OpTy>(op);
      assert(result && "Builder didn't return the right type");
      return result;
    }

    // Otherwise, the error message got emitted.  Just remove the operation
    // we made.
    op->erase();
    return OpTy();
  }

  /// Move the blocks that belong to "region" before the given position in
  /// another region "parent".  The two regions must be different.  The caller
  /// is responsible for creating or updating the operation transferring flow
  // of control to the region and pass it the correct block arguments.
  virtual void inlineRegionBefore(Region &region, Region &parent,
                                  Region::iterator before);
  void inlineRegionBefore(Region &region, Block *before);

  /// This method performs the final replacement for a pattern, where the
  /// results of the operation are updated to use the specified list of SSA
  /// values.  In addition to replacing and removing the specified operation,
  /// clients can specify a list of other nodes that this replacement may make
  /// (perhaps transitively) dead.  If any of those values are dead, this will
  /// remove them as well.
  virtual void replaceOp(Operation *op, ArrayRef<Value *> newValues,
                         ArrayRef<Value *> valuesToRemoveIfDead = {});

  /// Replaces the result op with a new op that is created without verification.
  /// The result values of the two ops must be the same types.
  template <typename OpTy, typename... Args>
  void replaceOpWithNewOp(Operation *op, Args &&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOpWithResultsOfAnotherOp(op, newOp.getOperation(), {});
  }

  /// Replaces the result op with a new op that is created without verification.
  /// The result values of the two ops must be the same types.  This allows
  /// specifying a list of ops that may be removed if dead.
  template <typename OpTy, typename... Args>
  void replaceOpWithNewOp(ArrayRef<Value *> valuesToRemoveIfDead, Operation *op,
                          Args &&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOpWithResultsOfAnotherOp(op, newOp.getOperation(),
                                    valuesToRemoveIfDead);
  }

  /// Split the operations starting at "before" (inclusive) out of the given
  /// block into a new block, and return it.
  virtual Block *splitBlock(Block *block, Block::iterator before) {
    return block->splitBlock(before);
  }

  /// This method is used as the final notification hook for patterns that end
  /// up modifying the pattern root in place, by changing its operands.  This is
  /// a minor efficiency win (it avoids creating a new operation and removing
  /// the old one) but also often allows simpler code in the client.
  ///
  /// The valuesToRemoveIfDead list is an optional list of values that the
  /// rewriter should remove if they are dead at this point.
  ///
  void updatedRootInPlace(Operation *op,
                          ArrayRef<Value *> valuesToRemoveIfDead = {});

protected:
  PatternRewriter(Region &region) : OpBuilder(region) {}
  virtual ~PatternRewriter();

  // These are the callback methods that subclasses can choose to implement if
  // they would like to be notified about certain types of mutations.

  /// This is implemented to create the specified operations and serves as a
  /// notification hook for rewriters that want to know about new operations.
  virtual Operation *createOperation(const OperationState &state) = 0;

  /// Notify the pattern rewriter that the specified operation has been mutated
  /// in place.  This is called after the mutation is done.
  virtual void notifyRootUpdated(Operation *op) {}

  /// Notify the pattern rewriter that the specified operation is about to be
  /// replaced with another set of operations.  This is called before the uses
  /// of the operation have been changed.
  virtual void notifyRootReplaced(Operation *op) {}

  /// This is called on an operation that a pattern match is removing, right
  /// before the operation is deleted.  At this point, the operation has zero
  /// uses.
  virtual void notifyOperationRemoved(Operation *op) {}

private:
  /// op and newOp are known to have the same number of results, replace the
  /// uses of op with uses of newOp
  void replaceOpWithResultsOfAnotherOp(Operation *op, Operation *newOp,
                                       ArrayRef<Value *> valuesToRemoveIfDead);
};

//===----------------------------------------------------------------------===//
// Pattern-driven rewriters
//===----------------------------------------------------------------------===//

/// This is a vector that owns the patterns inside of it.
using OwningRewritePatternList = std::vector<std::unique_ptr<RewritePattern>>;

/// This class manages optimization and execution of a group of rewrite
/// patterns, providing an API for finding and applying, the best match against
/// a given node.
///
class RewritePatternMatcher {
public:
  /// Create a RewritePatternMatcher with the specified set of patterns.
  explicit RewritePatternMatcher(OwningRewritePatternList &&patterns);

  /// Try to match the given operation to a pattern and rewrite it. Return
  /// true if any pattern matches.
  bool matchAndRewrite(Operation *op, PatternRewriter &rewriter);

private:
  RewritePatternMatcher(const RewritePatternMatcher &) = delete;
  void operator=(const RewritePatternMatcher &) = delete;

  /// The group of patterns that are matched for optimization through this
  /// matcher.
  OwningRewritePatternList patterns;
};

/// Rewrite the specified function by repeatedly applying the highest benefit
/// patterns in a greedy work-list driven manner. Return true if no more
/// patterns can be matched in the result function.
///
bool applyPatternsGreedily(FuncOp fn, OwningRewritePatternList &&patterns);

/// Helper class to create a list of rewrite patterns given a list of their
/// types and a list of attributes perfect-forwarded to each of the conversion
/// constructors.
template <typename Arg, typename... Args> struct RewriteListBuilder {
  template <typename... ConstructorArgs>
  static void build(OwningRewritePatternList &patterns,
                    ConstructorArgs &&... constructorArgs) {
    RewriteListBuilder<Args...>::build(
        patterns, std::forward<ConstructorArgs>(constructorArgs)...);
    RewriteListBuilder<Arg>::build(
        patterns, std::forward<ConstructorArgs>(constructorArgs)...);
  }
};

// Template specialization to stop recursion.
template <typename Arg> struct RewriteListBuilder<Arg> {
  template <typename... ConstructorArgs>
  static void build(OwningRewritePatternList &patterns,
                    ConstructorArgs &&... constructorArgs) {
    patterns.emplace_back(llvm::make_unique<Arg>(
        std::forward<ConstructorArgs>(constructorArgs)...));
  }
};
} // end namespace mlir

#endif // MLIR_PATTERN_MATCH_H
