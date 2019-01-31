//===- NestedMacher.h - Nested matcher for MLFunction -----------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_
#define MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_

#include "mlir/IR/InstVisitor.h"
#include "llvm/Support/Allocator.h"

namespace mlir {

struct NestedPattern;
class Instruction;

/// An NestedPattern captures nested patterns in the IR.
/// It is used in conjunction with a scoped NestedPatternContext which is an
/// llvm::BumpPtrAllocator that handles memory allocations efficiently and
/// avoids ownership issues.
///
/// In order to use NestedPatterns, first create a scoped context.
/// When the context goes out of scope, everything is freed.
/// This design simplifies the API by avoiding references to the context and
/// makes it clear that references to matchers must not escape.
///
/// Example:
///   {
///      NestedPatternContext context;
///      auto gemmLike = Doall(Doall(Red(LoadStores())));
///      auto matches = gemmLike.match(f);
///      // do work on matches
///   }  // everything is freed
///
///
/// Nested abstraction for matching results.
/// Provides access to the nested Instruction* captured by a Matcher.
///
/// A NestedMatch contains an Instruction* and the children NestedMatch and is
/// thus cheap to copy. NestedMatch is stored in a scoped bumper allocator whose
/// lifetime is managed by an RAII NestedPatternContext.
struct NestedMatch {
  static NestedMatch build(Instruction *instruction,
                           ArrayRef<NestedMatch> nestedMatches);
  NestedMatch(const NestedMatch &) = default;
  NestedMatch &operator=(const NestedMatch &) = default;

  explicit operator bool() { return matchedInstruction != nullptr; }

  Instruction *getMatchedInstruction() { return matchedInstruction; }
  ArrayRef<NestedMatch> getMatchedChildren() { return matchedChildren; }

private:
  friend class NestedPattern;
  friend class NestedPatternContext;

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  NestedMatch() = default;

  /// Payload, holds a NestedMatch and all its children along this branch.
  Instruction *matchedInstruction;
  ArrayRef<NestedMatch> matchedChildren;
};

/// A NestedPattern is a nested InstWalker that:
///   1. recursively matches a substructure in the tree;
///   2. uses a filter function to refine matches with extra semantic
///      constraints (passed via a lambda of type FilterFunctionType);
///   3. TODO(ntv) optionally applies actions (lambda).
///
/// Nested patterns are meant to capture imperfectly nested loops while matching
/// properties over the whole loop nest. For instance, in vectorization we are
/// interested in capturing all the imperfectly nested loops of a certain type
/// and such that all the load and stores have certain access patterns along the
/// loops' induction variables). Such NestedMatches are first captured using the
/// `match` function and are later processed to analyze properties and apply
/// transformations in a non-greedy way.
///
/// The NestedMatches captured in the IR can grow large, especially after
/// aggressive unrolling. As experience has shown, it is generally better to use
/// a plain InstWalker to match flat patterns but the current implementation is
/// competitive nonetheless.
using FilterFunctionType = std::function<bool(const Instruction &)>;
static bool defaultFilterFunction(const Instruction &) { return true; };
struct NestedPattern {
  NestedPattern(Instruction::Kind k, ArrayRef<NestedPattern> nested,
                FilterFunctionType filter = defaultFilterFunction);
  NestedPattern(const NestedPattern &) = default;
  NestedPattern &operator=(const NestedPattern &) = default;

  /// Returns all the top-level matches in `function`.
  void match(Function *function, SmallVectorImpl<NestedMatch> *matches) {
    State state(*this, matches);
    state.walkPostOrder(function);
  }

  /// Returns all the top-level matches in `inst`.
  void match(Instruction *inst, SmallVectorImpl<NestedMatch> *matches) {
    State state(*this, matches);
    state.walkPostOrder(inst);
  }

  /// Returns the depth of the pattern.
  unsigned getDepth() const;

private:
  friend class NestedPatternContext;
  friend class NestedMatch;
  friend class InstWalker<NestedPattern>;
  friend struct State;

  /// Helper state that temporarily holds matches for the next level of nesting.
  struct State : public InstWalker<State> {
    State(NestedPattern &pattern, SmallVectorImpl<NestedMatch> *matches)
        : pattern(pattern), matches(matches) {}
    void visitForInst(ForInst *forInst) { pattern.matchOne(forInst, matches); }
    void visitOperationInst(OperationInst *opInst) {
      pattern.matchOne(opInst, matches);
    }

  private:
    NestedPattern &pattern;
    SmallVectorImpl<NestedMatch> *matches;
  };

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  /// Matches this pattern against a single `inst` and fills matches with the
  /// result.
  void matchOne(Instruction *inst, SmallVectorImpl<NestedMatch> *matches);

  /// Instruction kind matched by this pattern.
  Instruction::Kind kind;

  /// Nested patterns to be matched.
  ArrayRef<NestedPattern> nestedPatterns;

  /// Extra filter function to apply to prune patterns as the IR is walked.
  FilterFunctionType filter;

  /// skip is an implementation detail needed so that we can implement match
  /// without switching on the type of the Instruction. The idea is that a
  /// NestedPattern first checks if it matches locally and then recursively
  /// applies its nested matchers to its elem->nested. Since we want to rely on
  /// the InstWalker impl rather than duplicate its the logic, we allow an
  /// off-by-one traversal to account for the fact that we write:
  ///
  ///  void match(Instruction *elem) {
  ///    for (auto &c : getNestedPatterns()) {
  ///      NestedPattern childPattern(...);
  ///                                  ^~~~ Needs off-by-one skip.
  ///
  Instruction *skip;
};

/// RAII structure to transparently manage the bump allocator for
/// NestedPattern and NestedMatch classes. This avoids passing a context to
/// all the API functions.
struct NestedPatternContext {
  NestedPatternContext() {
    assert(NestedMatch::allocator() == nullptr &&
           "Only a single NestedPatternContext is supported");
    assert(NestedPattern::allocator() == nullptr &&
           "Only a single NestedPatternContext is supported");
    NestedMatch::allocator() = &allocator;
    NestedPattern::allocator() = &allocator;
  }
  ~NestedPatternContext() {
    NestedMatch::allocator() = nullptr;
    NestedPattern::allocator() = nullptr;
  }
  llvm::BumpPtrAllocator allocator;
};

namespace matcher {
// Syntactic sugar NestedPattern builder functions.
NestedPattern Op(FilterFunctionType filter = defaultFilterFunction);
NestedPattern If(NestedPattern child);
NestedPattern If(FilterFunctionType filter, NestedPattern child);
NestedPattern If(ArrayRef<NestedPattern> nested = {});
NestedPattern If(FilterFunctionType filter,
                 ArrayRef<NestedPattern> nested = {});
NestedPattern For(NestedPattern child);
NestedPattern For(FilterFunctionType filter, NestedPattern child);
NestedPattern For(ArrayRef<NestedPattern> nested = {});
NestedPattern For(FilterFunctionType filter,
                  ArrayRef<NestedPattern> nested = {});

bool isParallelLoop(const Instruction &inst);
bool isReductionLoop(const Instruction &inst);
bool isLoadOrStore(const Instruction &inst);

} // end namespace matcher
} // end namespace mlir

#endif // MLIR_ANALYSIS_MLFUNCTIONMATCHER_H_
