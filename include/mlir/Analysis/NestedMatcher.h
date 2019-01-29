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
#include <utility>

namespace mlir {

struct NestedPatternStorage;
struct NestedMatchStorage;
class Instruction;

/// An NestedPattern captures nested patterns. It is used in conjunction with
/// a scoped NestedPatternContext which is an llvm::BumPtrAllocator that
/// handles memory allocations efficiently and avoids ownership issues.
///
/// In order to use NestedPatterns, first create a scoped context. When the
/// context goes out of scope, everything is freed.
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

/// Recursive abstraction for matching results.
/// Provides iteration over the Instruction* captured by a Matcher.
///
/// Implemented as a POD value-type with underlying storage pointer.
/// The underlying storage lives in a scoped bumper allocator whose lifetime
/// is managed by an RAII NestedPatternContext.
/// This is used by value everywhere.
struct NestedMatch {
  using EntryType = std::pair<Instruction *, NestedMatch>;
  using iterator = EntryType *;

  static NestedMatch build(ArrayRef<NestedMatch::EntryType> elements = {});
  NestedMatch(const NestedMatch &) = default;
  NestedMatch &operator=(const NestedMatch &) = default;

  explicit operator bool() { return !empty(); }

  iterator begin();
  iterator end();
  EntryType &front();
  EntryType &back();
  unsigned size() { return end() - begin(); }
  unsigned empty() { return size() == 0; }

private:
  friend class NestedPattern;
  friend class NestedPatternContext;
  friend class NestedMatchStorage;

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  NestedMatch(NestedMatchStorage *storage) : storage(storage){};

  /// Copy the specified array of elements into memory managed by our bump
  /// pointer allocator. The elements are all PODs by constructions.
  static NestedMatch copyInto(ArrayRef<NestedMatch::EntryType> elements);

  /// POD payload.
  NestedMatchStorage *storage;
};

/// A NestedPattern is a special type of InstWalker that:
///   1. recursively matches a substructure in the tree;
///   2. uses a filter function to refine matches with extra semantic
///      constraints (passed via a lambda of type FilterFunctionType);
///   3. TODO(ntv) Optionally applies actions (lambda).
///
/// Implemented as a POD value-type with underlying storage pointer.
/// The underlying storage lives in a scoped bumper allocator whose lifetime
/// is managed by an RAII NestedPatternContext.
/// This should be used by value everywhere.
using FilterFunctionType = std::function<bool(const Instruction &)>;
static bool defaultFilterFunction(const Instruction &) { return true; };
struct NestedPattern : public InstWalker<NestedPattern> {
  NestedPattern(Instruction::Kind k, ArrayRef<NestedPattern> nested,
                FilterFunctionType filter = defaultFilterFunction);
  NestedPattern(const NestedPattern &) = default;
  NestedPattern &operator=(const NestedPattern &) = default;

  /// Returns all the matches in `function`.
  NestedMatch match(Function *function);

  /// Returns all the matches nested under `instruction`.
  NestedMatch match(Instruction *instruction);

  unsigned getDepth();

private:
  friend class NestedPatternContext;
  friend InstWalker<NestedPattern>;

  /// Underlying global bump allocator managed by a NestedPatternContext.
  static llvm::BumpPtrAllocator *&allocator();

  Instruction::Kind getKind();
  ArrayRef<NestedPattern> getNestedPatterns();
  FilterFunctionType getFilterFunction();

  void matchOne(Instruction *elem);

  void visitForInst(ForInst *forInst) { matchOne(forInst); }
  void visitOperationInst(OperationInst *opInst) { matchOne(opInst); }

  /// POD paylod.
  /// Storage for the PatternMatcher.
  NestedPatternStorage *storage;

  // By-value POD wrapper to underlying storage pointer.
  NestedMatch matches;
};

/// RAII structure to transparently manage the bump allocator for
/// NestedPattern and NestedMatch classes.
struct NestedPatternContext {
  NestedPatternContext() {
    NestedPattern::allocator() = &allocator;
    NestedMatch::allocator() = &allocator;
  }
  ~NestedPatternContext() {
    NestedPattern::allocator() = nullptr;
    NestedMatch::allocator() = nullptr;
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
