//===- NestedMatcher.cpp - NestedMatcher Impl  ------------------*- C++ -*-===//
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

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/StandardOps/StandardOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// Underlying storage for NestedMatch.
struct NestedMatchStorage {
  MutableArrayRef<NestedMatch::EntryType> matches;
};

/// Underlying storage for NestedPattern.
struct NestedPatternStorage {
  NestedPatternStorage(Instruction::Kind k, ArrayRef<NestedPattern> c,
                       FilterFunctionType filter, Instruction *skip)
      : kind(k), nestedPatterns(c), filter(filter), skip(skip) {}

  Instruction::Kind kind;
  ArrayRef<NestedPattern> nestedPatterns;
  FilterFunctionType filter;
  /// skip is needed so that we can implement match without switching on the
  /// type of the Instruction.
  /// The idea is that a NestedPattern first checks if it matches locally
  /// and then recursively applies its nested matchers to its elem->nested.
  /// Since we want to rely on the InstWalker impl rather than duplicate its
  /// the logic, we allow an off-by-one traversal to account for the fact that
  /// we write:
  ///
  ///  void match(Instruction *elem) {
  ///    for (auto &c : getNestedPatterns()) {
  ///      NestedPattern childPattern(...);
  ///                                     ^~~~ Needs off-by-one skip.
  ///
  Instruction *skip;
};

} // end namespace mlir

using namespace mlir;

llvm::BumpPtrAllocator *&NestedMatch::allocator() {
  static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

NestedMatch NestedMatch::build(ArrayRef<NestedMatch::EntryType> elements) {
  auto *matches =
      allocator()->Allocate<NestedMatch::EntryType>(elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(), matches);
  auto *storage = allocator()->Allocate<NestedMatchStorage>();
  new (storage) NestedMatchStorage();
  storage->matches =
      MutableArrayRef<NestedMatch::EntryType>(matches, elements.size());
  auto *result = allocator()->Allocate<NestedMatch>();
  new (result) NestedMatch(storage);
  return *result;
}

NestedMatch::iterator NestedMatch::begin() { return storage->matches.begin(); }
NestedMatch::iterator NestedMatch::end() { return storage->matches.end(); }
NestedMatch::EntryType &NestedMatch::front() {
  return *storage->matches.begin();
}
NestedMatch::EntryType &NestedMatch::back() {
  return *(storage->matches.begin() + size() - 1);
}

/// Calls walk on `function`.
NestedMatch NestedPattern::match(Function *function) {
  assert(!matches && "NestedPattern already matched!");
  this->walkPostOrder(function);
  return matches;
}

/// Calls walk on `instruction`.
NestedMatch NestedPattern::match(Instruction *instruction) {
  assert(!matches && "NestedPattern already matched!");
  this->walkPostOrder(instruction);
  return matches;
}

unsigned NestedPattern::getDepth() {
  auto nested = getNestedPatterns();
  if (nested.empty()) {
    return 1;
  }
  unsigned depth = 0;
  for (auto c : nested) {
    depth = std::max(depth, c.getDepth());
  }
  return depth + 1;
}

/// Matches a single instruction in the following way:
///   1. checks the kind of instruction against the matcher, if different then
///      there is no match;
///   2. calls the customizable filter function to refine the single instruction
///      match with extra semantic constraints;
///   3. if all is good, recursivey matches the nested patterns;
///   4. if all nested match then the single instruction matches too and is
///      appended to the list of matches;
///   5. TODO(ntv) Optionally applies actions (lambda), in which case we will
///      want to traverse in post-order DFS to avoid invalidating iterators.
void NestedPattern::matchOne(Instruction *elem) {
  if (storage->skip == elem) {
    return;
  }
  // Structural filter
  if (elem->getKind() != getKind()) {
    return;
  }
  // Local custom filter function
  if (!getFilterFunction()(*elem)) {
    return;
  }

  SmallVector<NestedMatch::EntryType, 8> nestedEntries;
  for (auto c : getNestedPatterns()) {
    /// We create a new nestedPattern here because a matcher holds its
    /// results. So we concretely need multiple copies of a given matcher, one
    /// for each matching result.
    NestedPattern nestedPattern(c);
    // Skip elem in the walk immediately following. Without this we would
    // essentially need to reimplement walkPostOrder here.
    nestedPattern.storage->skip = elem;
    nestedPattern.walkPostOrder(elem);
    if (!nestedPattern.matches) {
      return;
    }
    for (auto m : nestedPattern.matches) {
      nestedEntries.push_back(m);
    }
  }

  SmallVector<NestedMatch::EntryType, 8> newEntries(
      matches.storage->matches.begin(), matches.storage->matches.end());
  newEntries.push_back(std::make_pair(elem, NestedMatch::build(nestedEntries)));
  matches = NestedMatch::build(newEntries);
}

llvm::BumpPtrAllocator *&NestedPattern::allocator() {
  static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

NestedPattern::NestedPattern(Instruction::Kind k,
                             ArrayRef<NestedPattern> nested,
                             FilterFunctionType filter)
    : storage(allocator()->Allocate<NestedPatternStorage>()),
      matches(NestedMatch::build({})) {
  auto *newChildren = allocator()->Allocate<NestedPattern>(nested.size());
  std::uninitialized_copy(nested.begin(), nested.end(), newChildren);
  // Initialize with placement new.
  new (storage) NestedPatternStorage(
      k, ArrayRef<NestedPattern>(newChildren, nested.size()), filter,
      nullptr /* skip */);
}

Instruction::Kind NestedPattern::getKind() { return storage->kind; }

ArrayRef<NestedPattern> NestedPattern::getNestedPatterns() {
  return storage->nestedPatterns;
}

FilterFunctionType NestedPattern::getFilterFunction() {
  return storage->filter;
}

namespace mlir {
namespace matcher {

NestedPattern Op(FilterFunctionType filter) {
  return NestedPattern(Instruction::Kind::OperationInst, {}, filter);
}

NestedPattern If(NestedPattern child) {
  return NestedPattern(Instruction::Kind::If, child, defaultFilterFunction);
}
NestedPattern If(FilterFunctionType filter, NestedPattern child) {
  return NestedPattern(Instruction::Kind::If, child, filter);
}
NestedPattern If(ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::If, nested, defaultFilterFunction);
}
NestedPattern If(FilterFunctionType filter, ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::If, nested, filter);
}

NestedPattern For(NestedPattern child) {
  return NestedPattern(Instruction::Kind::For, child, defaultFilterFunction);
}
NestedPattern For(FilterFunctionType filter, NestedPattern child) {
  return NestedPattern(Instruction::Kind::For, child, filter);
}
NestedPattern For(ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::For, nested, defaultFilterFunction);
}
NestedPattern For(FilterFunctionType filter, ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::For, nested, filter);
}

// TODO(ntv): parallel annotation on loops.
bool isParallelLoop(const Instruction &inst) {
  const auto *loop = cast<ForInst>(&inst);
  return (void *)loop || true; // loop->isParallel();
};

// TODO(ntv): reduction annotation on loops.
bool isReductionLoop(const Instruction &inst) {
  const auto *loop = cast<ForInst>(&inst);
  return (void *)loop || true; // loop->isReduction();
};

bool isLoadOrStore(const Instruction &inst) {
  const auto *opInst = dyn_cast<OperationInst>(&inst);
  return opInst && (opInst->isa<LoadOp>() || opInst->isa<StoreOp>());
};

} // end namespace matcher
} // end namespace mlir
