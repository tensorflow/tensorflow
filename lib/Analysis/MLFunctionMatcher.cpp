//===- MLFunctionMatcher.cpp - MLFunctionMatcher Impl  ----------*- C++ -*-===//
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

#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/StandardOps/StandardOps.h"

#include "llvm/Support/Allocator.h"

namespace mlir {

/// Underlying storage for MLFunctionMatches.
struct MLFunctionMatchesStorage {
  MLFunctionMatchesStorage(MLFunctionMatches::EntryType e) : matches({e}) {}

  SmallVector<MLFunctionMatches::EntryType, 8> matches;
};

/// Underlying storage for MLFunctionMatcher.
struct MLFunctionMatcherStorage {
  MLFunctionMatcherStorage(Instruction::Kind k,
                           MutableArrayRef<MLFunctionMatcher> c,
                           FilterFunctionType filter, Instruction *skip)
      : kind(k), childrenMLFunctionMatchers(c.begin(), c.end()), filter(filter),
        skip(skip) {}

  Instruction::Kind kind;
  SmallVector<MLFunctionMatcher, 4> childrenMLFunctionMatchers;
  FilterFunctionType filter;
  /// skip is needed so that we can implement match without switching on the
  /// type of the Instruction.
  /// The idea is that a MLFunctionMatcher first checks if it matches locally
  /// and then recursively applies its children matchers to its elem->children.
  /// Since we want to rely on the InstWalker impl rather than duplicate its
  /// the logic, we allow an off-by-one traversal to account for the fact that
  /// we write:
  ///
  ///  void match(Instruction *elem) {
  ///    for (auto &c : getChildrenMLFunctionMatchers()) {
  ///      MLFunctionMatcher childMLFunctionMatcher(...);
  ///                                                ^~~~ Needs off-by-one skip.
  ///
  Instruction *skip;
};

} // end namespace mlir

using namespace mlir;

llvm::BumpPtrAllocator *&MLFunctionMatches::allocator() {
  static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

void MLFunctionMatches::append(Instruction *inst, MLFunctionMatches children) {
  if (!storage) {
    storage = allocator()->Allocate<MLFunctionMatchesStorage>();
    new (storage) MLFunctionMatchesStorage(std::make_pair(inst, children));
  } else {
    storage->matches.push_back(std::make_pair(inst, children));
  }
}
MLFunctionMatches::iterator MLFunctionMatches::begin() {
  return storage ? storage->matches.begin() : nullptr;
}
MLFunctionMatches::iterator MLFunctionMatches::end() {
  return storage ? storage->matches.end() : nullptr;
}
MLFunctionMatches::EntryType &MLFunctionMatches::front() {
  assert(storage && "null storage");
  return *storage->matches.begin();
}
MLFunctionMatches::EntryType &MLFunctionMatches::back() {
  assert(storage && "null storage");
  return *(storage->matches.begin() + size() - 1);
}
/// Return the combination of multiple MLFunctionMatches as a new object.
static MLFunctionMatches combine(ArrayRef<MLFunctionMatches> matches) {
  MLFunctionMatches res;
  for (auto s : matches) {
    for (auto ss : s) {
      res.append(ss.first, ss.second);
    }
  }
  return res;
}

/// Calls walk on `function`.
MLFunctionMatches MLFunctionMatcher::match(Function *function) {
  assert(!matches && "MLFunctionMatcher already matched!");
  this->walkPostOrder(function);
  return matches;
}

/// Calls walk on `instruction`.
MLFunctionMatches MLFunctionMatcher::match(Instruction *instruction) {
  assert(!matches && "MLFunctionMatcher already matched!");
  this->walkPostOrder(instruction);
  return matches;
}

unsigned MLFunctionMatcher::getDepth() {
  auto children = getChildrenMLFunctionMatchers();
  if (children.empty()) {
    return 1;
  }
  unsigned depth = 0;
  for (auto &c : children) {
    depth = std::max(depth, c.getDepth());
  }
  return depth + 1;
}

/// Matches a single instruction in the following way:
///   1. checks the kind of instruction against the matcher, if different then
///      there is no match;
///   2. calls the customizable filter function to refine the single instruction
///      match with extra semantic constraints;
///   3. if all is good, recursivey matches the children patterns;
///   4. if all children match then the single instruction matches too and is
///      appended to the list of matches;
///   5. TODO(ntv) Optionally applies actions (lambda), in which case we will
///      want to traverse in post-order DFS to avoid invalidating iterators.
void MLFunctionMatcher::matchOne(Instruction *elem) {
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
  SmallVector<MLFunctionMatches, 8> childrenMLFunctionMatches;
  for (auto &c : getChildrenMLFunctionMatchers()) {
    /// We create a new childMLFunctionMatcher here because a matcher holds its
    /// results. So we concretely need multiple copies of a given matcher, one
    /// for each matching result.
    MLFunctionMatcher childMLFunctionMatcher = forkMLFunctionMatcherAt(c, elem);
    childMLFunctionMatcher.walkPostOrder(elem);
    if (!childMLFunctionMatcher.matches) {
      return;
    }
    childrenMLFunctionMatches.push_back(childMLFunctionMatcher.matches);
  }
  matches.append(elem, combine(childrenMLFunctionMatches));
}

llvm::BumpPtrAllocator *&MLFunctionMatcher::allocator() {
  static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

MLFunctionMatcher::MLFunctionMatcher(Instruction::Kind k,
                                     MLFunctionMatcher child,
                                     FilterFunctionType filter)
    : storage(allocator()->Allocate<MLFunctionMatcherStorage>()) {
  // Initialize with placement new.
  new (storage)
      MLFunctionMatcherStorage(k, {child}, filter, nullptr /* skip */);
}

MLFunctionMatcher::MLFunctionMatcher(
    Instruction::Kind k, MutableArrayRef<MLFunctionMatcher> children,
    FilterFunctionType filter)
    : storage(allocator()->Allocate<MLFunctionMatcherStorage>()) {
  // Initialize with placement new.
  new (storage)
      MLFunctionMatcherStorage(k, children, filter, nullptr /* skip */);
}

MLFunctionMatcher
MLFunctionMatcher::forkMLFunctionMatcherAt(MLFunctionMatcher tmpl,
                                           Instruction *inst) {
  MLFunctionMatcher res(tmpl.getKind(), tmpl.getChildrenMLFunctionMatchers(),
                        tmpl.getFilterFunction());
  res.storage->skip = inst;
  return res;
}

Instruction::Kind MLFunctionMatcher::getKind() { return storage->kind; }

MutableArrayRef<MLFunctionMatcher>
MLFunctionMatcher::getChildrenMLFunctionMatchers() {
  return storage->childrenMLFunctionMatchers;
}

FilterFunctionType MLFunctionMatcher::getFilterFunction() {
  return storage->filter;
}

namespace mlir {
namespace matcher {

MLFunctionMatcher Op(FilterFunctionType filter) {
  return MLFunctionMatcher(Instruction::Kind::OperationInst, {}, filter);
}

MLFunctionMatcher If(MLFunctionMatcher child) {
  return MLFunctionMatcher(Instruction::Kind::If, child, defaultFilterFunction);
}
MLFunctionMatcher If(FilterFunctionType filter, MLFunctionMatcher child) {
  return MLFunctionMatcher(Instruction::Kind::If, child, filter);
}
MLFunctionMatcher If(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Instruction::Kind::If, children,
                           defaultFilterFunction);
}
MLFunctionMatcher If(FilterFunctionType filter,
                     MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Instruction::Kind::If, children, filter);
}

MLFunctionMatcher For(MLFunctionMatcher child) {
  return MLFunctionMatcher(Instruction::Kind::For, child,
                           defaultFilterFunction);
}
MLFunctionMatcher For(FilterFunctionType filter, MLFunctionMatcher child) {
  return MLFunctionMatcher(Instruction::Kind::For, child, filter);
}
MLFunctionMatcher For(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Instruction::Kind::For, children,
                           defaultFilterFunction);
}
MLFunctionMatcher For(FilterFunctionType filter,
                      MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Instruction::Kind::For, children, filter);
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
