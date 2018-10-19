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
  MLFunctionMatcherStorage(Statement::Kind k,
                           MutableArrayRef<MLFunctionMatcher> c,
                           FilterFunctionType filter)
      : kind(k), childrenMLFunctionMatchers(c.begin(), c.end()),
        filter(filter) {}

  Statement::Kind kind;
  SmallVector<MLFunctionMatcher, 4> childrenMLFunctionMatchers;
  FilterFunctionType filter;
};

} // end namespace mlir

using namespace mlir;

llvm::BumpPtrAllocator *&MLFunctionMatches::allocator() {
  static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

void MLFunctionMatches::append(Statement *stmt, MLFunctionMatches children) {
  if (!storage) {
    storage = allocator()->Allocate<MLFunctionMatchesStorage>();
    new (storage) MLFunctionMatchesStorage(std::make_pair(stmt, children));
  } else {
    storage->matches.push_back(std::make_pair(stmt, children));
  }
}
MLFunctionMatches::iterator MLFunctionMatches::begin() {
  return storage->matches.begin();
}
MLFunctionMatches::iterator MLFunctionMatches::end() {
  return storage->matches.end();
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
MLFunctionMatches &MLFunctionMatcher::match(MLFunction *function) {
  assert(!matches && "MLFunctionMatcher already matched!");
  this->walk(function);
  return matches;
}

/// Calls walk on `statement`.
MLFunctionMatches &MLFunctionMatcher::match(Statement *statement) {
  assert(!matches && "MLFunctionMatcher already matched!");
  this->walk(statement);
  return matches;
}

/// matchOrSkipOne is needed so that we can implement match without switching on
/// the type of the Statement.
/// The idea is that a MLFunctionMatcher first checks if it matches locally and
/// then recursively applies its children matchers to its elem->children.
/// Since we want to rely on the StmtWalker impl rather than duplicate its
/// the logic, we allow an off-by-one traversal to account for the fact that
/// we write:
///
///  void match(Statement *elem) {
///    for (auto &c : getChildrenMLFunctionMatchers()) {
///      MLFunctionMatcher childMLFunctionMatcher(...);
///      childMLFunctionMatcher.walk(elem); <~~~ Needs off-by-one traversal.
///
void MLFunctionMatcher::matchOrSkipOne(Statement *elem) {
  if (skipOne) {
    skipOne = false;
    return;
  }
  matchOne(elem);
}

/// Matches a single statement in the following way:
///   1. checks the kind of statement against the matcher, if different then
///      there is no match;
///   2. calls the customizable filter function to refine the single statement
///      match with extra semantic constraints;
///   3. if all is good, recursivey matches the children patterns;
///   4. if all children match then the single statement matches too and is
///      appended to the list of matches;
///   5. TODO(ntv) Optionally applies actions (lambda), in which case we will
///      want to traverse in post-order DFS to avoid invalidating iterators.
void MLFunctionMatcher::matchOne(Statement *elem) {
  // Structural filter
  if (elem->getKind() != getKind()) {
    return;
  }
  // Local custom filter function
  if (!getFilterFunction()(elem)) {
    return;
  }
  SmallVector<MLFunctionMatches, 8> childrenMLFunctionMatches;
  for (auto &c : getChildrenMLFunctionMatchers()) {
    /// We create a new childMLFunctionMatcher here because a matcher holds its
    /// results So we concretely need multiple copies of a given matcher, one
    /// for each matching result.
    MLFunctionMatcher childMLFunctionMatcher = forkMLFunctionMatcher(c);
    childMLFunctionMatcher.walk(elem);
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

MLFunctionMatcher::MLFunctionMatcher(Statement::Kind k, MLFunctionMatcher child,
                                     FilterFunctionType filter)
    : storage(allocator()->Allocate<MLFunctionMatcherStorage>()),
      skipOne(false) {
  // Initialize with placement new.
  new (storage) MLFunctionMatcherStorage(k, {child}, filter);
}

MLFunctionMatcher::MLFunctionMatcher(
    Statement::Kind k, MutableArrayRef<MLFunctionMatcher> children,
    FilterFunctionType filter)
    : storage(allocator()->Allocate<MLFunctionMatcherStorage>()),
      skipOne(false) {
  // Initialize with placement new.
  new (storage) MLFunctionMatcherStorage(k, children, filter);
}

MLFunctionMatcher
MLFunctionMatcher::forkMLFunctionMatcher(MLFunctionMatcher tmpl) {
  MLFunctionMatcher res(tmpl.getKind(), tmpl.getChildrenMLFunctionMatchers(),
                        tmpl.getFilterFunction());
  res.skipOne = true;
  return res;
}

Statement::Kind MLFunctionMatcher::getKind() { return storage->kind; }

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
  return MLFunctionMatcher(Statement::Kind::Operation, {}, filter);
}

MLFunctionMatcher If(MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::If, child, defaultFilterFunction);
}
MLFunctionMatcher If(FilterFunctionType filter, MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::If, child, filter);
}
MLFunctionMatcher If(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::If, children,
                           defaultFilterFunction);
}
MLFunctionMatcher If(FilterFunctionType filter,
                     MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::If, children, filter);
}

MLFunctionMatcher For(MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::For, child, defaultFilterFunction);
}
MLFunctionMatcher For(FilterFunctionType filter, MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::For, child, filter);
}
MLFunctionMatcher For(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::For, children,
                           defaultFilterFunction);
}
MLFunctionMatcher For(FilterFunctionType filter,
                      MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::For, children, filter);
}

// TODO(ntv): parallel annotation on loops.
FilterFunctionType isParallelLoop = [](Statement *stmt) {
  auto *loop = cast<ForStmt>(stmt);
  return (void *)loop || true; // loop->isParallel();
};
MLFunctionMatcher Doall(MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::For, child, isParallelLoop);
}
MLFunctionMatcher Doall(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::For, children, isParallelLoop);
}

// TODO(ntv): reduction annotation on loops.
FilterFunctionType isReductionLoop = [](Statement *stmt) {
  auto *loop = cast<ForStmt>(stmt);
  return (void *)loop || true; // loop->isReduction();
};
MLFunctionMatcher Red(MLFunctionMatcher child) {
  return MLFunctionMatcher(Statement::Kind::For, child, isReductionLoop);
}
MLFunctionMatcher Red(MutableArrayRef<MLFunctionMatcher> children) {
  return MLFunctionMatcher(Statement::Kind::For, children, isReductionLoop);
}

FilterFunctionType isLoadOrStore = [](Statement *stmt) {
  auto *opStmt = dyn_cast<OperationStmt>(stmt);
  return opStmt && (opStmt->isa<LoadOp>() || opStmt->isa<StoreOp>());
};
MLFunctionMatcher LoadStores() {
  return MLFunctionMatcher(Statement::Kind::Operation, {}, isLoadOrStore);
}

} // end namespace matcher
} // end namespace mlir
