//===- UseDefAnalysis.h - Analysis for Transitive UseDef chains -----------===//
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
// This file implements Analysis functions specific to slicing in MLFunction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/ADT/SetVector.h"
#include <type_traits>

///
/// Implements Analysis functions specific to slicing in MLFunction.
///

using namespace mlir;

using llvm::DenseSet;
using llvm::SetVector;

/// Implementation detail that walks up the parents and records the ones with
/// the specified type.
/// TODO(ntv): could also be implemented as a collect parents followed by a
/// filter and made available outside this file.
template <typename T>
static inline SetVector<T *> getParentsOfType(Statement *stmt) {
  SetVector<T *> res;
  auto *current = stmt;
  while (auto *parent = current->getParentStmt()) {
    auto *typedParent = dyn_cast<T>(parent);
    if (typedParent) {
      assert(res.count(typedParent) == 0 && "Already inserted");
      res.insert(typedParent);
    }
    current = parent;
  }
  return res;
}

// Returns the enclosing ForStmt, from closest to farthest.
// Use reverse iterators to get from outermost to innermost loop.
static inline SetVector<ForStmt *> getEnclosingForStmts(Statement *stmt) {
  return getParentsOfType<ForStmt>(stmt);
}

// Returns the enclosing IfStmt, from closest to farthest.
// Use reverse iterators to get from outermost to innermost if conditional.
static inline SetVector<IfStmt *> getEnclosingIfStmts(Statement *stmt) {
  return getParentsOfType<IfStmt>(stmt);
}

bool mlir::strictlyScopedUnder(Statement *stmt, Statement *scope) {
  if (auto *forStmt = dyn_cast<ForStmt>(scope)) {
    return getEnclosingForStmts(stmt).count(forStmt) > 0;
  }
  if (auto *ifStmt = dyn_cast<IfStmt>(scope)) {
    return getEnclosingIfStmts(stmt).count(ifStmt) > 0;
  }
  auto *opStmt = cast<OperationStmt>(scope);
  (void)opStmt;
  assert(false && "NYI: domination by an OpertationStmt");
  return false;
}

void mlir::getForwardStaticSlice(Statement *stmt,
                                 SetVector<Statement *> *forwardStaticSlice,
                                 TransitiveFilter filter, bool topLevel) {
  if (!stmt) {
    return;
  }

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardStaticSlice in the current scope.
  if (!filter(stmt)) {
    return;
  }

  if (auto *opStmt = dyn_cast<OperationStmt>(stmt)) {
    assert(opStmt->getNumResults() <= 1 && "NYI: multiple results");
    if (opStmt->getNumResults() > 0) {
      for (auto &u : opStmt->getResult(0)->getUses()) {
        auto *ownerStmt = u.getOwner();
        if (forwardStaticSlice->count(ownerStmt) == 0) {
          getForwardStaticSlice(ownerStmt, forwardStaticSlice, filter,
                                /* topLevel */ false);
        }
      }
    }
  } else if (auto *forStmt = dyn_cast<ForStmt>(stmt)) {
    for (auto &u : forStmt->getUses()) {
      auto *ownerStmt = u.getOwner();
      if (forwardStaticSlice->count(ownerStmt) == 0) {
        getForwardStaticSlice(ownerStmt, forwardStaticSlice, filter,
                              /* topLevel */ false);
      }
    }
  } else {
    assert(false && "NYI slicing case");
  }

  // At the top level we reverse to get back the actual topological order.
  if (topLevel) {
    // std::reverse does not work out of the box on SetVector and I want an
    // in-place swap based thing (the real std::reverse, not the LLVM adapter).
    // TODO(clattner): Consider adding an extra method?
    std::vector<Statement *> v(forwardStaticSlice->takeVector());
    forwardStaticSlice->insert(v.rbegin(), v.rend());
  } else {
    forwardStaticSlice->insert(stmt);
  }
}

void mlir::getBackwardStaticSlice(Statement *stmt,
                                  SetVector<Statement *> *backwardStaticSlice,
                                  TransitiveFilter filter, bool topLevel) {
  if (!stmt) {
    return;
  }

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardStaticSlice in the current scope.
  if (!filter(stmt)) {
    return;
  }

  for (auto *operand : stmt->getOperands()) {
    auto *stmt = operand->getDefiningStmt();
    if (backwardStaticSlice->count(stmt) == 0) {
      getBackwardStaticSlice(stmt, backwardStaticSlice, filter,
                             /* topLevel */ false);
    }
  }

  // Don't insert the top level statement, we just queried on it and don't
  // want it in the results.
  if (!topLevel) {
    backwardStaticSlice->insert(stmt);
  }
}

SetVector<Statement *> mlir::getStaticSlice(Statement *stmt,
                                            TransitiveFilter backwardFilter,
                                            TransitiveFilter forwardFilter) {
  SetVector<Statement *> staticSlice;
  staticSlice.insert(stmt);

  int currentIndex = 0;
  SetVector<Statement *> backwardStaticSlice;
  SetVector<Statement *> forwardStaticSlice;
  while (currentIndex != staticSlice.size()) {
    auto *currentStmt = (staticSlice)[currentIndex];
    // Compute and insert the backwardStaticSlice starting from currentStmt.
    backwardStaticSlice.clear();
    getBackwardStaticSlice(currentStmt, &backwardStaticSlice, backwardFilter);
    staticSlice.insert(backwardStaticSlice.begin(), backwardStaticSlice.end());

    // Compute and insert the forwardStaticSlice starting from currentStmt.
    forwardStaticSlice.clear();
    getForwardStaticSlice(currentStmt, &forwardStaticSlice, forwardFilter);
    staticSlice.insert(forwardStaticSlice.begin(), forwardStaticSlice.end());
    ++currentIndex;
  }
  return topologicalSort(staticSlice);
}

namespace {
/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all statements but only record the ones that appear in `toSort`
/// for the final result.
struct DFSState {
  DFSState(const SetVector<Statement *> &set)
      : toSort(set), topologicalCounts(), seen() {}
  const SetVector<Statement *> &toSort;
  SmallVector<Statement *, 16> topologicalCounts;
  DenseSet<Statement *> seen;
};
} // namespace

static void DFSPostorder(Statement *current, DFSState *state) {
  auto *opStmt = cast<OperationStmt>(current);
  assert(opStmt->getNumResults() <= 1 && "NYI: multi-result");
  if (opStmt->getNumResults() > 0) {
    for (auto &u : opStmt->getResult(0)->getUses()) {
      auto *stmt = u.getOwner();
      DFSPostorder(stmt, state);
    }
  }
  bool inserted;
  using IterTy = decltype(state->seen.begin());
  IterTy iter;
  std::tie(iter, inserted) = state->seen.insert(current);
  if (inserted) {
    if (state->toSort.count(current) > 0) {
      state->topologicalCounts.push_back(current);
    }
  }
}

SetVector<Statement *>
mlir::topologicalSort(const SetVector<Statement *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    DFSPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Statement *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}
