//===- UseDefAnalysis.cpp - Analysis for Transitive UseDef chains ---------===//
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
// This file implements Analysis functions specific to slicing in Function.
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
/// Implements Analysis functions specific to slicing in Function.
///

using namespace mlir;

using llvm::DenseSet;
using llvm::SetVector;

void mlir::getForwardSlice(Statement *stmt,
                           SetVector<Statement *> *forwardSlice,
                           TransitiveFilter filter, bool topLevel) {
  if (!stmt) {
    return;
  }

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(stmt)) {
    return;
  }

  if (auto *opStmt = dyn_cast<OperationInst>(stmt)) {
    assert(opStmt->getNumResults() <= 1 && "NYI: multiple results");
    if (opStmt->getNumResults() > 0) {
      for (auto &u : opStmt->getResult(0)->getUses()) {
        auto *ownerStmt = u.getOwner();
        if (forwardSlice->count(ownerStmt) == 0) {
          getForwardSlice(ownerStmt, forwardSlice, filter,
                          /*topLevel=*/false);
        }
      }
    }
  } else if (auto *forStmt = dyn_cast<ForStmt>(stmt)) {
    for (auto &u : forStmt->getUses()) {
      auto *ownerStmt = u.getOwner();
      if (forwardSlice->count(ownerStmt) == 0) {
        getForwardSlice(ownerStmt, forwardSlice, filter,
                        /*topLevel=*/false);
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
    std::vector<Statement *> v(forwardSlice->takeVector());
    forwardSlice->insert(v.rbegin(), v.rend());
  } else {
    forwardSlice->insert(stmt);
  }
}

void mlir::getBackwardSlice(Statement *stmt,
                            SetVector<Statement *> *backwardSlice,
                            TransitiveFilter filter, bool topLevel) {
  if (!stmt) {
    return;
  }

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(stmt)) {
    return;
  }

  for (auto *operand : stmt->getOperands()) {
    auto *stmt = operand->getDefiningInst();
    if (backwardSlice->count(stmt) == 0) {
      getBackwardSlice(stmt, backwardSlice, filter,
                       /*topLevel=*/false);
    }
  }

  // Don't insert the top level statement, we just queried on it and don't
  // want it in the results.
  if (!topLevel) {
    backwardSlice->insert(stmt);
  }
}

SetVector<Statement *> mlir::getSlice(Statement *stmt,
                                      TransitiveFilter backwardFilter,
                                      TransitiveFilter forwardFilter) {
  SetVector<Statement *> slice;
  slice.insert(stmt);

  unsigned currentIndex = 0;
  SetVector<Statement *> backwardSlice;
  SetVector<Statement *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentStmt = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentStmt.
    backwardSlice.clear();
    getBackwardSlice(currentStmt, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentStmt.
    forwardSlice.clear();
    getForwardSlice(currentStmt, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return topologicalSort(slice);
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
  auto *opStmt = cast<OperationInst>(current);
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
