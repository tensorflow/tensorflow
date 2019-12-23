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
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/SetVector.h"

///
/// Implements Analysis functions specific to slicing in Function.
///

using namespace mlir;

using llvm::SetVector;

static void getForwardSliceImpl(Operation *op,
                                SetVector<Operation *> *forwardSlice,
                                TransitiveFilter filter) {
  if (!op) {
    return;
  }

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(op)) {
    return;
  }

  if (auto forOp = dyn_cast<AffineForOp>(op)) {
    for (auto *ownerInst : forOp.getInductionVar()->getUsers())
      if (forwardSlice->count(ownerInst) == 0)
        getForwardSliceImpl(ownerInst, forwardSlice, filter);
  } else if (auto forOp = dyn_cast<loop::ForOp>(op)) {
    for (auto *ownerInst : forOp.getInductionVar()->getUsers())
      if (forwardSlice->count(ownerInst) == 0)
        getForwardSliceImpl(ownerInst, forwardSlice, filter);
  } else {
    assert(op->getNumRegions() == 0 && "unexpected generic op with regions");
    assert(op->getNumResults() <= 1 && "unexpected multiple results");
    if (op->getNumResults() > 0) {
      for (auto *ownerInst : op->getResult(0)->getUsers())
        if (forwardSlice->count(ownerInst) == 0)
          getForwardSliceImpl(ownerInst, forwardSlice, filter);
    }
  }

  forwardSlice->insert(op);
}

void mlir::getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                           TransitiveFilter filter) {
  getForwardSliceImpl(op, forwardSlice, filter);
  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  forwardSlice->remove(op);

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  std::vector<Operation *> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

static void getBackwardSliceImpl(Operation *op,
                                 SetVector<Operation *> *backwardSlice,
                                 TransitiveFilter filter) {
  if (!op)
    return;

  assert((op->getNumRegions() == 0 || isa<AffineForOp>(op) ||
          isa<loop::ForOp>(op)) &&
         "unexpected generic op with regions");

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (!filter(op)) {
    return;
  }

  for (auto en : llvm::enumerate(op->getOperands())) {
    auto operand = en.value();
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      if (auto affIv = getForInductionVarOwner(operand)) {
        auto *affOp = affIv.getOperation();
        if (backwardSlice->count(affOp) == 0)
          getBackwardSliceImpl(affOp, backwardSlice, filter);
      } else if (auto loopIv = loop::getForInductionVarOwner(operand)) {
        auto *loopOp = loopIv.getOperation();
        if (backwardSlice->count(loopOp) == 0)
          getBackwardSliceImpl(loopOp, backwardSlice, filter);
      } else if (blockArg->getOwner() !=
                 &op->getParentOfType<FuncOp>().getBody().front()) {
        op->emitError("unsupported CF for operand ") << en.index();
        llvm_unreachable("Unsupported control flow");
      }
      continue;
    }
    auto *op = operand->getDefiningOp();
    if (backwardSlice->count(op) == 0) {
      getBackwardSliceImpl(op, backwardSlice, filter);
    }
  }

  backwardSlice->insert(op);
}

void mlir::getBackwardSlice(Operation *op,
                            SetVector<Operation *> *backwardSlice,
                            TransitiveFilter filter) {
  getBackwardSliceImpl(op, backwardSlice, filter);

  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  backwardSlice->remove(op);
}

SetVector<Operation *> mlir::getSlice(Operation *op,
                                      TransitiveFilter backwardFilter,
                                      TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentInst = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentInst.
    backwardSlice.clear();
    getBackwardSlice(currentInst, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentInst.
    forwardSlice.clear();
    getForwardSlice(currentInst, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return topologicalSort(slice);
}

namespace {
/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set)
      : toSort(set), topologicalCounts(), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;
};
} // namespace

static void DFSPostorder(Operation *current, DFSState *state) {
  assert(current->getNumResults() <= 1 && "NYI: multi-result");
  if (current->getNumResults() > 0) {
    for (auto &u : current->getResult(0)->getUses()) {
      auto *op = u.getOwner();
      DFSPostorder(op, state);
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

SetVector<Operation *>
mlir::topologicalSort(const SetVector<Operation *> &toSort) {
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
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}
