//===- DCE.cpp - Dead Code Elimination ------------------------------------===//
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
// This transformation pass performs a simple dead code elimination algorithm.
//
// The overall goal of the pass is to prove that Values are dead, which allows
// deleting ops and block arguments.
//
// This pass uses an optimistic algorithm that assumes everything is dead until
// proved otherwise, allowing it to delete recursively dead cycles.
//
// This is a simple fixed-point dataflow analysis algorithm on a lattice
// {Dead,Alive}. Because liveness flows backward, we generally try to
// iterate everything backward to speed up convergence to the fixed-point.
//
// This pass's key feature compared to the existing peephole dead op folding
// that happens during canonicalization is being able to delete recursively
// dead cycles of the use-def graph, including block arguments.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
using namespace mlir;

namespace {
/// Simple dead code elimination.
struct DCE : public OperationPass<DCE> {
  void runOnOperation() override;
};
} // namespace

namespace {
/// Data structure used to track which values have already been proved live.
///
/// Because Operation's can have multiple results, this data structure tracks
/// liveness for both Value's and Operation's to avoid having to look through
/// all Operation results when analyzing a use.
///
/// This data structure essentially tracks the dataflow lattice.
/// The set of values/ops proved live increases monotonically to a fixed-point.
class LiveMap {
public:
  /// Value methods.
  bool wasProvenLive(Value *value) { return liveValues.count(value); }
  void setProvedLive(Value *value) {
    changed |= liveValues.insert(value).second;
  }

  /// Operation methods.
  bool wasProvenLive(Operation *op) { return liveOps.count(op); }
  void setProvedLive(Operation *op) { changed |= liveOps.insert(op).second; }

  /// Methods for tracking if we have reached a fixed-point.
  void resetChanged() { changed = false; }
  bool hasChanged() { return changed; }

private:
  bool changed = false;
  DenseSet<Value *> liveValues;
  DenseSet<Operation *> liveOps;
};
} // namespace

static bool isUseSpeciallyKnownDead(OpOperand &use, LiveMap &liveMap) {
  Operation *owner = use.getOwner();
  unsigned operandIndex = use.getOperandNumber();
  // This pass generally treats all uses of an op as live if the op itself is
  // considered live. However, for successor operands to terminators we need a
  // finer-grained notion where we deduce liveness for operands individually.
  // The reason for this is easiest to think about in terms of a classical phi
  // node based SSA IR, where each successor operand is really an operand to a
  // *separate* phi node, rather than all operands to the branch itself as with
  // the block argument representation that MLIR uses.
  //
  // And similarly, because each successor operand is really an operand to a phi
  // node, rather than to the terminator op itself, a terminator op can't e.g.
  // "print" the value of a successor operand.
  if (owner->isKnownTerminator()) {
    if (auto arg = owner->getSuccessorBlockArgument(operandIndex))
      return !liveMap.wasProvenLive(*arg);
    return false;
  }
  return false;
}

static void processValue(Value *value, LiveMap &liveMap) {
  bool provedLive = llvm::any_of(value->getUses(), [&](OpOperand &use) {
    if (isUseSpeciallyKnownDead(use, liveMap))
      return false;
    return liveMap.wasProvenLive(use.getOwner());
  });
  if (provedLive)
    liveMap.setProvedLive(value);
}

static bool isOpIntrinsicallyLive(Operation *op) {
  // This pass doesn't modify the CFG, so terminators are never deleted.
  if (!op->isKnownNonTerminator())
    return true;
  // If the op has a side effect, we treat it as live.
  if (!op->hasNoSideEffect())
    return true;
  return false;
}

static void propagateLiveness(Operation *op, LiveMap &liveMap) {
  // All Value's are either a block argument or an op result.
  // We call processValue on those cases.

  // Recurse on any regions the op has.
  for (Region &region : op->getRegions()) {
    for (Block *block : llvm::post_order(&region.front())) {
      // We process block arguments after the ops in the block, to promote
      // faster convergence to a fixed point (we try to visit uses before defs).
      for (Operation &op : llvm::reverse(block->getOperations()))
        propagateLiveness(&op, liveMap);
      for (Value *value : block->getArguments())
        processValue(value, liveMap);
    }
  }

  // Process the op itself.
  if (isOpIntrinsicallyLive(op)) {
    liveMap.setProvedLive(op);
    return;
  }
  for (Value *value : op->getResults())
    processValue(value, liveMap);
  bool provedLive = llvm::any_of(op->getResults(), [&](Value *value) {
    return liveMap.wasProvenLive(value);
  });
  if (provedLive)
    liveMap.setProvedLive(op);
}

static void eraseTerminatorSuccessorOperands(Operation *terminator,
                                             LiveMap &liveMap) {
  for (unsigned succI = 0, succE = terminator->getNumSuccessors();
       succI < succE; succI++) {
    // Iterating successors in reverse is not strictly needed, since we
    // aren't erasing any successors. But it is slightly more efficient
    // since it will promote later operands of the terminator being erased
    // first, reducing the quadratic-ness.
    unsigned succ = succE - succI - 1;
    for (unsigned argI = 0, argE = terminator->getNumSuccessorOperands(succ);
         argI < argE; argI++) {
      // Iterating args in reverse is needed for correctness, to avoid
      // shifting later args when earlier args are erased.
      unsigned arg = argE - argI - 1;
      Value *value = terminator->getSuccessor(succ)->getArgument(arg);
      if (!liveMap.wasProvenLive(value)) {
        terminator->eraseSuccessorOperand(succ, arg);
      }
    }
  }
}

static void deleteDeadness(MutableArrayRef<Region> regions, LiveMap &liveMap) {
  for (Region &region : regions) {
    // We do the deletion in an order that deletes all uses before deleting
    // defs.
    // MLIR's SSA structural invariants guarantee that except for block
    // arguments, the use-def graph is acyclic, so this is possible with a
    // single walk of ops and then a final pass to clean up block arguments.
    //
    // To do this, we visit ops in an order that visits domtree children
    // before domtree parents. A CFG post-order (with reverse iteration with a
    // block) satisfies that without needing an explicit domtree calculation.
    for (Block *block : llvm::post_order(&region.front())) {
      eraseTerminatorSuccessorOperands(block->getTerminator(), liveMap);
      for (Operation &childOp :
           llvm::make_early_inc_range(llvm::reverse(block->getOperations()))) {
        deleteDeadness(childOp.getRegions(), liveMap);
        if (!liveMap.wasProvenLive(&childOp))
          childOp.erase();
      }
    }
    // Delete block arguments.
    // The entry block has an unknown contract with their enclosing block, so
    // skip it.
    for (Block &block : llvm::drop_begin(region.getBlocks(), 1)) {
      // Iterate in reverse to avoid shifting later arguments when deleting
      // earlier arguments.
      for (unsigned i = 0, e = block.getNumArguments(); i < e; i++)
        if (!liveMap.wasProvenLive(block.getArgument(e - i - 1)))
          block.eraseArgument(e - i - 1, /*updatePredTerms=*/false);
    }
  }
}

void DCE::runOnOperation() {
  LiveMap liveMap;
  do {
    liveMap.resetChanged();
    propagateLiveness(getOperation(), liveMap);
  } while (liveMap.hasChanged());

  deleteDeadness(getOperation()->getRegions(), liveMap);
}

std::unique_ptr<Pass> mlir::createDCEPass() { return std::make_unique<DCE>(); }

static PassRegistration<DCE> pass("dce", "Dead code elimination");
