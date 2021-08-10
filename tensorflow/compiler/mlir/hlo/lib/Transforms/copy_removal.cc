/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

namespace {

class CopyRemoval : BufferPlacementTransformationBase {
public:
  explicit CopyRemoval(Operation *op)
      : BufferPlacementTransformationBase(op), userange(op, allocs, aliases),
        dominators(op) {}

  void removeCopy(Operation *op) {
    DenseMap<Value, UseInterval::Vector> useIntervalMap;
    fillProcessSet(useIntervalMap);

    while (!toProcess.empty()) {
      // Get the first item in the toProcess set and pop it.
      Value currentValue = toProcess.front();
      toProcess.remove(currentValue);

      UseInterval::Vector &currentIntervals = useIntervalMap[currentValue];
      // The currentInterval is a reference of the UseInterval that is updated
      // throughout the copy removal.
      for (UseInterval &currentInterval : currentIntervals) {
        size_t lastUseId = currentInterval.right;
        Operation *currentLastUse = userange.getOperation(lastUseId);

        // Check if the currentLastUse is already in the remove list.
        if (toErase.find(currentLastUse) != toErase.end())
          continue;

        // Check if currentLastUse implements a CopyOpInterface.
        auto copyOpInterface = dyn_cast<CopyOpInterface>(currentLastUse);
        if (!copyOpInterface)
          continue;

        // Check if the source is the currentValue.
        if (copyOpInterface.getSource() != currentValue)
          continue;

        Value copyTarget = copyOpInterface.getTarget();
        UseInterval::Vector &targetIntervals = useIntervalMap[copyTarget];
        // Find the UseInterval of the target that contains the copy operation.
        UseInterval *targetInterval =
            llvm::find_if(targetIntervals, [=](const UseInterval &i) {
              return i.contains(lastUseId);
            });

        // Check if all target uses in the interval are domitated by the
        // current last use.
        if (!checkDominance(currentLastUse, lastUseId, copyTarget,
                            targetInterval))
          continue;

        // Replace all uses of the target with the source after the
        // currentLastUse in the current interval.
        copyTarget.replaceUsesWithIf(currentValue, [&](OpOperand &operand) {
          Operation *targetOp = operand.getOwner();
          size_t targetUseId = userange.computeId(copyTarget, targetOp);
          assert(dominators.dominates(currentLastUse->getBlock(),
                                      targetOp->getBlock()) &&
                 "Current last use does not dominate a target use!");
          return targetUseId > lastUseId &&
                 targetInterval->contains(targetUseId);
        });

        // Replacing the target with the source and removing the copy operation
        // changes the current UseInterval of source and target. The source
        // interval must be extended to the right border of the target interval
        // and the target interval must to be shortened to its last use.
        currentInterval.right = targetInterval->right;
        targetInterval->right =
            computeTargetUseInterval(targetInterval, copyTarget, lastUseId);

        // Add the copy operation to the erase list and insert source and target
        // to the process list again, because of the updated UseIntervals.
        toErase.insert(currentLastUse);
        toProcess.insert(currentValue);
        toProcess.insert(copyTarget);
      }
    }
    // Erase the copy operations.
    for (auto eraseOp : toErase)
      eraseOp->erase();

    // Erase all allocs without uses.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      if (alloc.use_empty())
        alloc.getDefiningOp()->erase();
    }
  }

private:
  /// Iterate over all allocs and their aliases and add them to the process set.
  /// Also, get the userange intervals if the alloc/alias has any uses.
  void fillProcessSet(DenseMap<Value, UseInterval::Vector> &useIntervalMap) {
    // Add all allocs and their aliases to the process set.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);

      ValueSetT aliasSet = aliases.resolve(allocValue);
      for (Value alias : aliasSet) {
        if (useIntervalMap.find(alias) != useIntervalMap.end())
          continue;

        // If the value has no uses/ empty userange, do not add it to the
        // process set.
        auto userangeInterval = userange.getUserangeInterval(alias);
        if (!userangeInterval)
          continue;

        toProcess.insert(alias);
        useIntervalMap[alias] = *userangeInterval.getValue();
      }
    }
  }

  /// Compute the new right border of the given UseInterval that is smaller than
  /// the lastUseId.
  size_t computeTargetUseInterval(const UseInterval *targetInterval,
                                  Value target, size_t lastUseId) {
    auto targetUsePostions = *userange.getUserangePositions(target);
    size_t targetLastUseId = targetInterval->right;
    // Iterate over all UsePositions and check if the use is inside the
    // targetInterval. If yes, check if the use is before the lastUse and update
    // the targetLastUseId, otherwise break and return targetLastUseId.
    // Note: The uses are sorted in ascending order.
    for (auto &targetUsePos : *targetUsePostions) {
      if (targetInterval->contains(targetUsePos.first)) {
        if (targetUsePos.first >= lastUseId)
          break;
        targetLastUseId = targetUsePos.first;
      }
    }
    return targetLastUseId;
  }

  /// Check if all uses after the useOpId are dominated by useOp in the
  /// UseInterval of target that contains useOp.
  /// Note: The target has always at least one use which is the copy operation.
  bool checkDominance(Operation *useOp, size_t useOpId, Value target,
                      const UseInterval *targetInterval) {
    Block *useBlock = useOp->getBlock();
    UserangeAnalysis::UsePositionList targetUsePosList =
        *userange.getUserangePositions(target).getValue();
    // Check if any use after the useOpId that is inside the UseInterval does
    // not dominate the target use. Erased operations are ignored as uses.
    return !llvm::any_of(targetUsePosList,
                         [=](const UserangeAnalysis::UsePosition targetUsePos) {
                           Operation *targetUse = targetUsePos.second;
                           return targetUsePos.first > useOpId &&
                                  targetUsePos.first <= targetInterval->right &&
                                  toErase.find(targetUse) == toErase.end() &&
                                  !dominators.dominates(useBlock,
                                                        targetUse->getBlock());
                         });
  }

  /// The current set with values to process.
  llvm::SetVector<Value> toProcess;

  /// A set containing copy operations that can be erased.
  SmallPtrSet<Operation *, 16> toErase;

  /// The current userange info.
  UserangeAnalysis userange;

  /// The current dominance info.
  DominanceInfo dominators;

}; // namespace

struct CopyRemovalPass : public CopyRemovalBase<CopyRemovalPass> {
  void runOnFunction() override {
    Operation *funcOp = getFunction();
    CopyRemoval removal(funcOp);
    removal.removeCopy(funcOp);
  }
};

} // namespace

std::unique_ptr<FunctionPass> createCopyRemovalPass() {
  return std::make_unique<CopyRemovalPass>();
}

} // namespace mlir
