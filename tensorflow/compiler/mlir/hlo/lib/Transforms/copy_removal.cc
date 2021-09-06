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
    // A set with the copy Operation to process.
    llvm::SetVector<Operation *> toProcess;
    fillProcessSet(toProcess);

    while (!toProcess.empty()) {
      Operation *currentOp = toProcess.pop_back_val();

      // Cast the Operation and get the Source and Target.
      auto copyOpInterface = dyn_cast<CopyOpInterface>(currentOp);
      Value copySource = copyOpInterface.getSource();
      Value copyTarget = copyOpInterface.getTarget();

      // Get the UserangeIntervals.
      UseInterval::Vector sourceInterval =
          *userange.getUserangeInterval(copySource).getValue();
      UseInterval::Vector targetInterval =
          *userange.getUserangeInterval(copyTarget).getValue();

      // Compute the intersection.
      UseInterval::intervalIntersect(sourceInterval, targetInterval);

      // If the sourceInterval contains more than one UseInterval, there are
      // multiple operations that intersect. The sourceInterval must have at
      // least one UseInterval that contains the copyOp.
      if (sourceInterval.size() > 1)
        continue;

      // Check if all Operations inside the intersection are part of the copyOp.
      if (!checkAncestor(currentOp, *sourceInterval.begin()))
        continue;

      // Check if the currentOp dominates all uses of the copyTarget.
      if (!checkDominance(currentOp, copyTarget))
        continue;

      // Replace all uses of the target with the source.
      copyTarget.replaceAllUsesWith(copySource);

      // Merge the Useranges.
      UseInterval::intervalMerge(sourceInterval, targetInterval);
      toErase.insert(currentOp);
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
  /// Iterate over all allocs and their aliases and add their uses to the
  /// process set that implement a CopyOpInterface, where the alloc or alias is
  /// the source of the CopyOpInterface.
  void fillProcessSet(llvm::SetVector<Operation *> &toProcess) {
    // A Set that contains the already processed aliases.
    SmallPtrSet<Value, 16U> processedAliases;

    // Iterate over the allocs.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);

      // Resolve the aliases of the current alloc and iterate over them.
      const ValueSetT &aliasSet = aliases.resolve(allocValue);
      for (Value alias : aliasSet) {
        // If the alias is already processed, continue.
        if (!processedAliases.insert(alias).second)
          continue;

        // If the value has no uses/ empty userange, continue.
        auto userangeInterval = userange.getUserangeInterval(alias);
        if (!userangeInterval)
          continue;

        // Iterate over the UseIntervals and check if the last Operation in the
        // UseInterval implements a CopyOpInterface.
        for (UseInterval interval : *userangeInterval.getValue()) {
          Operation *currentLastUse = userange.getOperation(interval.end);
          auto copyOpInterface = dyn_cast<CopyOpInterface>(currentLastUse);
          if (!copyOpInterface)
            continue;

          // Check if the source is the alias.
          if (copyOpInterface.getSource() != alias)
            continue;

          toProcess.insert(currentLastUse);
        }
      }
    }
  }

  /// Check if all uses of the target Value are dominated by given Operation.
  /// Note: The target has always at least one use which is the copy operation.
  bool checkDominance(Operation *useOp, Value target) {
    Block *useBlock = useOp->getBlock();
    const UserangeAnalysis::UsePositionList &targetUsePosList =
        *userange.getUserangePositions(target).getValue();
    // Check if any use of the target is not dominated by the useOp. Erased
    // operations are ignored as uses.
    return !llvm::any_of(targetUsePosList,
                         [=](const UserangeAnalysis::UsePosition targetUsePos) {
                           Operation *targetUse = targetUsePos.second;
                           return toErase.find(targetUse) == toErase.end() &&
                                  !dominators.dominates(useBlock,
                                                        targetUse->getBlock());
                         });
  }

  /// Check if the given Operation is an ancestor of the operations inside the
  /// UseInterval.
  bool checkAncestor(Operation *op, UseInterval &interval) {
    // Divide the start and end by two to remove read/write properties.
    for (int id = interval.start / 2, e = interval.end / 2; id <= e; ++id) {
      // Get the operation from the id. Multiply the id by 2, because the
      // userange operates on doubled ids. Return false if the operation is not
      // an ancestor.
      if (!op->isAncestor(userange.getOperation(id * 2)))
        return false;
    }
    return true;
  }

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
