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
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace {

class CopyRemoval : bufferization::BufferPlacementTransformationBase {
 public:
  explicit CopyRemoval(Operation *op)
      : BufferPlacementTransformationBase(op),
        userange_(op, allocs, aliases),
        dominators_(op) {}

  void removeCopy() {
    // A vector with the allocation value / copy operation pairs s to process.
    llvm::SmallVector<CopyOpInterface> toProcess;
    fillProcessSetAndResolveAliases(toProcess);

    DenseMap<Value, UseInterval::Vector> updatedUserange;
    DenseMap<Value, UserangeAnalysis::UsePositionList> updatedUsepositions;

    // Lambda expression to update the userange interval.
    auto lambdaUserangeUpdate = [&](Value v,
                                    DenseMap<Value, UseInterval::Vector> &map)
        -> UseInterval::Vector & { return insertUserangeInterval(v, map); };

    // A set containing copy operations that can be erased.
    SmallPtrSet<Operation *, 16> toErase;
    while (!toProcess.empty()) {
      CopyOpInterface copyOp = toProcess.pop_back_val();

      // Cast the Operation and get the Source and Target.
      Value copySource = copyOp.getSource();
      Value copyTarget = copyOp.getTarget();

      // Only remove copies if they do not affect maps.
      if (copySource.getType().cast<MemRefType>().getLayout() !=
          copyTarget.getType().cast<MemRefType>().getLayout())
        continue;

      // Get the UserangeIntervals.
      auto sourceAlloc = alias_to_alloc_map_[copySource];
      UseInterval::Vector sourceInterval =
          getOrInsert(sourceAlloc, updatedUserange, lambdaUserangeUpdate);
      auto targetAlloc = alias_to_alloc_map_[copyTarget];
      UseInterval::Vector targetInterval =
          getOrInsert(targetAlloc, updatedUserange, lambdaUserangeUpdate);

      UseInterval::Vector intersect = sourceInterval;

      // Compute the intersection.
      UseInterval::intervalIntersect(intersect, targetInterval);

      // If the sourceInterval contains more than one UseInterval, there are
      // multiple operations that intersect. The sourceInterval must have at
      // least one UseInterval that contains the copyOp.
      if (intersect.size() != 1) continue;

      // Check if all operations inside the intersection are benign, part of the
      // copyOp or a dealloc.
      if (!usesInIntervalAreSafe(copyOp, copySource, *intersect.begin()))
        continue;

      // Check if the currentOp dominates all uses of the copyTarget.
      if (!checkDominance(copyOp, copyTarget.getUsers(), toErase)) continue;

      // The last op in the intersection of the use ranges needs to be a
      // dealloc, as it ended the original source range. If we do the reuse,
      // we have to remove that dealloc to extend the liferange of the original
      // value.
      auto *lastOp = userange_.getOperation(intersect.back().end);
      if (!isDeallocOperationFor(lastOp, copySource)) continue;
      toErase.insert(lastOp);

      // Merge the Useranges.
      UseInterval::intervalMerge(sourceInterval, targetInterval);

      // Replace all uses of the target with the source.
      copyTarget.replaceAllUsesWith(copySource);
      toErase.insert(copyOp);
    }
    // Erase the copy operations.
    for (auto *eraseOp : toErase) eraseOp->erase();

    // Erase all allocs without uses.
    for (const bufferization::BufferPlacementAllocs::AllocEntry &entry :
         allocs) {
      Value alloc = std::get<0>(entry);
      if (alloc.use_empty()) alloc.getDefiningOp()->erase();
    }
  }

 private:
  /// Iterate over all allocs and their aliases and add their uses to the
  /// process set that implement a CopyOpInterface, where the alloc or alias is
  /// the source of the CopyOpInterface.
  void fillProcessSetAndResolveAliases(
      llvm::SmallVectorImpl<CopyOpInterface> &toProcess) {
    // A Set that contains the already processed aliases.
    SmallPtrSet<Value, 16U> processedAliases;

    // Iterate over the allocs.
    for (const bufferization::BufferPlacementAllocs::AllocEntry &entry :
         allocs) {
      Value allocValue = std::get<0>(entry);

      // Resolve the aliases of the current alloc and iterate over them.
      // At the same time, merge the use ranges of aliases into the use range
      // of the corresponding allocation.
      const ValueSetT &aliasSet = aliases.resolve(allocValue);
      for (Value alias : aliasSet) {
        // If the alias is already processed, continue.
        if (!processedAliases.insert(alias).second) continue;
        // Union the use ranges.
        userange_.unionRanges(allocValue, alias);
        // Remember the alias.
        alias_to_alloc_map_.insert({alias, allocValue});
        // If any of the uses are a copy, we have a canidate.
        for (auto *user : alias.getUsers()) {
          auto copyOp = dyn_cast<CopyOpInterface>(user);
          if (!copyOp) continue;
          if (copyOp.getSource() != alias) continue;
          toProcess.push_back(copyOp);
        }
      }
    }
  }

  /// Find the given Value in the DenseMap and return the pointer. If the given
  /// Value is not in the Map, insert a copy of the given original to the
  /// DenseMap using the pased update function and return a pointer to that
  /// element.
  template <typename T, typename TFunc>
  T &getOrInsert(Value v, DenseMap<Value, T> &updateMap,
                 const TFunc &updateFunc) {
    auto iter = updateMap.find(v);
    if (iter != updateMap.end()) return iter->second;
    return updateFunc(v, updateMap);
  }

  /// Insert the original userange intervals of the operation in the map.
  UseInterval::Vector &insertUserangeInterval(
      Value v, DenseMap<Value, UseInterval::Vector> &updateMap) {
    const auto *original = userange_.getUserangeInterval(v).getValue();
    auto &entry = updateMap[v];
    entry = *original;
    return entry;
  }

  /// Check if all users in the given range are dominated by given operation.
  /// Note: The target has always at least one use which is the copy operation.
  bool checkDominance(Operation *operation, const Value::user_range &userRange,
                      SmallPtrSet<Operation *, 16> &ignoreSet) {
    // Check if any use of the target is not dominated by the useOp. Erased
    // operations are ignored as uses.
    return llvm::all_of(userRange, [=](Operation *user) {
      return ignoreSet.count(user) || dominators_.dominates(operation, user);
    });
  }

  /// Checks whether op is a dealloction operation for value.
  /// This helper is aware of aliasing via the alias_to_alloc_map_.
  bool isDeallocOperationFor(Operation *op, Value value) {
    auto effect = dyn_cast<MemoryEffectOpInterface>(op);
    Value originalAlloc = alias_to_alloc_map_[value];
    return effect && effect.hasEffect<MemoryEffects::Free>() &&
           llvm::any_of(op->getOperands(), [&](Value operand) {
             Value operandAlloc = alias_to_alloc_map_[operand];
             return operandAlloc == originalAlloc;
           });
  }

  /// Checks whether all uses within the given interval are safe, i.e., there
  /// are no conflicts.
  /// This currently means that the interval may only contain non-sideeffecting
  /// operations or a dealloc of the given source value.
  bool usesInIntervalAreSafe(Operation *op, Value source,
                             UseInterval &interval) {
    // Divide the start and end by two to remove read/write properties.
    for (int id = interval.start / 2, e = interval.end / 2; id <= e; ++id) {
      // Get the operation from the id. Multiply the id by 2, because the
      // userange operates on doubled ids. Return false if the operation is not
      // an ancestor.
      // TODO(herhut): This is a bit of a big hammer. Ideally this should only
      //               look at use positions. Refactor to use those here.
      Operation *op_in_interval = userange_.getOperation(id * 2);
      if (op->isAncestor(op_in_interval)) continue;
      auto effect = dyn_cast<MemoryEffectOpInterface>(op_in_interval);
      // If we do not know about effects, fail.
      if (!effect) return false;
      // If it has no effect we are safe. It is OK if it gets the operand as
      // it does not use it.
      if (effect.hasNoEffect()) continue;
      if (isDeallocOperationFor(op_in_interval, source)) continue;
      return false;
    }
    return true;
  }

  /// The current userange info.
  UserangeAnalysis userange_;

  /// A map from aliases to their allocation value.
  DenseMap<Value, Value> alias_to_alloc_map_;

  /// The current dominance info.
  DominanceInfo dominators_;
};

struct CopyRemovalPass : public CopyRemovalBase<CopyRemovalPass> {
  void runOnOperation() override {
    Operation *funcOp = getOperation();
    CopyRemoval removal(funcOp);
    removal.removeCopy();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCopyRemovalPass() {
  return std::make_unique<CopyRemovalPass>();
}

}  // namespace mlir
