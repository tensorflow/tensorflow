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
#include "llvm/ADT/SetOperations.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

namespace {
/// Builds a userange information from the given value and its liveness. The
/// information includes all operations that are within the userange.
struct UserangeInfoBuilder {
  using OperationListT = Liveness::OperationListT;
  using ValueSetT = BufferAliasAnalysis::ValueSetT;

 public:
  /// Constructs an Userange builder.
  UserangeInfoBuilder(Liveness liveness, ValueSetT values,
                      OperationListT opList)
      : values(values), opList(opList), liveness(liveness) {}

  /// Computes the userange of the current value by iterating over all of its
  /// uses.
  Liveness::OperationListT computeUserange() {
    // Iterate over all associated uses.
    for (Operation* use : opList) {
      // If one the parents implements a LoopLikeOpInterface we need to add all
      // operations inside of its regions to the userange.
      if (Operation* loopParent = use->getParentOfType<LoopLikeOpInterface>())
        addAllOperationsInRegion(loopParent);

      // Check if the parent block has already been processed.
      Block* useBlock = findTopLiveBlock(use);
      if (!startBlocks.insert(useBlock).second || visited.contains(useBlock))
        continue;

      // Add all operations inside the block that are within the userange.
      findOperationsInUse(useBlock);
    }
    return currentUserange;
  }

 private:
  /// Finds the highest level block that has the current value in its liveOut
  /// set.
  Block* findTopLiveBlock(Operation* op) const {
    Operation* topOp = op;
    while (const LivenessBlockInfo* blockInfo =
               liveness.getLiveness(op->getBlock())) {
      if (llvm::any_of(values,
                       [&](Value v) { return blockInfo->isLiveOut(v); }))
        topOp = op;
      op = op->getParentOp();
    }
    return topOp->getBlock();
  }

  /// Adds all operations from start to end to the userange of the current
  /// value. If an operation implements a nested region all operations inside of
  /// it are included as well. If includeEnd is false the end operation is not
  /// added.
  template <bool includeEnd = true>
  void addAllOperationsBetween(Operation* start, Operation* end) {
    if (includeEnd) {
      currentUserange.push_back(start);
      addAllOperationsInRegion(start);
    }

    while (start != end) {
      if (includeEnd) start = start->getNextNode();
      addAllOperationsInRegion(start);
      currentUserange.push_back(start);
      if (!includeEnd) start = start->getNextNode();
    }
  }

  /// Adds all operations that are in use in the given block to the userange of
  /// the current value. Additionally iterate over all successors where the
  /// value is live.
  void findOperationsInUse(Block* block) {
    SmallVector<Block*, 8> blocksToProcess;
    addOperationsInBlockAndFindSuccessors(
        block, block, getStartOperation(block), blocksToProcess);
    while (!blocksToProcess.empty()) {
      Block* toProcess = blocksToProcess.pop_back_val();
      addOperationsInBlockAndFindSuccessors(
          block, toProcess, &toProcess->front(), blocksToProcess);
    }
  }

  /// Adds the operations between the given start operation and the computed end
  /// operation to the userange. If the current value is live out, add all
  /// successor blocks that have the value live in to the process queue. If we
  /// find a loop, add the operations before the first use in block to the
  /// userange (if any). The startBlock is the block where the iteration over
  /// all successors started and is propagated further to find potential loops.
  void addOperationsInBlockAndFindSuccessors(
      const Block* startBlock, Block* toProcess, Operation* start,
      SmallVector<Block*, 8>& blocksToProcess) {
    const LivenessBlockInfo* blockInfo = liveness.getLiveness(toProcess);
    Operation* end = getEndOperation(toProcess);

    addAllOperationsBetween(start, end);

    // If the value is live out we need to process all successors at which the
    // value is live in.
    if (!llvm::any_of(values, [&](Value v) { return blockInfo->isLiveOut(v); }))
      return;
    for (Block* successor : toProcess->getSuccessors()) {
      // If the successor is the startBlock, we found a loop and only have to
      // add the operations from the block front to the first use of the
      // value.
      if (successor == startBlock) {
        start = &successor->front();
        end = getStartOperation(successor);
        addAllOperationsBetween<false>(start, end);
        // Else we need to check if the value is live in and the successor
        // has not been visited before. If so we also need to process it.
      } else if (llvm::any_of(
                     values,
                     [&](Value v) {
                       return liveness.getLiveness(successor)->isLiveIn(v);
                     }) &&
                 visited.insert(successor).second)
        blocksToProcess.emplace_back(successor);
    }
  }

  /// Iterates over all regions of a given operation and adds all operations
  /// inside those regions to the userange of the current value.
  void addAllOperationsInRegion(Operation* parentOp) {
    // Iterate over all regions of the parentOp.
    for (Region& region : parentOp->getRegions()) {
      // Iterate over blocks inside the region.
      for (Block& block : region) {
        // If the blocks have been used as a startBlock before, we need to add
        // all operations between the block front and the startOp of the value.
        if (startBlocks.contains(&block)) {
          Operation* start = &block.front();
          Operation* end = getStartOperation(&block);
          addAllOperationsBetween<false>(start, end);
          // If the block has never been seen before, we need to add all
          // operations inside.
        } else if (visited.insert(&block).second) {
          for (Operation& op : block) {
            addAllOperationsInRegion(&op);
            currentUserange.emplace_back(&op);
          }
          continue;
        }
        // If the block has either been visited before or was used as a
        // startBlock, we need to add all operations between the endOp of the
        // value and the end of the block.
        Operation* end = getEndOperation(&block);
        if (end == &block.back()) continue;
        addAllOperationsBetween(end->getNextNode(), &block.back());
      }
    }
  }

  /// Find the start operation of the current value inside the given block.
  Operation* getStartOperation(Block* block) {
    Operation* startOperation = &block->back();
    for (Operation* useOp : opList) {
      // Find the associated operation in the current block (if any).
      useOp = block->findAncestorOpInBlock(*useOp);
      // Check whether the use is in our block and after the current end
      // operation.
      if (useOp && useOp->isBeforeInBlock(startOperation))
        startOperation = useOp;
    }
    return startOperation;
  }

  /// Find the end operation of the current value inside the given block.
  Operation* getEndOperation(Block* block) {
    const LivenessBlockInfo* blockInfo = liveness.getLiveness(block);
    if (llvm::any_of(values, [&](Value v) { return blockInfo->isLiveOut(v); }))
      return &block->back();

    Operation* endOperation = &block->front();
    for (Operation* useOp : opList) {
      // Find the associated operation in the current block (if any).
      useOp = block->findAncestorOpInBlock(*useOp);
      // Check whether the use is in our block and after the current end
      // operation.
      if (useOp && endOperation->isBeforeInBlock(useOp)) endOperation = useOp;
    }
    return endOperation;
  }

  /// The current Value.
  ValueSetT values;

  /// The list of all operations used by the values.
  OperationListT opList;

  /// The result list of the userange computation.
  OperationListT currentUserange;

  /// The set of visited blocks during the userange computation.
  SmallPtrSet<Block*, 32> visited;

  /// The set of blocks that the userange computation started from.
  SmallPtrSet<Block*, 8> startBlocks;

  /// The current liveness info.
  Liveness liveness;
};
}  // namespace

UserangeAnalysis::UserangeAnalysis(Operation* op,
                                   const BufferPlacementAllocs& allocs,
                                   const BufferAliasAnalysis& aliases)
    : liveness(op) {
  // Walk over all operations and map them to an ID.
  DenseMap<Operation*, SmallPtrSet<Value, 2>> opToReadMap;
  op->walk([&](Operation* operation) {
    getMemoryEffects(operation, opToReadMap);
    operationIds.insert({operation, operationIds.size()});
  });

  // Compute the use range for every allocValue and its aliases. Merge them
  // and compute an interval. Add all computed intervals to the useIntervalMap.
  for (const BufferPlacementAllocs::AllocEntry& entry : allocs) {
    Value allocValue = std::get<0>(entry);
    OperationListT useList;
    for (auto& use : allocValue.getUses()) useList.emplace_back(use.getOwner());
    useMap.insert(std::make_pair(allocValue, useList));
    UserangeInfoBuilder builder(liveness, {allocValue}, useList);
    OperationListT liveOperations = builder.computeUserange();

    // Sort the operation list by ids.
    std::sort(liveOperations.begin(), liveOperations.end(),
              [&](Operation* left, Operation* right) {
                return operationIds[left] < operationIds[right];
              });

    // Iterate over all aliases and add their useranges to the userange of the
    // current value. Also add the useInterval of each alias to the
    // useIntervalMap.
    ValueSetT aliasSet = aliases.resolve(allocValue);
    for (Value alias : aliasSet) {
      if (alias == allocValue) continue;
      if (!aliasUseranges.count(alias)) {
        OperationListT aliasOperations;
        // If the alias is a BlockArgument then the value is live with the first
        // operation inside that block. Otherwise the liveness analysis is
        // sufficient for the use range.
        if (alias.isa<BlockArgument>()) {
          aliasOperations.emplace_back(&alias.getParentBlock()->front());
          for (auto& use : alias.getUses())
            aliasOperations.emplace_back(use.getOwner());
          // Compute the use range for the alias and sort the operations
          // afterwards.
          UserangeInfoBuilder aliasBuilder(liveness, {alias}, aliasOperations);
          aliasOperations = aliasBuilder.computeUserange();
          std::sort(aliasOperations.begin(), aliasOperations.end(),
                    [&](Operation* left, Operation* right) {
                      return operationIds[left] < operationIds[right];
                    });
        } else
          aliasOperations = liveness.resolveLiveness(alias);
        aliasUseranges.insert({alias, aliasOperations});
        useIntervalMap.insert(
            {alias,
             computeInterval(alias, aliasUseranges[alias], opToReadMap)});
      }
      liveOperations = mergeUseranges(liveOperations, aliasUseranges[alias]);
    }
    aliasCache.insert(std::make_pair(allocValue, aliasSet));

    // Map the current allocValue to the computed useInterval.
    useIntervalMap.insert(std::make_pair(
        allocValue, computeInterval(allocValue, liveOperations, opToReadMap)));
  }
}

/// Checks if the use intervals of the given values interfere.
bool UserangeAnalysis::rangesInterfere(Value itemA, Value itemB) const {
  return intervalUnion(itemA, itemB).hasValue();
}

/// Merges the userange of itemB into the userange of itemA.
/// Note: This assumes that there is no interference between the two
/// ranges.
void UserangeAnalysis::unionRanges(Value itemA, Value itemB) {
  OperationListT unionList = useMap[itemA];
  for (Operation *op : useMap[itemB]) unionList.emplace_back(op);
  useMap[itemA] = unionList;

  replaceMap[itemA].insert(itemB);
  UserangeInfoBuilder builder(liveness, replaceMap[itemA], useMap[itemA]);
  OperationListT liveOperations = builder.computeUserange();

  llvm::set_union(aliasCache[itemA], aliasCache[itemB]);
  for (Value alias : aliasCache[itemA])
    liveOperations = mergeUseranges(liveOperations, aliasUseranges[alias]);

  DenseMap<Operation*, SmallPtrSet<Value, 2>> opToReadMap;
  for (Operation *op : liveOperations) getMemoryEffects(op, opToReadMap);

  // Sort the operation list by ids.
  std::sort(liveOperations.begin(), liveOperations.end(),
            [&](Operation* left, Operation* right) {
              return operationIds[left] < operationIds[right];
            });

  // Compute new interval.
  useIntervalMap[itemA] = computeInterval(itemA, liveOperations, opToReadMap);
}

/// Builds an IntervalVector corresponding to the given OperationList.
UserangeAnalysis::IntervalVector UserangeAnalysis::computeInterval(
    Value value, const Liveness::OperationListT& operationList,
    DenseMap<Operation*, SmallPtrSet<Value, 2>>& opToReadMap) {
  assert(!operationList.empty() && "Operation list must not be empty");
  size_t start = computeID(value, *operationList.begin(), opToReadMap);
  size_t last = start;
  UserangeAnalysis::IntervalVector intervals;
  // Iterate over all operations in the operationList. If the gap between the
  // respective operationIds is greater 1 create a new interval.
  for (auto opIter = ++operationList.begin(), e = operationList.end();
       opIter != e; ++opIter) {
    size_t current = computeID(value, *opIter, opToReadMap);
    if (current - last > 2) {
      intervals.emplace_back(UserangeAnalysis::UseInterval(start, last));
      start = current;
    }
    last = current;
  }
  intervals.emplace_back(UserangeAnalysis::UseInterval(start, last));
  return intervals;
}

/// Checks each operand inside the operation for its memory effects and
/// separates them into read and write. Operands with read effects are added to
/// the opToReadMap.
void UserangeAnalysis::getMemoryEffects(
    Operation* op, DenseMap<Operation*, SmallPtrSet<Value, 2>>& opToReadMap) {
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallPtrSet<Value, 2> readSet;
      SmallPtrSet<Value, 2> writeSet;
      for (auto operand : op->getOperands()) {
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effectInterface.getEffectsOnValue(operand, effects);
        for (auto effect : effects) {
          if (isa<MemoryEffects::Write>(effect.getEffect())) {
            writeSet.insert(operand);
            readSet.erase(operand);
          } else if (isa<MemoryEffects::Read>(effect.getEffect()) &&
                     !writeSet.contains(operand))
            readSet.insert(operand);
        }
      }
      if (!readSet.empty() && !writeSet.empty())
        opToReadMap.insert({op, readSet});
    }
  }
}

/// Computes the ID for the operation. If the operation contains operands which
/// have read effects, the returning ID will be odd. This allows us to
/// perform a replace in place.
size_t UserangeAnalysis::computeID(
    Value v, Operation* op,
    DenseMap<Operation*, SmallPtrSet<Value, 2>>& opToReadMap) {
  size_t current = operationIds[op] * 2;
  auto mapIter = opToReadMap.find(op);
  if (mapIter != opToReadMap.end() && mapIter->second.contains(v)) current--;
  return current;
}

/// Merge two sorted (by operationID) OperationLists and ignore double
/// entries. Return the new computed OperationList.
Liveness::OperationListT UserangeAnalysis::mergeUseranges(
    const Liveness::OperationListT& first,
    const Liveness::OperationListT& second) const {
  Liveness::OperationListT mergeResult;
  // Union the two OperationLists.
  std::set_union(first.begin(), first.end(), second.begin(), second.end(),
                 std::back_inserter(mergeResult),
                 [&](Operation* left, Operation* right) {
                   return operationIds.find(left)->second <
                          operationIds.find(right)->second;
                 });

  return mergeResult;
}

/// Performs an interval union of the interval vectors from the given values.
/// Returns an empty Optional if there is an interval interference.
llvm::Optional<UserangeAnalysis::IntervalVector>
UserangeAnalysis::intervalUnion(Value itemA, Value itemB) const {
  ValueSetT intersect = aliasCache.find(itemA)->second;
  llvm::set_intersect(intersect, aliasCache.find(itemB)->second);
  IntervalVector tmpIntervalA = useIntervalMap.find(itemA)->second;

  // If the two values share a common alias, then the alias does not count as
  // interference and should be removed.
  if (!intersect.empty()) {
    for (Value alias : intersect) {
      IntervalVector aliasInterval = useIntervalMap.find(alias)->second;
      intervalSubtract(tmpIntervalA, aliasInterval);
    }
  }

  IntervalVector intervalUnion;
  IntervalVector currentInterval = useIntervalMap.find(itemB)->second;

  auto iterA = tmpIntervalA.begin();
  auto iterB = currentInterval.begin();
  auto endA = tmpIntervalA.end();
  auto endB = currentInterval.end();
  // Iterate over both interval vectors simultaneously.
  while (iterA != endA && iterB != endB) {
    // iterA comes before iterB.
    if (iterA->first < iterB->first && iterA->second < iterB->first)
      intervalUnion.emplace_back(*iterA++);
    // iterB comes before iterA.
    else if (iterB->first < iterA->first && iterB->second < iterA->first)
      intervalUnion.emplace_back(*iterB++);
    // There is an interval interference. We thus have to return an empty
    // Optional.
    else
      return llvm::None;
  }
  // Push the remaining intervals.
  for (; iterA != endA; ++iterA) intervalUnion.emplace_back(*iterA);
  for (; iterB != endB; ++iterB) intervalUnion.emplace_back(*iterB);

  // Merge consecutive intervals that have no gap between each other.
  for (auto it = intervalUnion.begin(); it != intervalUnion.end() - 1;) {
    if (it->second >= (it + 1)->first - 2) {
      (it + 1)->first = it->first;
      it = intervalUnion.erase(it);
    } else
      ++it;
  }
  return llvm::Optional<UserangeAnalysis::IntervalVector>(intervalUnion);
}

/// Performs an interval subtraction => A = A - B.
/// Note: This assumes that all intervals of b are included in some interval
///       of a.
void UserangeAnalysis::intervalSubtract(IntervalVector& a,
                                        const IntervalVector& b) const {
  auto iterB = b.begin();
  auto endB = b.end();
  for (auto iterA = a.begin(), endA = a.end();
       iterA != endA && iterB != endB;) {
    // iterA is strictly before iterB => increment iterA.
    if (iterA->second < iterB->first) ++iterA;
    // Usually, we would expect the case of iterB beeing strictly before iterA.
    // However, due to the initial assumption that all intervals of b are
    // included in some interval of a, we do not need to check if iterB is
    // striclty before iterA.
    // iterB is at the start of iterA, but iterA has some values that go
    // beyond those of iterB. We have to set the lower bound of iterA to the
    // upper bound of iterB + 1 and increment iterB.
    // A(3, 100) - B(3, 5) => A(6,100)
    else if (iterA->first == iterB->first && iterA->first <= iterB->second &&
             iterA->second > iterB->second) {
      iterA->first = iterB->second + 1;
      ++iterB;
    }
    // iterB is at the end of iterA, but iterA has some values that come
    // before iterB. We have to set the upper bound of iterA to the lower
    // bound of iterB - 1 and increment both iterators.
    // A(4, 50) - B(40, 50) => A(4, 39)
    else if (iterA->second >= iterB->first && iterA->second == iterB->second &&
             iterA->first < iterB->first) {
      iterA->second = iterB->first - 1;
      ++iterA;
      ++iterB;
    }
    // iterB is in the middle of iterA. We have to split iterA and increment
    // iterB.
    // A(2, 10) B(5, 7) => (2, 4), (8, 10)
    else if (iterA->first < iterB->first && iterA->second > iterB->second) {
      iterA->first = iterB->second + 1;
      iterA = a.insert(iterA, UseInterval(iterA->first, iterB->first - 1));
      ++iterA;
      ++iterB;
    }
    // Both intervals are equal. We have to erase the whole interval.
    // A(5, 5) B(5, 5) => {}
    else {
      iterA = a.erase(iterA);
      ++iterB;
    }
  }
}

void UserangeAnalysis::print(raw_ostream& os) {
  os << "// ---- UserangeAnalysis -----\n";
  std::vector<Value> values;
  for (auto const& item : useIntervalMap) {
    values.emplace_back(item.first);
  }
  std::sort(values.begin(), values.end(), [&](Value left, Value right) {
    if (left.getDefiningOp()) {
      if (right.getDefiningOp())
        return operationIds[left.getDefiningOp()] <
               operationIds[right.getDefiningOp()];
      else
        return true;
    }
    if (right.getDefiningOp()) return false;
    return operationIds[&left.getParentBlock()->front()] <
           operationIds[&right.getParentBlock()->front()];
  });
  for (auto value : values) {
    os << "Value: " << value << (value.getDefiningOp() ? "\n" : "");
    auto rangeIt = useIntervalMap[value].begin();
    os << "Userange: {(" << rangeIt->first << ", " << rangeIt->second << ")";
    rangeIt++;
    for (auto e = useIntervalMap[value].end(); rangeIt != e; ++rangeIt) {
      os << ", (" << rangeIt->first << ", " << rangeIt->second << ")";
    }
    os << "}\n";
  }
  os << "// ---------------------------\n";
}
