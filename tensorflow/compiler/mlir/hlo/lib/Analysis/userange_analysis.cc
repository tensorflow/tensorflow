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

#include <algorithm>
#include <utility>

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
  using ValueSetT = BufferViewFlowAnalysis::ValueSetT;

 public:
  /// Constructs an Userange builder.
  UserangeInfoBuilder(Liveness pLiveness, ValueSetT pValues,
                      OperationListT pOpList)
      : values(std::move(pValues)),
        opList(std::move(pOpList)),
        liveness(std::move(pLiveness)) {}

  /// Computes the userange of the current value by iterating over all of its
  /// uses.
  Liveness::OperationListT computeUserange() {
    Region *topRegion = findTopRegion();
    // Iterate over all associated uses.
    for (Operation *use : opList) {
      // If one of the parents implements a LoopLikeOpInterface we need to add
      // all operations inside of its regions to the userange.
      Operation *loopParent = use->getParentOfType<LoopLikeOpInterface>();
      if (loopParent && topRegion->isProperAncestor(use->getParentRegion()))
        addAllOperationsInRegion(loopParent);

      // Check if the parent block has already been processed.
      Block *useBlock = findTopLiveBlock(use);
      if (!startBlocks.insert(useBlock).second || visited.contains(useBlock))
        continue;

      // Add all operations inside the block that are within the userange.
      findOperationsInUse(useBlock);
    }
    return currentUserange;
  }

 private:
  /// Find the top most Region of all values stored in the values set.
  Region *findTopRegion() const {
    Region *topRegion = nullptr;
    llvm::for_each(values, [&](Value v) {
      Region *other = v.getParentRegion();
      if (!topRegion || topRegion->isAncestor(other)) topRegion = other;
    });
    return topRegion;
  }

  /// Finds the highest level block that has the current value in its liveOut
  /// set.
  Block *findTopLiveBlock(Operation *op) const {
    Operation *topOp = op;
    while (const LivenessBlockInfo *blockInfo =
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
  void addAllOperationsBetween(Operation *start, Operation *end) {
    currentUserange.push_back(start);
    addAllOperationsInRegion(start);

    while (start != end) {
      start = start->getNextNode();
      addAllOperationsInRegion(start);
      currentUserange.push_back(start);
    }
  }

  /// Adds all operations that are uses of the value in the given block to the
  /// userange of the current value. Additionally iterate over all successors
  /// where the value is live.
  void findOperationsInUse(Block *block) {
    SmallVector<Block *, 8> blocksToProcess;
    addOperationsInBlockAndFindSuccessors(
        block, block, getStartOperation(block), blocksToProcess);
    while (!blocksToProcess.empty()) {
      Block *toProcess = blocksToProcess.pop_back_val();
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
      const Block *startBlock, Block *toProcess, Operation *start,
      SmallVector<Block *, 8> &blocksToProcess) {
    const LivenessBlockInfo *blockInfo = liveness.getLiveness(toProcess);
    Operation *end = getEndOperation(toProcess);

    addAllOperationsBetween(start, end);

    // If the value is live out we need to process all successors at which the
    // value is live in.
    if (!llvm::any_of(values, [&](Value v) { return blockInfo->isLiveOut(v); }))
      return;
    for (Block *successor : toProcess->getSuccessors()) {
      // If the successor is the startBlock, we found a loop and only have to
      // add the operations from the block front to the first use of the
      // value.
      if (!llvm::any_of(values, [&](Value v) {
            return liveness.getLiveness(successor)->isLiveIn(v);
          }))
        continue;
      if (successor == startBlock) {
        start = &successor->front();
        end = getStartOperation(successor);
        if (start != end) addAllOperationsBetween(start, end->getPrevNode());
        // Else we need to check if the value is live in and the successor
        // has not been visited before. If so we also need to process it.
      } else if (visited.insert(successor).second) {
        blocksToProcess.emplace_back(successor);
      }
    }
  }

  /// Iterates over all regions of a given operation and adds all operations
  /// inside those regions to the userange of the current value.
  void addAllOperationsInRegion(Operation *parentOp) {
    // Iterate over all regions of the parentOp.
    for (Region &region : parentOp->getRegions()) {
      // Iterate over blocks inside the region.
      for (Block &block : region) {
        // If the blocks have been used as a startBlock before, we need to add
        // all operations between the block front and the startOp of the value.
        if (startBlocks.contains(&block)) {
          Operation *start = &block.front();
          Operation *end = getStartOperation(&block);
          if (start != end) addAllOperationsBetween(start, end->getPrevNode());

          // If the block has never been seen before, we need to add all
          // operations inside.
        } else if (visited.insert(&block).second) {
          for (Operation &op : block) {
            addAllOperationsInRegion(&op);
            currentUserange.emplace_back(&op);
          }
          continue;
        }
        // If the block has either been visited before or was used as a
        // startBlock, we need to add all operations between the endOp of the
        // value and the end of the block.
        Operation *end = getEndOperation(&block);
        if (end == &block.back()) continue;
        addAllOperationsBetween(end->getNextNode(), &block.back());
      }
    }
  }

  /// Find the start operation of the current value inside the given block.
  Operation *getStartOperation(Block *block) {
    Operation *startOperation = &block->back();
    for (Operation *useOp : opList) {
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
  Operation *getEndOperation(Block *block) {
    const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
    if (llvm::any_of(values, [&](Value v) { return blockInfo->isLiveOut(v); }))
      return &block->back();

    Operation *endOperation = &block->front();
    for (Operation *useOp : opList) {
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
  SmallPtrSet<Block *, 32> visited;

  /// The set of blocks that the userange computation started from.
  SmallPtrSet<Block *, 8> startBlocks;

  /// The current liveness info.
  Liveness liveness;
};
}  // namespace

UserangeAnalysis::UserangeAnalysis(Operation *op,
                                   const BufferPlacementAllocs &allocs,
                                   const BufferViewFlowAnalysis &aliases)
    : liveness(op) {
  // Walk over all operations and map them to an ID.
  op->walk([&](Operation *operation) {
    gatherMemoryEffects(operation);
    operationIds.insert({operation, operationIds.size()});
  });

  // Compute the use range for every allocValue and its aliases. Merge them
  // and compute an interval. Add all computed intervals to the useIntervalMap.
  for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
    Value allocValue = std::get<0>(entry);
    OperationListT useList;
    for (auto &use : allocValue.getUses()) useList.emplace_back(use.getOwner());
    UserangeInfoBuilder builder(liveness, {allocValue}, useList);
    OperationListT liveOperations = builder.computeUserange();

    // Sort the operation list by ids.
    std::sort(liveOperations.begin(), liveOperations.end(),
              [&](Operation *left, Operation *right) {
                return operationIds[left] < operationIds[right];
              });

    IntervalVector allocInterval = computeInterval(allocValue, liveOperations);
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
          for (auto &use : alias.getUses())
            aliasOperations.emplace_back(use.getOwner());
          // Compute the use range for the alias and sort the operations
          // afterwards.
          UserangeInfoBuilder aliasBuilder(liveness, {alias}, aliasOperations);
          aliasOperations = aliasBuilder.computeUserange();
          std::sort(aliasOperations.begin(), aliasOperations.end(),
                    [&](Operation *left, Operation *right) {
                      return operationIds[left] < operationIds[right];
                    });
        } else {
          aliasOperations = liveness.resolveLiveness(alias);
        }

        aliasUseranges.insert({alias, aliasOperations});
        useIntervalMap.insert(
            {alias, computeInterval(alias, aliasUseranges[alias])});
      }
      allocInterval =
          std::get<0>(intervalMerge(allocInterval, useIntervalMap[alias]));
    }
    aliasCache.insert(std::make_pair(allocValue, aliasSet));

    // Map the current allocValue to the computed useInterval.
    useIntervalMap.insert(std::make_pair(allocValue, allocInterval));
  }
}

/// Checks if the use intervals of the given values interfere.
bool UserangeAnalysis::rangesInterfere(Value itemA, Value itemB) const {
  return intervalUnion(itemA, itemB);
}

/// Merges the userange of itemB into the userange of itemA.
/// Note: This assumes that there is no interference between the two
/// ranges.
void UserangeAnalysis::unionRanges(Value itemA, Value itemB) {
  IntervalVector unionInterval =
      std::get<0>(intervalMerge(useIntervalMap[itemA], useIntervalMap[itemB]));

  llvm::set_union(aliasCache[itemA], aliasCache[itemB]);
  for (Value alias : aliasCache[itemA])
    unionInterval =
        std::get<0>(intervalMerge(unionInterval, useIntervalMap[alias]));

  // Compute new interval.
  useIntervalMap[itemA] = unionInterval;
}

/// Builds an IntervalVector corresponding to the given OperationList.
UserangeAnalysis::IntervalVector UserangeAnalysis::computeInterval(
    Value value, const Liveness::OperationListT &operationList) {
  assert(!operationList.empty() && "Operation list must not be empty");
  size_t start = computeID(value, *operationList.begin());
  size_t last = start;
  UserangeAnalysis::IntervalVector intervals;
  // Iterate over all operations in the operationList. If the gap between the
  // respective operationIds is greater 1 create a new interval.
  for (auto opIter = ++operationList.begin(), e = operationList.end();
       opIter != e; ++opIter) {
    size_t current = computeID(value, *opIter);
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
void UserangeAnalysis::gatherMemoryEffects(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallPtrSet<Value, 2> readEffectSet;
      SmallPtrSet<Value, 2> writeEffectSet;
      for (auto operand : op->getOperands()) {
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effectInterface.getEffectsOnValue(operand, effects);
        for (auto effect : effects) {
          if (isa<MemoryEffects::Write>(effect.getEffect()))
            writeEffectSet.insert(operand);
          else if (isa<MemoryEffects::Read>(effect.getEffect()))
            readEffectSet.insert(operand);
        }
      }
      opReadWriteMap.insert(
          {op, std::make_pair(readEffectSet, writeEffectSet)});
    }
  }
}

/// Computes the ID for the operation. If the operation contains operands which
/// have read effects, the returning ID will be odd. This allows us to
/// perform a replace in place.
size_t UserangeAnalysis::computeID(Value v, Operation *op) const {
  size_t doubledID = operationIds.find(op)->second * 2;
  auto mapIter = opReadWriteMap.find(op);
  if (mapIter == opReadWriteMap.end()) return doubledID;
  auto reads = mapIter->second.first;
  auto writes = mapIter->second.second;
  if (reads.contains(v) && !writes.contains(v)) return doubledID - 1;
  return doubledID;
}

/// Merge two IntervalVectors into a new IntervalVector. Return a pair with the
/// resulting IntervalVector and a boolean if there were interferences during
/// merging.
std::pair<UserangeAnalysis::IntervalVector, bool>
UserangeAnalysis::intervalMerge(const IntervalVector &intervalA,
                                const IntervalVector &intervalB) const {
  IntervalVector mergeResult;

  bool interference = false;
  auto iterA = intervalA.begin();
  auto iterB = intervalB.begin();
  auto endA = intervalA.end();
  auto endB = intervalB.end();
  UseInterval current;
  while (iterA != endA || iterB != endB) {
    if (iterA == endA) {
      // Only intervals from intervalB are left.
      current = *iterB;
      ++iterB;
    } else if (iterB == endB) {
      // Only intervals from intervalA are left.
      current = *iterA;
      ++iterA;
    } else if (iterA->second < iterB->first) {
      // A is strict before B: A(0,2), B(4,6)
      current = *iterA;
      ++iterA;
    } else if (iterB->second < iterA->first) {
      // B is strict before A: A(6,8), B(2,4)
      current = *iterB;
      ++iterB;
    } else {
      // A and B interfere.
      interference = true;
      current = UseInterval(std::min(iterA->first, iterB->first),
                            std::max(iterA->second, iterB->second));
      ++iterA;
      ++iterB;
    }
    // Merge current with last element in mergeResult, if the intervals are
    // consecutive and there is no gap.
    if (mergeResult.empty()) {
      mergeResult.emplace_back(current);
      continue;
    }
    UseInterval *mergeResultLast = (mergeResult.end() - 1);
    int diff = current.first - mergeResultLast->second;
    if (diff <= 2 && mergeResultLast->second < current.second)
      mergeResultLast->second = current.second;
    else if (diff > 2)
      mergeResult.emplace_back(current);
  }

  return std::make_pair(mergeResult, interference);
}

/// Performs an interval union of the interval vectors from the given values.
/// Returns an empty Optional if there is an interval interference.
bool UserangeAnalysis::intervalUnion(Value itemA, Value itemB) const {
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

  return std::get<1>(
      intervalMerge(tmpIntervalA, useIntervalMap.find(itemB)->second));
}

/// Performs an interval subtraction => A = A - B.
/// Note: This assumes that all intervals of b are included in some interval
///       of a.
void UserangeAnalysis::intervalSubtract(IntervalVector &a,
                                        const IntervalVector &b) const {
  auto iterB = b.begin();
  auto endB = b.end();
  for (auto iterA = a.begin(), endA = a.end();
       iterA != endA && iterB != endB;) {
    // iterA is strictly before iterB => increment iterA.
    if (iterA->second < iterB->first) {
      ++iterA;
    } else if (iterA->first == iterB->first && iterA->second > iterB->second) {
      // Usually, we would expect the case of iterB beeing strictly before
      // iterA. However, due to the initial assumption that all intervals of b
      // are included in some interval of a, we do not need to check if iterB is
      // strictly before iterA.
      // iterB is at the start of iterA, but iterA has some values that go
      // beyond those of iterB. We have to set the lower bound of iterA to the
      // upper bound of iterB + 1 and increment iterB.
      // A(3, 100) - B(3, 5) => A(6,100)
      iterA->first = iterB->second + 1;
      ++iterB;
    } else if (iterA->second == iterB->second && iterA->first < iterB->first) {
      // iterB is at the end of iterA, but iterA has some values that come
      // before iterB. We have to set the end of iterA to the start of iterB - 1
      // and increment both iterators.
      // A(4, 50) - B(40, 50) => A(4, 39)
      iterA->second = iterB->first - 1;
      ++iterA;
      ++iterB;
    } else if (iterA->first < iterB->first && iterA->second > iterB->second) {
      // iterB is in the middle of iterA. We have to split iterA and increment
      // iterB.
      // A(2, 10) - B(5, 7) => (2, 4), (8, 10)
      size_t endA = iterA->second;
      iterA->second = iterB->first - 1;
      iterA = a.insert(iterA, UseInterval(iterB->second + 1, endA));
      ++iterB;
    } else {
      // Both intervals are equal. We have to erase the whole interval.
      // A(5, 5) - B(5, 5) => {}
      iterA = a.erase(iterA);
      ++iterB;
    }
  }
}

void UserangeAnalysis::dump(raw_ostream &os) {
  os << "// ---- UserangeAnalysis -----\n";
  std::vector<Value> values;
  for (auto const &item : useIntervalMap) {
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
