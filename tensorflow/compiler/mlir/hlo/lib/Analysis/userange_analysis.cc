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
  UserangeInfoBuilder(Liveness liveness, ValueSetT values,
                      OperationListT op_list)
      : values(std::move(values)),
        op_list(std::move(op_list)),
        liveness(std::move(liveness)) {}

  /// Computes the userange of the current value by iterating over all of its
  /// uses.
  Liveness::OperationListT computeUserange() {
    Region *topRegion = findTopRegion();
    // Iterate over all associated uses.
    for (Operation *use : op_list) {
      // If one of the parents implements a LoopLikeOpInterface we need to add
      // all operations inside of its regions to the userange.
      Operation *loopParent = use->getParentOfType<LoopLikeOpInterface>();
      if (loopParent && topRegion->isProperAncestor(use->getParentRegion()))
        addAllOperationsInRegion(loopParent);

      // Check if the parent block has already been processed.
      Block *useBlock = findTopLiveBlock(use);
      if (!start_blocks.insert(useBlock).second || visited.contains(useBlock))
        continue;

      // Add all operations inside the block that are within the userange.
      findOperationsInUse(useBlock);
    }
    return current_userange;
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
    current_userange.push_back(start);
    addAllOperationsInRegion(start);

    while (start != end) {
      start = start->getNextNode();
      addAllOperationsInRegion(start);
      current_userange.push_back(start);
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
        blocksToProcess.push_back(successor);
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
        if (start_blocks.contains(&block)) {
          Operation *start = &block.front();
          Operation *end = getStartOperation(&block);
          if (start != end) addAllOperationsBetween(start, end->getPrevNode());

          // If the block has never been seen before, we need to add all
          // operations inside.
        } else if (visited.insert(&block).second) {
          for (Operation &op : block) {
            addAllOperationsInRegion(&op);
            current_userange.push_back(&op);
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
    for (Operation *useOp : op_list) {
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
    for (Operation *useOp : op_list) {
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
  OperationListT op_list;

  /// The result list of the userange computation.
  OperationListT current_userange;

  /// The set of visited blocks during the userange computation.
  SmallPtrSet<Block *, 32> visited;

  /// The set of blocks that the userange computation started from.
  SmallPtrSet<Block *, 8> start_blocks;

  /// The current liveness info.
  Liveness liveness;
};
}  // namespace

/// Empty UseInterval Constructor.
UseInterval::UseInterval()
    : start(std::numeric_limits<size_t>::max()),
      end(std::numeric_limits<size_t>::min()) {}

/// Performs an interval subtraction => A = A - B.
void UseInterval::intervalSubtract(UseInterval::Vector &a,
                                   const UseInterval::Vector &b) {
  const auto *iterB = b.begin();
  const auto *endB = b.end();
  for (auto *iterA = a.begin(); iterA != a.end() && iterB != endB;) {
    // iterA is strictly before iterB => increment iterA.
    if (*iterA < *iterB) {
      ++iterA;
      // iterB is strictly before iterA => increment iterB.
    } else if (*iterA > *iterB) {
      ++iterB;
      // iterB overlaps with the start of iterA, but iterA has some values that
      // go beyond those of iterB. We have to set the start of iterA to the end
      // of iterB + 1 and increment iterB. A(3, 100) - B(3, 5) => A(6,100)
    } else if (iterA->start >= iterB->start && iterA->end > iterB->end) {
      iterA->start = iterB->end + 1;
      ++iterB;
      // iterB overlaps with the end of iterA, but iterA has some values that
      // come before iterB. We have to set the end of iterA to the start of
      // iterB - 1 and increment iterA. A(4, 50) - B(40, 50) => A(4, 39)
    } else if (iterA->end <= iterB->end && iterA->start < iterB->start) {
      iterA->end = iterB->start - 1;
      ++iterA;
      // iterB is in the middle of iterA. We have to split iterA and increment
      // iterB.
      // A(2, 10) - B(5, 7) => (2, 4), (8, 10)
    } else if (iterA->start < iterB->start && iterA->end > iterB->end) {
      size_t endA = iterA->end;
      iterA->end = iterB->start - 1;
      iterA = a.insert(iterA, UseInterval(iterB->end + 1, endA));
      ++iterB;
      // Both intervals are equal. We have to erase the whole interval.
      // A(5, 5) - B(5, 5) => {}
    } else {
      iterA = a.erase(iterA);
      ++iterB;
    }
  }
}

/// Performs an interval intersection => A = A ^ B.
void UseInterval::intervalIntersect(UseInterval::Vector &a,
                                    const UseInterval::Vector &b) {
  const auto *iterB = b.begin();
  const auto *endB = b.end();
  for (auto *iterA = a.begin(); iterA != a.end();) {
    // iterB points to the end, therefore the remaining UseIntervals from A must
    // be erased or iterA is strictly before iterB => erase iterA.
    if (iterB == endB || *iterA < *iterB) {
      iterA = a.erase(iterA);
      // iterB is strictly before iterA => increment iterB.
    } else if (*iterA > *iterB) {
      ++iterB;
      // iterB overlaps with iterA => reduce the interval to the overlap and
      // insert the ending split-off to vector A again.
    } else {
      size_t currentEndA = iterA->end;
      iterA->start = std::max(iterA->start, iterB->start);
      iterA->end = std::min(currentEndA, iterB->end);
      if (currentEndA > iterB->end) {
        iterA = a.insert(std::next(iterA),
                         UseInterval(iterB->end + 1, currentEndA));
        ++iterB;
      } else {
        ++iterA;
      }
    }
  }
}

/// Performs an interval merge => A = A u B.
/// Note: All overlapping and contiguous UseIntervals are merged.
void UseInterval::intervalMerge(UseInterval::Vector &a,
                                const UseInterval::Vector &b) {
  const auto *iterB = b.begin();
  const auto *endB = b.end();
  // Iterate over UseInterval::Vector a and b.
  for (auto *iterA = a.begin(); iterA != a.end() && iterB != endB;) {
    // Let A be the UseInterval of iterA and B the UseInterval of iterB.
    // Check if A is before B.
    if (*iterA < *iterB) {
      // Check if A and B can be merged if they are contiguous. If the merge
      // result contains the next elements of A, we can erase them.
      if (iterA->isContiguous(*iterB)) {
        mergeAndEraseContiguousIntervals(a, iterA, *iterB);
        ++iterB;
      }
      ++iterA;
      // Check if B is before A.
    } else if (*iterA > *iterB) {
      // Check if A and B can be merged if they are contiguous, else add B
      // to the Vector of A.
      if (iterB->isContiguous(*iterA))
        iterA->mergeWith(*iterB);
      else
        iterA = a.insert(iterA, *iterB);
      ++iterB;
      // The UseIntervals interfere and must be merged.
    } else {
      mergeAndEraseContiguousIntervals(a, iterA, *iterB);
      ++iterB;
    }
  }
  // If there are remaining UseIntervals in b, add them to a.
  if (iterB != endB) a.insert(a.end(), iterB, endB);
}

/// Merge the UseIntervals and erase overlapping and contiguouse UseIntervals
/// of the UseInterval::Vector.
void UseInterval::mergeAndEraseContiguousIntervals(
    UseInterval::Vector &interval, UseInterval *iter,
    const UseInterval &toMerge) {
  // Return if the iter points to the end.
  if (iter == interval.end()) return;

  // Merge the UseIntervals.
  iter->mergeWith(toMerge);

  // Find the next UseInterval from iter that is not contiguous with the merged
  // iter.
  UseInterval *next = std::next(iter);
  while (next != interval.end() && iter->isContiguous(*next)) {
    if (iter->end < next->end) iter->end = next->end;
    ++next;
  }
  // Remove contiguous UseIntervals.
  if (std::next(iter) != next) iter = interval.erase(std::next(iter), next);
}

UserangeAnalysis::UserangeAnalysis(
    Operation *op, const bufferization::BufferPlacementAllocs &allocs,
    const BufferViewFlowAnalysis &aliases)
    : liveness(op) {
  // Walk over all operations and map them to an ID.
  op->walk([&](Operation *operation) {
    gatherMemoryEffects(operation);
    operationIds.insert({operation, operationIds.size()});
    operations.push_back(operation);
  });

  // Compute the use range for every allocValue and its aliases. Merge them
  // and compute an interval. Add all computed intervals to the useIntervalMap.
  for (const bufferization::BufferPlacementAllocs::AllocEntry &entry : allocs) {
    Value allocValue = std::get<0>(entry);
    const Value::use_range &allocUses = allocValue.getUses();
    size_t dist = std::distance(allocUses.begin(), allocUses.end());
    OperationListT useList;
    useList.reserve(dist);
    for (auto &use : allocUses) useList.push_back(use.getOwner());
    computeUsePositions(allocValue);

    UserangeInfoBuilder builder(liveness, {allocValue}, useList);
    OperationListT liveOperations = builder.computeUserange();

    // Sort the operation list by ids.
    std::sort(liveOperations.begin(), liveOperations.end(),
              [&](Operation *left, Operation *right) {
                return operationIds[left] < operationIds[right];
              });

    UseInterval::Vector allocInterval =
        computeInterval(allocValue, liveOperations);
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
          aliasOperations.push_back(&alias.getParentBlock()->front());
          for (auto &use : alias.getUses())
            aliasOperations.push_back(use.getOwner());
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
        computeUsePositions(alias);
      }
      UseInterval::intervalMerge(allocInterval, useIntervalMap[alias]);
      mergeUsePositions(usePositionMap[allocValue], usePositionMap[alias]);
    }
    aliasCache.insert(std::make_pair(allocValue, aliasSet));

    // Map the current allocValue to the computed useInterval.
    useIntervalMap.insert(std::make_pair(allocValue, allocInterval));
  }
}

/// Computes the doubled Id for the given value inside the operation based on
/// the program sequence. If the value has only read effects, the returning ID
/// will be even, otherwise odd.
size_t UserangeAnalysis::computeId(Value v, Operation *op) const {
  size_t doubledID = (operationIds.find(op)->second + 1) * 2 - 1;
  auto mapIter = opReadWriteMap.find(op);
  if (mapIter == opReadWriteMap.end()) return doubledID;
  auto reads = mapIter->second.first;
  auto writes = mapIter->second.second;
  if (reads.contains(v) && !writes.contains(v)) return doubledID - 1;
  return doubledID;
}

/// Computes the UsePositions of the given Value, sorts and inserts them into
/// the usePositionMap.
void UserangeAnalysis::computeUsePositions(Value v) {
  // Get the uses of v.
  const Value::use_range &uses = v.getUses();

  // Create a UsePositionList.
  UsePositionList usePosList;
  size_t dist = std::distance(uses.begin(), uses.end());
  usePosList.reserve(dist);

  // Add all ids and Operations to the UsePositionList.
  for (auto &use : uses) {
    Operation *useOwner = use.getOwner();
    usePosList.emplace_back(computeId(v, useOwner), useOwner);
  }

  // Sort the UsePositions by ascending Ids.
  std::sort(usePosList.begin(), usePosList.end(),
            [](const UsePosition &a, const UsePosition &b) {
              return a.first < b.first;
            });

  // Insert the UsePositionList into the usePositionMap.
  usePositionMap.insert(std::make_pair(v, usePosList));
}

/// Merges listB into listA, sorts the result and removes all duplicates.
void UserangeAnalysis::mergeUsePositions(UsePositionList &listA,
                                         const UsePositionList &listB) {
  // Insert listB into listA.
  listA.insert(listA.end(), listB.begin(), listB.end());

  // Sort the resulting listA.
  std::sort(listA.begin(), listA.end(),
            [](const UsePosition &a, const UsePosition &b) {
              return a.first < b.first;
            });

  // Remove duplicates.
  listA.erase(std::unique(listA.begin(), listA.end()), listA.end());
}

/// Checks if the use intervals of the given values interfere.
bool UserangeAnalysis::rangesInterfere(Value itemA, Value itemB) const {
  ValueSetT intersect = aliasCache.find(itemA)->second;
  llvm::set_intersect(intersect, aliasCache.find(itemB)->second);
  UseInterval::Vector tmpIntervalA = useIntervalMap.find(itemA)->second;
  const UseInterval::Vector &intervalsB = useIntervalMap.find(itemB)->second;

  // If the two values share a common alias, then the alias does not count as an
  // interference and should be removed.
  if (!intersect.empty()) {
    for (Value alias : intersect) {
      const UseInterval::Vector &aliasInterval =
          useIntervalMap.find(alias)->second;
      UseInterval::intervalSubtract(tmpIntervalA, aliasInterval);
    }
  }

  // Iterate over both UseInterval::Vector and check if they interfere.
  const auto *iterB = intervalsB.begin();
  const auto *endB = intervalsB.end();
  for (auto iterA = tmpIntervalA.begin(), endA = tmpIntervalA.end();
       iterA != endA && iterB != endB;) {
    if (*iterA < *iterB)
      ++iterA;
    else if (*iterA > *iterB)
      ++iterB;
    else
      return true;
  }
  return false;
}

/// Merges the userange of itemB into the userange of itemA.
void UserangeAnalysis::unionRanges(Value itemA, Value itemB) {
  UseInterval::intervalMerge(useIntervalMap[itemA], useIntervalMap[itemB]);
}

/// Builds an UseInterval::Vector corresponding to the given OperationList.
UseInterval::Vector UserangeAnalysis::computeInterval(
    Value value, const Liveness::OperationListT &operationList) {
  assert(!operationList.empty() && "Operation list must not be empty");
  size_t start = computeId(value, *operationList.begin());
  size_t last = start;
  UseInterval::Vector intervals;
  // Iterate over all operations in the operationList. If the gap between the
  // respective operationIds is greater 1 create a new interval.
  for (auto opIter = ++operationList.begin(), e = operationList.end();
       opIter != e; ++opIter) {
    size_t current = computeId(value, *opIter);
    if (current - last > 2) {
      intervals.emplace_back(start, last);
      start = current;
    }
    last = current;
  }
  intervals.emplace_back(start, last);
  return intervals;
}

/// Checks each operand within the operation for its memory effects and
/// separates them into read and write.
void UserangeAnalysis::gatherMemoryEffects(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallPtrSet<Value, 2> readEffectSet;
      SmallPtrSet<Value, 2> writeEffectSet;
      SmallVector<MemoryEffects::EffectInstance> effects;
      for (auto operand : op->getOperands()) {
        effects.clear();
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

/// Computes the doubled Id back to the OperationId.
size_t UserangeAnalysis::unwrapId(size_t id) const { return id / 2; }

void UserangeAnalysis::dump(raw_ostream &os) {
  os << "// ---- UserangeAnalysis -----\n";
  llvm::SmallVector<Value> values;
  values.reserve(useIntervalMap.size());
  for (auto const &item : useIntervalMap) {
    values.push_back(item.first);
  }
  std::sort(values.begin(), values.end(), [&](Value left, Value right) {
    if (left.getDefiningOp()) {
      if (right.getDefiningOp())
        return operationIds[left.getDefiningOp()] <
               operationIds[right.getDefiningOp()];
      return true;
    }
    if (right.getDefiningOp()) return false;
    return operationIds[&left.getParentBlock()->front()] <
           operationIds[&right.getParentBlock()->front()];
  });
  for (auto value : values) {
    os << "Value: " << value << (value.getDefiningOp() ? "\n" : "");
    auto *rangeIt = useIntervalMap[value].begin();
    os << "Userange: {(" << rangeIt->start << ", " << rangeIt->end << ")";
    rangeIt++;
    for (auto *e = useIntervalMap[value].end(); rangeIt != e; ++rangeIt) {
      os << ", (" << rangeIt->start << ", " << rangeIt->end << ")";
    }
    os << "}\n";
  }
  os << "// ---------------------------\n";
}
