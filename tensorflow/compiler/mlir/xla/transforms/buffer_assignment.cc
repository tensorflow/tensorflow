/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/transforms/buffer_assignment.h"
#include "absl/memory/memory.h"
#include "mlir/IR/Function.h"   // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"     // TF:llvm-project

namespace mlir {
namespace xla {
namespace detail {

//===----------------------------------------------------------------------===//
// BufferAssignmentAliasAnalysis
//===----------------------------------------------------------------------===//

/// Constructs a new alias analysis using the op provided.
BufferAssignmentAliasAnalysis::BufferAssignmentAliasAnalysis(Operation* op) {
  build(op->getRegions());
}

/// Finds all immediate and indirect aliases this value could potentially
/// have.
BufferAssignmentAliasAnalysis::ValueSetT BufferAssignmentAliasAnalysis::resolve(
    Value value) const {
  ValueSetT result;
  resolveRecursive(value, result);
  return result;
}

/// Recursively determines alias information for the given value. It stores
/// all newly found potential aliases in the given result set.
void BufferAssignmentAliasAnalysis::resolveRecursive(Value value,
                                                     ValueSetT& result) const {
  if (!result.insert(value).second) return;
  auto it = aliases.find(value);
  if (it == aliases.end()) return;
  for (auto alias : it->second) {
    resolveRecursive(alias, result);
  }
}

/// This function constructs a mapping from values to its immediate aliases. It
/// iterates over all blocks, gets their predecessors, determines the values
/// that will be passed to the corresponding block arguments and inserts them
/// into map.
void BufferAssignmentAliasAnalysis::build(MutableArrayRef<Region> regions) {
  for (Region& region : regions)
    for (Block& block : region) {
      // Iterate over all predecessor and get the mapped values to their
      // corresponding block arguments values.
      for (auto pred : block.getPredecessors()) {
        // Determine the current successor index of the current predecessor.
        unsigned successorIndex = 0;
        for (auto successor : llvm::enumerate(pred->getSuccessors())) {
          if (successor.value() == &block) {
            successorIndex = successor.index();
            break;
          }
        }
        // Get the terminator and the values that will be passed to our block.
        auto terminator = pred->getTerminator();
        auto successorOps = terminator->getSuccessorOperands(successorIndex);
        // Build the actual mapping of values to their immediate aliases.
        for (auto arg : block.getArguments()) {
          auto value = successorOps[arg.getArgNumber()];
          aliases[value].insert(arg);
        }
      }
    }
}
}  // namespace detail

//===----------------------------------------------------------------------===//
// BufferAssignmentPositions
//===----------------------------------------------------------------------===//

/// Creates a new positions tuple including alloc and dealloc positions.
BufferAssignmentPositions::BufferAssignmentPositions(Operation* allocPosition,
                                                     Operation* deallocPosition)
    : allocPosition(allocPosition), deallocPosition(deallocPosition) {}

//===----------------------------------------------------------------------===//
// BufferAssignment
//===----------------------------------------------------------------------===//

/// Finds a proper placement block to store alloc/dealloc node according to the
/// algorithm descirbed at the top of the file. It supports dominator and
/// post-dominator analyses via template arguments.
template <typename AliasesT, typename DominatorT>
static Block* findPlacementBlock(Value value, const AliasesT& aliases,
                                 const DominatorT& doms) {
  assert(!value.isa<BlockArgument>() & "Cannot place a block argument");
  // Start with the current block the value is defined in.
  Block* dom = value.getDefiningOp()->getBlock();
  // Iterate over all aliases and their uses to find a safe placement block
  // according to the given dominator information.
  for (auto alias : aliases) {
    for (auto user : alias.getUsers()) {
      // Move upwards in the dominator tree to find an appropriate
      // dominator block that takes the current use into account.
      dom = doms.findNearestCommonDominator(dom, user->getBlock());
    }
  }
  return dom;
}

/// Finds a proper alloc positions according to the algorithm described at the
/// top of the file.
template <typename AliasesT, typename DominatorT>
static Operation* getAllocPosition(Value value, const Liveness& liveness,
                                   const AliasesT& aliases,
                                   const DominatorT& dominators) {
  // Determine the actual block to place the alloc and get liveness information.
  auto placementBlock = findPlacementBlock(value, aliases, dominators);
  auto livenessInfo = liveness.getLiveness(placementBlock);

  // We have to ensure that the alloc will be before the first use of all
  // aliases of the given value. We first assume that there are no uses in the
  // placementBlock and that we can safely place the alloc before the terminator
  // at the end of the block.
  Operation* startOperation = placementBlock->getTerminator();
  // Iterate over all aliases and ensure that the startOperation will point to
  // the first operation of all potential aliases in the placementBlock.
  for (auto alias : aliases) {
    auto aliasStartOperation = livenessInfo->getStartOperation(alias);
    // Check whether the aliasStartOperation lies in the desired block and
    // whether it is before the current startOperation. If yes, this will be the
    // new startOperation.
    if (aliasStartOperation->getBlock() == placementBlock &&
        aliasStartOperation->isBeforeInBlock(startOperation))
      startOperation = aliasStartOperation;
  }
  // startOperation is the first operation before which we can safely store the
  // alloc taking all potential aliases into account.
  return startOperation;
}

/// Finds a proper dealloc positions according to the algorithm described at the
/// top of the file.
template <typename AliasesT, typename DominatorT>
static Operation* getDeallocPosition(Value value, const Liveness& liveness,
                                     const AliasesT& aliases,
                                     const DominatorT& postDominators) {
  // Determine the actual block to place the dealloc and get liveness
  // information.
  auto placementBlock = findPlacementBlock(value, aliases, postDominators);
  auto livenessInfo = liveness.getLiveness(placementBlock);

  // We have to ensure that the dealloc will be after the last use of all
  // aliases of the given value. We first assume that there are no uses in the
  // placementBlock and that we can safely place the dealloc at the beginning.
  Operation* endOperation = &placementBlock->front();
  // Iterate over all aliases and ensure that the endOperation will point to the
  // last operation of all potential aliases in the placementBlock.
  for (auto alias : aliases) {
    auto aliasEndOperation = livenessInfo->getEndOperation(alias, endOperation);
    // Check whether the aliasEndOperation lies in the desired block and whether
    // it is behind the current endOperation. If yes, this will be the new
    // endOperation.
    if (aliasEndOperation->getBlock() == placementBlock &&
        endOperation->isBeforeInBlock(aliasEndOperation))
      endOperation = aliasEndOperation;
  }
  // endOperation is the last operation behind which we can safely store the
  // dealloc taking all potential aliases into account.
  return endOperation;
}

/// Creates a new BufferAssignment analysis.
BufferAssignment::BufferAssignment(Operation* op)
    : operation(op),
      liveness(op),
      dominators(op),
      postDominators(op),
      aliases(op) {}

/// Computes the actual positions to place allocs and deallocs for the given
/// value.
BufferAssignmentPositions BufferAssignment::computeAllocAndDeallocPositions(
    Value value) const {
  // Check for an artifical case that a dead value is passed to this function
  if (value.use_empty())
    return BufferAssignmentPositions(value.getDefiningOp(),
                                     value.getDefiningOp());
  // Get all possible aliases
  auto possibleValues = aliases.resolve(value);
  return BufferAssignmentPositions(
      getAllocPosition(value, liveness, possibleValues, dominators),
      getDeallocPosition(value, liveness, possibleValues, postDominators));
}

}  // namespace xla
}  // namespace mlir
