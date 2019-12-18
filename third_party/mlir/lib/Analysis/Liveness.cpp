//===- Liveness.cpp - Liveness analysis for MLIR --------------------------===//
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
// Implementation of the liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Builds and holds block information during the construction phase.
struct BlockInfoBuilder {
  using ValueSetT = Liveness::ValueSetT;

  /// Constructs an empty block builder.
  BlockInfoBuilder() : block(nullptr) {}

  /// Fills the block builder with initial liveness information.
  BlockInfoBuilder(Block *block) : block(block) {
    // Mark all block arguments (phis) as defined.
    for (BlockArgument *argument : block->getArguments())
      defValues.insert(argument);

    // Check all result values and whether their uses
    // are inside this block or not (see outValues).
    for (Operation &operation : *block)
      for (Value *result : operation.getResults()) {
        defValues.insert(result);

        // Check whether this value will be in the outValues
        // set (its uses escape this block). Due to the SSA
        // properties of the program, the uses must occur after
        // the definition. Therefore, we do not have to check
        // additional conditions to detect an escaping value.
        for (OpOperand &use : result->getUses())
          if (use.getOwner()->getBlock() != block) {
            outValues.insert(result);
            break;
          }
      }

    // Check all operations for used operands.
    for (Operation &operation : block->getOperations())
      for (Value *operand : operation.getOperands()) {
        // If the operand is already defined in the scope of this
        // block, we can skip the value in the use set.
        if (!defValues.count(operand))
          useValues.insert(operand);
      }
  }

  /// Updates live-in information of the current block.
  /// To do so it uses the default liveness-computation formula:
  /// newIn = use union out \ def.
  /// The methods returns true, if the set has changed (newIn != in),
  /// false otherwise.
  bool updateLiveIn() {
    ValueSetT newIn = useValues;
    llvm::set_union(newIn, outValues);
    llvm::set_subtract(newIn, defValues);

    // It is sufficient to check the set sizes (instead of their contents)
    // since the live-in set can only grow monotonically during all update
    // operations.
    if (newIn.size() == inValues.size())
      return false;

    inValues = newIn;
    return true;
  }

  /// Updates live-out information of the current block.
  /// It iterates over all successors and unifies their live-in
  /// values with the current live-out values.
  template <typename SourceT> void updateLiveOut(SourceT &source) {
    for (Block *succ : block->getSuccessors()) {
      BlockInfoBuilder &builder = source[succ];
      llvm::set_union(outValues, builder.inValues);
    }
  }

  /// The current block.
  Block *block;

  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  /// The set of all defined values.
  ValueSetT defValues;

  /// The set of all used values.
  ValueSetT useValues;
};

/// Builds the internal liveness block mapping.
static void buildBlockMapping(MutableArrayRef<Region> regions,
                              DenseMap<Block *, BlockInfoBuilder> &builders) {
  llvm::SetVector<Block *> toProcess;

  // Initialize all block structures
  for (Region &region : regions)
    for (Block &block : region) {
      BlockInfoBuilder &builder =
          builders.try_emplace(&block, &block).first->second;

      if (builder.updateLiveIn())
        toProcess.insert(block.pred_begin(), block.pred_end());
    }

  // Propagate the in and out-value sets (fixpoint iteration)
  while (!toProcess.empty()) {
    Block *current = toProcess.pop_back_val();
    BlockInfoBuilder &builder = builders[current];

    // Update the current out values.
    builder.updateLiveOut(builders);

    // Compute (potentially) updated live in values.
    if (builder.updateLiveIn())
      toProcess.insert(current->pred_begin(), current->pred_end());
  }
}

//===----------------------------------------------------------------------===//
// Liveness
//===----------------------------------------------------------------------===//

/// Creates a new Liveness analysis that computes liveness
/// information for all associated regions.
Liveness::Liveness(Operation *op) : operation(op) { build(op->getRegions()); }

/// Initializes the internal mappings.
void Liveness::build(MutableArrayRef<Region> regions) {

  // Build internal block mapping.
  DenseMap<Block *, BlockInfoBuilder> builders;
  buildBlockMapping(regions, builders);

  // Store internal block data.
  for (auto &entry : builders) {
    BlockInfoBuilder &builder = entry.second;
    LivenessBlockInfo &info = blockMapping[entry.first];

    info.block = builder.block;
    info.inValues = std::move(builder.inValues);
    info.outValues = std::move(builder.outValues);
  }
}

/// Gets liveness info (if any) for the given value.
Liveness::OperationListT Liveness::resolveLiveness(Value *value) const {
  OperationListT result;
  SmallPtrSet<Block *, 32> visited;
  SmallVector<Block *, 8> toProcess;

  // Start with the defining block
  Block *currentBlock;
  if (Operation *defOp = value->getDefiningOp())
    currentBlock = defOp->getBlock();
  else
    currentBlock = cast<BlockArgument>(value)->getOwner();
  toProcess.push_back(currentBlock);
  visited.insert(currentBlock);

  // Start with all associated blocks
  for (OpOperand &use : value->getUses()) {
    Block *useBlock = use.getOwner()->getBlock();
    if (visited.insert(useBlock).second)
      toProcess.push_back(useBlock);
  }

  while (!toProcess.empty()) {
    // Get block and block liveness information.
    Block *block = toProcess.back();
    toProcess.pop_back();
    const LivenessBlockInfo *blockInfo = getLiveness(block);

    // Note that start and end will be in the same block.
    Operation *start = blockInfo->getStartOperation(value);
    Operation *end = blockInfo->getEndOperation(value, start);

    result.push_back(start);
    while (start != end) {
      start = start->getNextNode();
      result.push_back(start);
    }

    for (Block *successor : block->getSuccessors()) {
      if (getLiveness(successor)->isLiveIn(value) &&
          visited.insert(successor).second)
        toProcess.push_back(successor);
    }
  }

  return result;
}

/// Gets liveness info (if any) for the block.
const LivenessBlockInfo *Liveness::getLiveness(Block *block) const {
  auto it = blockMapping.find(block);
  return it == blockMapping.end() ? nullptr : &it->second;
}

/// Returns a reference to a set containing live-in values.
const Liveness::ValueSetT &Liveness::getLiveIn(Block *block) const {
  return getLiveness(block)->in();
}

/// Returns a reference to a set containing live-out values.
const Liveness::ValueSetT &Liveness::getLiveOut(Block *block) const {
  return getLiveness(block)->out();
}

/// Returns true if the given operation represent the last use of the
/// given value.
bool Liveness::isLastUse(Value *value, Operation *operation) const {
  Block *block = operation->getBlock();
  const LivenessBlockInfo *blockInfo = getLiveness(block);

  // The given value escapes the associated block.
  if (blockInfo->isLiveOut(value))
    return false;

  Operation *endOperation = blockInfo->getEndOperation(value, operation);
  // If the operation is a real user of `value` the first check is sufficient.
  // If not, we will have to test whether the end operation is executed before
  // the given operation in the block.
  return endOperation == operation || endOperation->isBeforeInBlock(operation);
}

/// Dumps the liveness information in a human readable format.
void Liveness::dump() const { print(llvm::errs()); }

/// Dumps the liveness information to the given stream.
void Liveness::print(raw_ostream &os) const {
  os << "// ---- Liveness -----\n";

  // Builds unique block/value mappings for testing purposes.
  DenseMap<Block *, size_t> blockIds;
  DenseMap<Operation *, size_t> operationIds;
  DenseMap<Value *, size_t> valueIds;
  for (Region &region : operation->getRegions())
    for (Block &block : region) {
      blockIds.insert({&block, blockIds.size()});
      for (BlockArgument *argument : block.getArguments())
        valueIds.insert({argument, valueIds.size()});
      for (Operation &operation : block) {
        operationIds.insert({&operation, operationIds.size()});
        for (Value *result : operation.getResults())
          valueIds.insert({result, valueIds.size()});
      }
    }

  // Local printing helpers
  auto printValueRef = [&](Value *value) {
    if (Operation *defOp = value->getDefiningOp())
      os << "val_" << defOp->getName();
    else {
      auto blockArg = cast<BlockArgument>(value);
      os << "arg" << blockArg->getArgNumber() << "@"
         << blockIds[blockArg->getOwner()];
    }
    os << " ";
  };

  auto printValueRefs = [&](const ValueSetT &values) {
    std::vector<Value *> orderedValues(values.begin(), values.end());
    std::sort(orderedValues.begin(), orderedValues.end(),
              [&](Value *left, Value *right) {
                return valueIds[left] < valueIds[right];
              });
    for (Value *value : orderedValues)
      printValueRef(value);
  };

  // Dump information about in and out values.
  for (Region &region : operation->getRegions())
    for (Block &block : region) {
      os << "// - Block: " << blockIds[&block] << "\n";
      auto liveness = getLiveness(&block);
      os << "// --- LiveIn: ";
      printValueRefs(liveness->inValues);
      os << "\n// --- LiveOut: ";
      printValueRefs(liveness->outValues);
      os << "\n";

      // Print liveness intervals.
      os << "// --- BeginLiveness";
      for (Operation &op : block) {
        if (op.getNumResults() < 1)
          continue;
        os << "\n";
        for (Value *result : op.getResults()) {
          os << "// ";
          printValueRef(result);
          os << ":";
          auto liveOperations = resolveLiveness(result);
          std::sort(liveOperations.begin(), liveOperations.end(),
                    [&](Operation *left, Operation *right) {
                      return operationIds[left] < operationIds[right];
                    });
          for (Operation *operation : liveOperations) {
            os << "\n//     ";
            operation->print(os);
          }
        }
      }
      os << "\n// --- EndLiveness\n";
    }
  os << "// -------------------\n";
}

//===----------------------------------------------------------------------===//
// LivenessBlockInfo
//===----------------------------------------------------------------------===//

/// Returns true if the given value is in the live-in set.
bool LivenessBlockInfo::isLiveIn(Value *value) const {
  return inValues.count(value);
}

/// Returns true if the given value is in the live-out set.
bool LivenessBlockInfo::isLiveOut(Value *value) const {
  return outValues.count(value);
}

/// Gets the start operation for the given value
/// (must be referenced in this block).
Operation *LivenessBlockInfo::getStartOperation(Value *value) const {
  Operation *definingOp = value->getDefiningOp();
  // The given value is either live-in or is defined
  // in the scope of this block.
  if (isLiveIn(value) || !definingOp)
    return &block->front();
  return definingOp;
}

/// Gets the end operation for the given value using the start operation
/// provided (must be referenced in this block).
Operation *LivenessBlockInfo::getEndOperation(Value *value,
                                              Operation *startOperation) const {
  // The given value is either dying in this block or live-out.
  if (isLiveOut(value))
    return &block->back();

  // Resolve the last operation (must exist by definition).
  Operation *endOperation = startOperation;
  for (OpOperand &use : value->getUses()) {
    Operation *useOperation = use.getOwner();
    // Check whether the use is in our block and after
    // the current end operation.
    if (useOperation->getBlock() == block &&
        endOperation->isBeforeInBlock(useOperation))
      endOperation = useOperation;
  }
  return endOperation;
}
