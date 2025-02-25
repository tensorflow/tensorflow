/* Copyright 2025 The OpenXLA Authors.

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

// This file implements logic for computing correct alloc and dealloc positions.
// Furthermore, buffer deallocation also adds required new clone operations to
// ensure that all buffers are deallocated. The main class is the
// BufferDeallocationPass class that implements the underlying algorithm. In
// order to put allocations and deallocations at safe positions, it is
// significantly important to put them into the correct blocks. However, the
// liveness analysis does not pay attention to aliases, which can occur due to
// branches (and their associated block arguments) in general. For this purpose,
// BufferDeallocation firstly finds all possible aliases for a single value
// (using the BufferViewFlowAnalysis class). Consider the following example:
//
// ^bb0(%arg0):
//   cf.cond_br %cond, ^bb1, ^bb2
// ^bb1:
//   cf.br ^exit(%arg0)
// ^bb2:
//   %new_value = ...
//   cf.br ^exit(%new_value)
// ^exit(%arg1):
//   return %arg1;
//
// We should place the dealloc for %new_value in exit. However, we have to free
// the buffer in the same block, because it cannot be freed in the post
// dominator. However, this requires a new clone buffer for %arg1 that will
// contain the actual contents. Using the class BufferViewFlowAnalysis, we
// will find out that %new_value has a potential alias %arg1. In order to find
// the dealloc position we have to find all potential aliases, iterate over
// their uses and find the common post-dominator block (note that additional
// clones and buffers remove potential aliases and will influence the placement
// of the deallocs). In all cases, the computed block can be safely used to free
// the %new_value buffer (may be exit or bb2) as it will die and we can use
// liveness information to determine the exact operation after which we have to
// insert the dealloc. However, the algorithm supports introducing clone buffers
// and placing deallocs in safe locations to ensure that all buffers will be
// freed in the end.
//
// TODO:
// The current implementation does not support explicit-control-flow loops and
// the resulting code will be invalid with respect to program semantics.
// However, structured control-flow loops are fully supported. Furthermore, it
// doesn't accept functions which return buffers already.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

#include "deallocation/transforms/passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace deallocation {
#define GEN_PASS_DEF_BUFFERDEALLOCATION
#include "deallocation/transforms/passes.h.inc"

namespace {

using mlir::Block;
using mlir::BlockArgument;
using mlir::BranchOpInterface;
using mlir::DialectRegistry;
using mlir::DominanceInfo;
using mlir::failure;
using mlir::FailureOr;
using mlir::LivenessBlockInfo;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OperandRange;
using mlir::Operation;
using mlir::Pass;
using mlir::PostDominanceInfo;
using mlir::Region;
using mlir::RegionBranchOpInterface;
using mlir::RegionBranchPoint;
using mlir::RegionBranchTerminatorOpInterface;
using mlir::RegionSuccessor;
using mlir::SetVector;
using mlir::SmallVector;
using mlir::success;
using mlir::SuccessorOperands;
using mlir::Value;
using mlir::WalkResult;
using mlir::bufferization::AllocationOpInterface;
using mlir::bufferization::BufferPlacementAllocs;
using mlir::bufferization::BufferPlacementTransformationBase;
using mlir::bufferization::CloneOp;
using mlir::func::FuncOp;
using mlir::memref::DeallocOp;

/// Walks over all immediate return-like terminators in the given region.
static LogicalResult walkReturnOperations(
    Region *region,
    llvm::function_ref<LogicalResult(RegionBranchTerminatorOpInterface)> func) {
  for (Block &block : *region) {
    Operation *terminator = block.getTerminator();
    // Skip non region-return-like terminators.
    if (auto regionTerminator =
            dyn_cast<RegionBranchTerminatorOpInterface>(terminator)) {
      if (failed(func(regionTerminator))) {
        return failure();
      }
    }
  }
  return success();
}

/// Checks if all operations that have at least one attached region implement
/// the RegionBranchOpInterface. This is not required in edge cases, where we
/// have a single attached region and the parent operation has no results.
static bool validateSupportedControlFlow(Operation *op) {
  WalkResult result = op->walk([&](Operation *operation) {
    // Only check ops that are inside a function.
    if (!operation->getParentOfType<FuncOp>()) {
      return WalkResult::advance();
    }

    auto regions = operation->getRegions();
    // Walk over all operations in a region and check if the operation has at
    // least one region and implements the RegionBranchOpInterface. If there
    // is an operation that does not fulfill this condition, we cannot apply
    // the deallocation steps. Furthermore, we accept cases, where we have a
    // region that returns no results, since, in that case, the intra-region
    // control flow does not affect the transformation.
    size_t size = regions.size();
    if (((size == 1 && !operation->getResults().empty()) || size > 1) &&
        !dyn_cast<RegionBranchOpInterface>(operation)) {
      operation->emitError(
          "All operations with attached regions need to "
          "implement the RegionBranchOpInterface.");
    }

    return WalkResult::advance();
  });
  return !result.wasSkipped();
}

//===----------------------------------------------------------------------===//
// Backedges analysis
//===----------------------------------------------------------------------===//

/// A straight-forward program analysis which detects loop backedges induced by
/// explicit control flow.
class Backedges {
 public:
  using BlockSetT = mlir::SmallPtrSet<Block *, 16>;
  using BackedgeSetT = llvm::DenseSet<std::pair<Block *, Block *>>;

  /// Constructs a new backedges analysis using the op provided.
  explicit Backedges(Operation *op) { recurse(op); }

  /// Returns the number of backedges formed by explicit control flow.
  size_t size() const { return edgeSet.size(); }

  /// Returns the start iterator to loop over all backedges.
  BackedgeSetT::const_iterator begin() const { return edgeSet.begin(); }

  /// Returns the end iterator to loop over all backedges.
  BackedgeSetT::const_iterator end() const { return edgeSet.end(); }

 private:
  /// Enters the current block and inserts a backedge into the `edgeSet` if we
  /// have already visited the current block. The inserted edge links the given
  /// `predecessor` with the `current` block.
  bool enter(Block &current, Block *predecessor) {
    bool inserted = visited.insert(&current).second;
    if (!inserted) {
      edgeSet.insert(std::make_pair(predecessor, &current));
    }
    return inserted;
  }

  /// Leaves the current block.
  void exit(Block &current) { visited.erase(&current); }

  /// Recurses into the given operation while taking all attached regions into
  /// account.
  void recurse(Operation *op) {
    Block *current = op->getBlock();
    // If the current op implements the `BranchOpInterface`, there can be
    // cycles in the scope of all successor blocks.
    if (isa<BranchOpInterface>(op)) {
      for (Block *succ : current->getSuccessors()) {
        recurse(*succ, current);
      }
    }
    // Recurse into all distinct regions and check for explicit control-flow
    // loops.
    for (Region &region : op->getRegions()) {
      if (!region.empty()) {
        recurse(region.front(), current);
      }
    }
  }

  /// Recurses into explicit control-flow structures that are given by
  /// the successor relation defined on the block level.
  void recurse(Block &block, Block *predecessor) {
    // Try to enter the current block. If this is not possible, we are
    // currently processing this block and can safely return here.
    if (!enter(block, predecessor)) {
      return;
    }

    // Recurse into all operations and successor blocks.
    for (Operation &op : block.getOperations()) {
      recurse(&op);
    }

    // Leave the current block.
    exit(block);
  }

  /// Stores all blocks that are currently visited and on the processing stack.
  BlockSetT visited;

  /// Stores all backedges in the format (source, target).
  BackedgeSetT edgeSet;
};

//===----------------------------------------------------------------------===//
// BufferDeallocation
//===----------------------------------------------------------------------===//

/// The buffer deallocation transformation which ensures that all allocs in the
/// program have a corresponding de-allocation. As a side-effect, it might also
/// introduce clones that in turn leads to additional deallocations.
class BufferDeallocation : public BufferPlacementTransformationBase {
 public:
  using AliasAllocationMapT =
      llvm::DenseMap<Value, mlir::bufferization::AllocationOpInterface>;

  explicit BufferDeallocation(Operation *op)
      : BufferPlacementTransformationBase(op),
        dominators(op),
        postDominators(op) {}

  /// Checks if all allocation operations either provide an already existing
  /// deallocation operation or implement the AllocationOpInterface. In
  /// addition, this method initializes the internal alias to
  /// AllocationOpInterface mapping in order to get compatible
  /// AllocationOpInterface implementations for aliases.
  LogicalResult prepare() {
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      // Get the defining allocation operation.
      Value alloc = std::get<0>(entry);
      auto allocationInterface =
          alloc.getDefiningOp<mlir::bufferization::AllocationOpInterface>();
      // If there is no existing deallocation operation and no implementation of
      // the AllocationOpInterface, we cannot apply the BufferDeallocation pass.
      if (!std::get<1>(entry) && !allocationInterface) {
        return alloc.getDefiningOp()->emitError(
            "Allocation is not deallocated explicitly nor does the operation "
            "implement the AllocationOpInterface.");
      }

      // Register the current allocation interface implementation.
      aliasToAllocations[alloc] = allocationInterface;

      // Get the alias information for the current allocation node.
      for (Value alias : aliases.resolve(alloc)) {
        // TODO: check for incompatible implementations of the
        // AllocationOpInterface. This could be realized by promoting the
        // AllocationOpInterface to a DialectInterface.
        aliasToAllocations[alias] = allocationInterface;
      }
    }
    return success();
  }

  /// Performs the actual placement/creation of all temporary clone and dealloc
  /// nodes.
  LogicalResult deallocate() {
    // Add additional clones that are required.
    if (failed(introduceClones())) {
      return failure();
    }

    // Place deallocations for all allocation entries.
    return placeDeallocs();
  }

 private:
  /// Introduces required clone operations to avoid memory leaks.
  LogicalResult introduceClones() {
    // Initialize the set of values that require a dedicated memory free
    // operation since their operands cannot be safely deallocated in a post
    // dominator.
    SetVector<Value> valuesToFree;
    llvm::SmallDenseSet<std::tuple<Value, Block *>> visitedValues;
    SmallVector<std::tuple<Value, Block *>, 8> toProcess;

    // Check dominance relation for proper dominance properties. If the given
    // value node does not dominate an alias, we will have to create a clone in
    // order to free all buffers that can potentially leak into a post
    // dominator.
    auto findUnsafeValues = [&](Value source, Block *definingBlock) {
      auto it = aliases.find(source);
      if (it == aliases.end()) {
        return;
      }
      for (Value value : it->second) {
        if (valuesToFree.count(value) > 0) {
          continue;
        }
        Block *parentBlock = value.getParentBlock();
        // Check whether we have to free this particular block argument or
        // generic value. We have to free the current alias if it is either
        // defined in a non-dominated block or it is defined in the same block
        // but the current value is not dominated by the source value.
        if (!dominators.dominates(definingBlock, parentBlock) ||
            (definingBlock == parentBlock && isa<BlockArgument>(value))) {
          toProcess.emplace_back(value, parentBlock);
          valuesToFree.insert(value);
        } else if (visitedValues.insert(std::make_tuple(value, definingBlock))
                       .second) {
          toProcess.emplace_back(value, definingBlock);
        }
      }
    };

    // Detect possibly unsafe aliases starting from all allocations.
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      findUnsafeValues(allocValue, allocValue.getDefiningOp()->getBlock());
    }
    // Try to find block arguments that require an explicit free operation
    // until we reach a fix point.
    while (!toProcess.empty()) {
      auto current = toProcess.pop_back_val();
      findUnsafeValues(std::get<0>(current), std::get<1>(current));
    }

    // Update buffer aliases to ensure that we free all buffers and block
    // arguments at the correct locations.
    aliases.remove(valuesToFree);

    // Add new allocs and additional clone operations.
    for (Value value : valuesToFree) {
      if (failed(isa<BlockArgument>(value)
                     ? introduceBlockArgCopy(cast<BlockArgument>(value))
                     : introduceValueCopyForRegionResult(value))) {
        return failure();
      }

      // Register the value to require a final dealloc. Note that we do not have
      // to assign a block here since we do not want to move the allocation node
      // to another location.
      allocs.registerAlloc(std::make_tuple(value, nullptr));
    }
    return success();
  }

  /// Introduces temporary clones in all predecessors and copies the source
  /// values into the newly allocated buffers.
  LogicalResult introduceBlockArgCopy(BlockArgument blockArg) {
    // Allocate a buffer for the current block argument in the block of
    // the associated value (which will be a predecessor block by
    // definition).
    Block *block = blockArg.getOwner();
    for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
      // Get the terminator and the value that will be passed to our
      // argument.
      Operation *terminator = (*it)->getTerminator();
      auto branchInterface = cast<mlir::BranchOpInterface>(terminator);
      SuccessorOperands operands =
          branchInterface.getSuccessorOperands(it.getSuccessorIndex());

      // Query the associated source value.
      Value sourceValue = operands[blockArg.getArgNumber()];
      if (!sourceValue) {
        return failure();
      }
      // Wire new clone and successor operand.
      // Create a new clone at the current location of the terminator.
      auto clone = introduceCloneBuffers(sourceValue, terminator);
      if (failed(clone)) {
        return failure();
      }
      operands.slice(blockArg.getArgNumber(), 1).assign(*clone);
    }

    // Check whether the block argument has implicitly defined predecessors via
    // the RegionBranchOpInterface. This can be the case if the current block
    // argument belongs to the first block in a region and the parent operation
    // implements the RegionBranchOpInterface.
    Region *argRegion = block->getParent();
    Operation *parentOp = argRegion->getParentOp();
    RegionBranchOpInterface regionInterface;
    if (&argRegion->front() != block ||
        !(regionInterface = dyn_cast<RegionBranchOpInterface>(parentOp))) {
      return success();
    }

    if (failed(introduceClonesForRegionSuccessors(
            regionInterface, argRegion->getParentOp()->getRegions(), blockArg,
            [&](RegionSuccessor &successorRegion) {
              // Find a predecessor of our argRegion.
              return successorRegion.getSuccessor() == argRegion;
            }))) {
      return failure();
    }

    // Check whether the block argument belongs to an entry region of the
    // parent operation. In this case, we have to introduce an additional clone
    // for buffer that is passed to the argument.
    SmallVector<RegionSuccessor, 2> successorRegions;
    regionInterface.getSuccessorRegions(/*point=*/RegionBranchPoint::parent(),
                                        successorRegions);
    auto *it =
        llvm::find_if(successorRegions, [&](RegionSuccessor &successorRegion) {
          return successorRegion.getSuccessor() == argRegion;
        });
    if (it == successorRegions.end()) {
      return success();
    }

    // Determine the actual operand to introduce a clone for and rewire the
    // operand to point to the clone instead.
    auto operands = regionInterface.getEntrySuccessorOperands(argRegion);
    size_t operandIndex =
        llvm::find(it->getSuccessorInputs(), blockArg).getIndex() +
        operands.getBeginOperandIndex();
    Value operand = parentOp->getOperand(operandIndex);
    assert(operand ==
               operands[operandIndex - operands.getBeginOperandIndex()] &&
           "region interface operands don't match parentOp operands");
    auto clone = introduceCloneBuffers(operand, parentOp);
    if (failed(clone)) {
      return failure();
    }

    parentOp->setOperand(operandIndex, *clone);
    return success();
  }

  /// Introduces temporary clones in front of all associated nested-region
  /// terminators and copies the source values into the newly allocated buffers.
  LogicalResult introduceValueCopyForRegionResult(Value value) {
    // Get the actual result index in the scope of the parent terminator.
    Operation *operation = value.getDefiningOp();
    auto regionInterface = cast<RegionBranchOpInterface>(operation);
    // Filter successors that return to the parent operation.
    auto regionPredicate = [&](RegionSuccessor &successorRegion) {
      // If the RegionSuccessor has no associated successor, it will return to
      // its parent operation.
      return !successorRegion.getSuccessor();
    };
    // Introduce a clone for all region "results" that are returned to the
    // parent operation. This is required since the parent's result value has
    // been considered critical. Therefore, the algorithm assumes that a clone
    // of a previously allocated buffer is returned by the operation (like in
    // the case of a block argument).
    return introduceClonesForRegionSuccessors(
        regionInterface, operation->getRegions(), value, regionPredicate);
  }

  /// Introduces buffer clones for all terminators in the given regions. The
  /// regionPredicate is applied to every successor region in order to restrict
  /// the clones to specific regions.
  template <typename TPredicate>
  LogicalResult introduceClonesForRegionSuccessors(
      RegionBranchOpInterface regionInterface,
      mlir::MutableArrayRef<Region> regions, Value argValue,
      const TPredicate &regionPredicate) {
    for (Region &region : regions) {
      // Query the regionInterface to get all successor regions of the current
      // one.
      SmallVector<RegionSuccessor, 2> successorRegions;
      regionInterface.getSuccessorRegions(region, successorRegions);
      // Try to find a matching region successor.
      RegionSuccessor *regionSuccessor =
          llvm::find_if(successorRegions, regionPredicate);
      if (regionSuccessor == successorRegions.end()) {
        continue;
      }
      // Get the operand index in the context of the current successor input
      // bindings.
      size_t operandIndex =
          llvm::find(regionSuccessor->getSuccessorInputs(), argValue)
              .getIndex();

      // Iterate over all immediate terminator operations to introduce
      // new buffer allocations. Thereby, the appropriate terminator operand
      // will be adjusted to point to the newly allocated buffer instead.
      if (failed(walkReturnOperations(
              &region, [&](RegionBranchTerminatorOpInterface terminator) {
                // Get the actual mutable operands for this terminator op.
                auto terminatorOperands =
                    terminator.getMutableSuccessorOperands(*regionSuccessor);
                // Extract the source value from the current terminator.
                // This conversion needs to exist on a separate line due to a
                // bug in GCC conversion analysis.
                OperandRange immutableTerminatorOperands = terminatorOperands;
                Value sourceValue = immutableTerminatorOperands[operandIndex];
                // Create a new clone at the current location of the terminator.
                auto clone = introduceCloneBuffers(sourceValue, terminator);
                if (failed(clone)) {
                  return failure();
                }
                // Wire clone and terminator operand.
                terminatorOperands.slice(operandIndex, 1).assign(*clone);
                return success();
              }))) {
        return failure();
      }
    }
    return success();
  }

  /// Creates a new memory allocation for the given source value and clones
  /// its content into the newly allocated buffer. The terminator operation is
  /// used to insert the clone operation at the right place.
  FailureOr<Value> introduceCloneBuffers(Value sourceValue,
                                         Operation *terminator) {
    // Avoid multiple clones of the same source value. This can happen in the
    // presence of loops when a branch acts as a backedge while also having
    // another successor that returns to its parent operation. Note: that
    // copying copied buffers can introduce memory leaks since the invariant of
    // BufferDeallocation assumes that a buffer will be only cloned once into a
    // temporary buffer. Hence, the construction of clone chains introduces
    // additional allocations that are not tracked automatically by the
    // algorithm.
    if (clonedValues.contains(sourceValue)) {
      return sourceValue;
    }
    // Create a new clone operation that copies the contents of the old
    // buffer to the new one.
    auto clone = buildClone(terminator, sourceValue);
    if (succeeded(clone)) {
      // Remember the clone of original source value.
      clonedValues.insert(*clone);
    }
    return clone;
  }

  /// Finds correct dealloc positions according to the algorithm described at
  /// the top of the file for all alloc nodes and block arguments that can be
  /// handled by this analysis.
  LogicalResult placeDeallocs() {
    // Move or insert deallocs using the previously computed information.
    // These deallocations will be linked to their associated allocation nodes
    // since they don't have any aliases that can (potentially) increase their
    // liveness.
    for (const BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      auto aliasesSet = aliases.resolve(alloc);
      assert(!aliasesSet.empty() && "must contain at least one alias");

      // Determine the actual block to place the dealloc and get liveness
      // information.
      Block *placementBlock = mlir::bufferization::findCommonDominator(
          alloc, aliasesSet, postDominators);
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(placementBlock);

      // We have to ensure that the dealloc will be after the last use of all
      // aliases of the given value. We first assume that there are no uses in
      // the placementBlock and that we can safely place the dealloc at the
      // beginning.
      Operation *endOperation = &placementBlock->front();

      // Iterate over all aliases and ensure that the endOperation will point
      // to the last operation of all potential aliases in the placementBlock.
      for (Value alias : aliasesSet) {
        // Ensure that the start operation is at least the defining operation of
        // the current alias to avoid invalid placement of deallocs for aliases
        // without any uses.
        Operation *beforeOp = endOperation;
        if (alias.getDefiningOp() &&
            !(beforeOp = placementBlock->findAncestorOpInBlock(
                  *alias.getDefiningOp()))) {
          continue;
        }

        Operation *aliasEndOperation =
            livenessInfo->getEndOperation(alias, beforeOp);
        // Check whether the aliasEndOperation lies in the desired block and
        // whether it is behind the current endOperation. If yes, this will be
        // the new endOperation.
        if (aliasEndOperation->getBlock() == placementBlock &&
            endOperation->isBeforeInBlock(aliasEndOperation)) {
          endOperation = aliasEndOperation;
        }
      }
      // endOperation is the last operation behind which we can safely store
      // the dealloc taking all potential aliases into account.

      // If there is an existing dealloc, move it to the right place.
      Operation *deallocOperation = std::get<1>(entry);
      if (deallocOperation) {
        deallocOperation->moveAfter(endOperation);
      } else {
        // If the Dealloc position is at the terminator operation of the
        // block, then the value should escape from a deallocation.
        Operation *nextOp = endOperation->getNextNode();
        if (!nextOp) {
          continue;
        }
        // If there is no dealloc node, insert one in the right place.
        if (failed(buildDealloc(nextOp, alloc))) {
          return failure();
        }
      }
    }
    return success();
  }

  /// Builds a deallocation operation compatible with the given allocation
  /// value. If there is no registered AllocationOpInterface implementation for
  /// the given value (e.g. in the case of a function parameter), this method
  /// builds a memref::DeallocOp.
  LogicalResult buildDealloc(Operation *op, Value alloc) {
    OpBuilder builder(op);
    auto it = aliasToAllocations.find(alloc);
    if (it != aliasToAllocations.end()) {
      // Call the allocation op interface to build a supported and
      // compatible deallocation operation.
      auto dealloc = it->second.buildDealloc(builder, alloc);
      if (!dealloc) {
        return op->emitError()
               << "allocations without compatible deallocations are "
                  "not supported";
      }
    } else {
      // Build a "default" DeallocOp for unknown allocation sources.
      builder.create<mlir::memref::DeallocOp>(alloc.getLoc(), alloc);
    }
    return success();
  }

  /// Builds a clone operation compatible with the given allocation value. If
  /// there is no registered AllocationOpInterface implementation for the given
  /// value (e.g. in the case of a function parameter), this method builds a
  /// bufferization::CloneOp.
  FailureOr<Value> buildClone(Operation *op, Value alloc) {
    OpBuilder builder(op);
    auto it = aliasToAllocations.find(alloc);
    if (it != aliasToAllocations.end()) {
      // Call the allocation op interface to build a supported and
      // compatible clone operation.
      auto clone = it->second.buildClone(builder, alloc);
      if (clone) {
        return *clone;
      }
      return (LogicalResult)(op->emitError()
                             << "allocations without compatible clone ops "
                                "are not supported");
    }
    // Build a "default" CloneOp for unknown allocation sources.
    return builder.create<mlir::bufferization::CloneOp>(alloc.getLoc(), alloc)
        .getResult();
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The post dominator info to move the dependent allocs in the right
  /// position.
  PostDominanceInfo postDominators;

  /// Stores already cloned buffers to avoid additional clones of clones.
  ValueSetT clonedValues;

  /// Maps aliases to their source allocation interfaces (inverse mapping).
  AliasAllocationMapT aliasToAllocations;
};

LogicalResult deallocateBuffers(Operation *op) {
  if (isa<ModuleOp>(op)) {
    WalkResult result = op->walk([&](FuncOp funcOp) {
      if (failed(deallocateBuffers(funcOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  // Ensure that there are supported loops only.
  Backedges backedges(op);
  if (backedges.size()) {
    op->emitError("Only structured control-flow loops are supported.");
    return failure();
  }

  // Check that the control flow structures are supported.
  if (!validateSupportedControlFlow(op)) {
    return failure();
  }

  // Gather all required allocation nodes and prepare the deallocation phase.
  BufferDeallocation deallocation(op);

  // Check for supported AllocationOpInterface implementations and prepare the
  // internal deallocation pass.
  if (failed(deallocation.prepare())) {
    return failure();
  }

  // Place all required temporary clone and dealloc nodes.
  if (failed(deallocation.deallocate())) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BufferDeallocationPass
//===----------------------------------------------------------------------===//

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional clones if
/// necessary. It uses the algorithm described at the top of the file.
struct BufferDeallocationPass
    : public impl::BufferDeallocationBase<BufferDeallocationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }

    if (failed(deallocateBuffers(func))) {
      signalPassFailure();
    }
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// BufferDeallocationPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createBufferDeallocationPass() {
  return std::make_unique<BufferDeallocationPass>();
}

}  // namespace deallocation
}  // namespace mlir
