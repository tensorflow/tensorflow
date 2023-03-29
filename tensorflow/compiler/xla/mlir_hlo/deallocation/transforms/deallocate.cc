/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "deallocation/utils/util.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace deallocation {
namespace {

bool isMemref(Value v) { return v.getType().isa<BaseMemRefType>(); }

struct TransformResult {
  // Allocs that are no longer owned by the current block. Note that it is valid
  // for an alloc to be both in `acquired` and `released`, if it was temporarily
  // released and then reacquired. It is valid to release an alloc that's not
  // owned by the current block, if some ancestor that is reachable without
  // crossing a loop boundary owns it.
  breaks_if_you_move_ops::ValueSet released;

  // Allocs that are now owned by the current block. Order matters here - it's
  // the same order as in the terminator/result list.
  SmallVector<Value> acquired;
};

bool doesAlias(Operation* op, Value v,
               breaks_if_you_move_ops::ValueEquivalenceClasses& aliases,
               bool considerOperands = true) {
  auto eq = [&](Value other) { return aliases.isEquivalent(v, other); };
  return op && ((considerOperands && llvm::any_of(op->getOperands(), eq)) ||
                llvm::any_of(op->getResults(), eq) ||
                llvm::any_of(op->getRegions(), [&](Region& region) {
                  return llvm::any_of(region.getOps(), [&](Operation& subOp) {
                    return doesAlias(&subOp, v, aliases);
                  });
                }));
}

struct Deallocator {
  // Transforms the operation and returns any allocations whose ownership is
  // transferred to the parent block.
  // `ownedMemrefs` contains the memrefs owned by the immediate parent block at
  // the point of `op`.
  FailureOr<TransformResult> transformOp(
      Operation* op, const breaks_if_you_move_ops::ValueSet& ownedMemrefs);
  FailureOr<TransformResult> transformOp(
      RegionBranchOpInterface op,
      const breaks_if_you_move_ops::ValueSet& ownedMemrefs);
  // Transforms the block and collects newly acquired/released allocs. Does not
  // modify the block's terminator.
  FailureOr<TransformResult> transformBlock(Block& block,
                                            bool ownsInputs = true);

  // If `value` is guaranteed to be derived from a particular alloc, returns it.
  // Otherwise, returns null.
  Value getUniquePossibleAlloc(Value value);

  breaks_if_you_move_ops::ValueEquivalenceClasses aliases;
};

Value Deallocator::getUniquePossibleAlloc(Value v) {
  Value result = {};
  for (auto it = aliases.findLeader(v); it != aliases.member_end(); ++it) {
    if (it->getType().isa<OwnershipIndicatorType>()) {
      if (result) return {};
      result = *it;
    }
  }
  return result;
}

FailureOr<TransformResult> Deallocator::transformBlock(Block& block,
                                                       bool ownsInputs) {
  auto loc = block.getParent()->getLoc();
  auto ownershipTy = OwnershipIndicatorType::get(loc.getContext());
  // Introduce block arguments for the owned inputs.
  breaks_if_you_move_ops::ValueSet ownedMemrefs;
  if (ownsInputs) {
    for (auto arg : llvm::to_vector(
             llvm::make_filter_range(block.getArguments(), isMemref))) {
      // Add an argument for a potentially owned memref.
      auto newArg = block.addArgument(ownershipTy, loc);
      ownedMemrefs.insert(newArg);
      aliases.unionSets(arg, newArg);
    }
  }

  for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
    auto opResult = transformOp(&op, ownedMemrefs);
    if (failed(opResult)) return failure();
    // Remove released memrefs.
    for (auto v : opResult->released) {
      if (!ownedMemrefs.erase(v)) {
        block.getParentOp()->emitOpError(
            "released an alloc that was not owned");
        return failure();
      }
    }
    ownedMemrefs.insert(opResult->acquired.begin(), opResult->acquired.end());
  }
  auto yieldedMemrefs = llvm::to_vector(
      llvm::make_filter_range(block.getTerminator()->getOperands(), isMemref));

  // Handle owned memrefs that don't alias with any yielded memref first.
  for (auto v : ownedMemrefs) {
    if (!llvm::any_of(yieldedMemrefs, [&](Value yielded) {
          return aliases.isEquivalent(yielded, v);
        })) {
      // This owned memref does not escape, so we can put it in its own
      // retain and place it as early as possible.
      auto* insertionPoint = block.getTerminator();
      while (insertionPoint->getPrevNode() &&
             !doesAlias(insertionPoint->getPrevNode(), v, aliases)) {
        insertionPoint = insertionPoint->getPrevNode();
      }
      ImplicitLocOpBuilder b(loc, insertionPoint);
      b.create<RetainOp>(TypeRange{}, ValueRange{}, ValueRange{v});
    }
  }

  // Group yielded memrefs and owned memrefs by equivalence class leader.
  auto groupByLeader = [&](auto& values) {
    breaks_if_you_move_ops::ValueMap<SmallVector<Value>> result;
    for (auto v : values) {
      aliases.insert(v);
      result[aliases.getLeaderValue(v)].push_back(v);
    }
    return result;
  };
  auto yieldedByLeader = groupByLeader(yieldedMemrefs);
  auto ownedByLeader = groupByLeader(ownedMemrefs);

  // Create one retain per equivalence class.
  ImplicitLocOpBuilder b(loc, block.getTerminator());
  TransformResult blockResult;
  auto null = b.create<NullOp>();
  blockResult.acquired =
      SmallVector<Value>(yieldedMemrefs.size(), null.getResult());
  for (auto [leader, yielded] : yieldedByLeader) {
    auto& ownedGroup = ownedByLeader[leader];
    if (ownedGroup.size() == 1 && yielded.size() == 1 &&
        getUniquePossibleAlloc(yielded.front()) == ownedGroup.front()) {
      // We know the alloc that the yielded memref is derived from, so we can
      // omit the retain op. This would better be a canonicalization pattern,
      // but it requires an alias analysis, which we already have here.
      blockResult.acquired[llvm::find(yieldedMemrefs, yielded.front()) -
                           yieldedMemrefs.begin()] = ownedGroup.front();
      continue;
    }

    SmallVector<Type> types(yielded.size(), ownershipTy);
    auto retain = b.create<RetainOp>(types, yielded, ownedGroup);
    for (auto [retained, result] : llvm::zip(retain.getResults(), yielded)) {
      aliases.unionSets(retained, result);
      blockResult.acquired[llvm::find(yieldedMemrefs, result) -
                           yieldedMemrefs.begin()] = retained;
    }
  }
  if (!llvm::is_contained(blockResult.acquired, null.getResult())) null.erase();
  return blockResult;
}

FailureOr<TransformResult> Deallocator::transformOp(
    RegionBranchOpInterface op,
    const breaks_if_you_move_ops::ValueSet& ownedMemrefs) {
  SmallVector<int64_t> originalNumArgsByRegion;
  SmallVector<TransformResult> transformResultsByRegion;
  transformResultsByRegion.reserve(op->getNumRegions());

  for (auto [index, region] : llvm::enumerate(op->getRegions())) {
    assert(region.getBlocks().size() <= 1 &&
           "expected regions to have at most one block");
    auto edges = getSuccessorRegions(op, index);
    originalNumArgsByRegion.push_back(region.getNumArguments());

    auto& result = transformResultsByRegion.emplace_back();
    if (region.empty()) continue;

    // Transform region and collect owned memrefs.
    auto transformResultOrError = transformBlock(region.front());
    if (failed(transformResultOrError)) {
      return failure();
    }
    if (!transformResultOrError->released.empty()) {
      op->emitOpError("block unexpectededly released a memref");
    }
    result = *std::move(transformResultOrError);
  }

  // Adjust terminator operands.
  for (auto [region, transformResult] :
       llvm::zip(op->getRegions(), transformResultsByRegion)) {
    if (region.empty()) continue;
    auto* terminator = region.front().getTerminator();
    terminator->setOperands(terminator->getNumOperands(), 0,
                            transformResult.acquired);
  }

  ImplicitLocOpBuilder b(op.getLoc(), op);
  SmallVector<Value> operands = op->getOperands();
  breaks_if_you_move_ops::ValueSet released;
  // If we pass an owned memref to the loop and don't reuse it afterwards, we
  // can transfer ownership.
  for (auto operand : llvm::make_filter_range(operands, isMemref)) {
    auto isLastUse = [&]() {
      for (auto* candidate = op.getOperation(); candidate != nullptr;
           candidate = candidate->getNextNode()) {
        if (doesAlias(candidate, operand, aliases,
                      /*considerOperands=*/candidate != op.getOperation()))
          return false;
      }
      return true;
    };

    auto eq = [&](Value v) { return aliases.isEquivalent(v, operand); };
    auto releasable = llvm::find_if(ownedMemrefs, eq);
    bool isReleasable =
        releasable != ownedMemrefs.end() && llvm::none_of(released, eq);
    if (isReleasable && isLastUse()) {
      // This is an alloc that is not used again, so we can pass ownership
      // to the loop.
      op->insertOperands(op->getNumOperands(), *releasable);
      released.insert(*releasable);
    } else {
      // Either the operand is not an alloc or it's reused.
      op->insertOperands(op->getNumOperands(), b.create<NullOp>().getResult());
    }
  }

  RegionBranchOpInterface newOp = moveRegionsToNewOpButKeepOldOp(op);
  auto numOriginalResults = op->getNumResults();
  auto newResults = newOp->getResults().take_front(numOriginalResults);
  auto retained = newOp->getResults().drop_front(numOriginalResults);
  op->replaceAllUsesWith(newResults);
  op->erase();

  auto setupAliases = [&](std::optional<unsigned> index) {
    for (auto& region : getSuccessorRegions(newOp, index)) {
      for (auto [pred, succ] : llvm::zip(region.getPredecessorOperands(),
                                         region.getSuccessorValues())) {
        aliases.unionSets(pred, succ);
      }
    }
  };
  auto setMemrefAliases = [this](ValueRange a, ValueRange b) {
    for (auto [aa, bb] : llvm::zip(llvm::make_filter_range(a, isMemref), b)) {
      aliases.unionSets(aa, bb);
    }
  };
  setupAliases(std::nullopt);
  for (uint32_t i = 0; i < newOp->getNumRegions(); ++i) {
    setupAliases(i);
    auto args = newOp->getRegion(i).getArguments();
    auto n = originalNumArgsByRegion[i];
    setMemrefAliases(args.take_front(n), args.drop_front(n));
  }
  setMemrefAliases(newResults, retained);
  return TransformResult{released, retained};
}

// Returns the set of values that are potentially owned by the op.
FailureOr<TransformResult> Deallocator::transformOp(
    Operation* op, const breaks_if_you_move_ops::ValueSet& ownedMemrefs) {
  if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(op)) {
    return transformOp(rbi, ownedMemrefs);
  }
  if (auto alloc = llvm::dyn_cast<memref::AllocOp>(op)) {
    OpBuilder b(alloc.getContext());
    b.setInsertionPointAfter(alloc);
    auto owned = b.create<deallocation::OwnOp>(alloc.getLoc(), alloc);
    aliases.unionSets(alloc, owned);
    return TransformResult{{}, {owned}};
  }
  if (llvm::isa<memref::DeallocOp>(op)) {
    op->emitOpError("expected no dealloc ops to be present in the input");
    return failure();
  }
  if (auto func = llvm::dyn_cast<func::FuncOp>(op)) {
    auto transformedBlock =
        transformBlock(func.getBody().front(), /*ownsInputs=*/false);
    if (failed(transformedBlock)) return failure();
    if (!transformedBlock->released.empty()) {
      op->emitOpError("block unexpectededly released a memref");
    }
    return TransformResult{{}, transformedBlock->acquired};
  }

  // Deallocate ops inside unknown op regions.
  // Also assert that unknown ops with regions return no memrefs. There is no
  // way to generically transform such ops, if they exist. Eventually we'll need
  // an interface for this.
  if (op->getNumRegions() > 0) {
    assert(llvm::none_of(op->getResults(), isMemref));
    for (auto& region : op->getRegions()) {
      for (auto& block : region.getBlocks()) {
        auto transformedBlock = transformBlock(block, /*ownsInputs=*/false);
        if (failed(transformedBlock)) return failure();
        if (!transformedBlock->acquired.empty() ||
            !transformedBlock->released.empty()) {
          op->emitOpError("block unexpectededly released or returned an alloc");
          return failure();
        }
      }
    }
  }

  // Assume any memref operand may alias any memref result.
  for (auto result : llvm::make_filter_range(op->getResults(), isMemref)) {
    for (auto arg : llvm::make_filter_range(op->getOperands(), isMemref)) {
      if (getElementTypeOrSelf(result.getType()) ==
          getElementTypeOrSelf(arg.getType())) {
        aliases.unionSets(result, arg);
      }
    }
  }
  // No new allocations or releases.
  return TransformResult{};
}

#define GEN_PASS_DEF_DEALLOCATEPASS
#include "deallocation/transforms/passes.h.inc"

struct DeallocatePass : public impl::DeallocatePassBase<DeallocatePass> {
  void runOnOperation() override {
    if (failed(Deallocator().transformOp(getOperation(), {}))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createDeallocatePass() {
  return std::make_unique<DeallocatePass>();
}

}  // namespace deallocation
}  // namespace mlir
