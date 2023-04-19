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

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>

#include "deallocation/IR/deallocation_ops.h"
#include "deallocation/transforms/passes.h"
#include "deallocation/utils/util.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
  // Collects values that are the actual memrefs.
  breaks_if_you_move_ops::ValueSet released;

  // Allocs that are now owned by the current block. Order matters here - it's
  // the same order as in the terminator/result list.
  // Collects values that are the ownership indicators.
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
  void setOwnershipIndicator(Value owned, Value indicator);
  Value findOwnershipIndicator(Value v);

  // Transform ops, introducing deallocs.
  LogicalResult transformModuleOp(ModuleOp op);
  LogicalResult transformFuncOp(func::FuncOp op);
  FailureOr<TransformResult> transformBlock(Block& block,
                                            bool ownsInputs = true);
  FailureOr<breaks_if_you_move_ops::ValueSet> transformIfImplicitCapture(
      scf::IfOp op, TransformResult& ifResult, TransformResult& elseResult);
  FailureOr<TransformResult> transformOp(
      RegionBranchOpInterface op,
      const breaks_if_you_move_ops::ValueSet& ownedMemrefs);
  FailureOr<TransformResult> transformOp(func::CallOp op);
  FailureOr<TransformResult> transformOp(
      Operation* op, const breaks_if_you_move_ops::ValueSet& ownedMemrefs);

  // Internal state keeping track of
  //   - inter-function aliasing,
  //   - intra-function aliasing, and
  //   - ownership indicators per memref.
  std::map<func::FuncOp, SmallVector<llvm::SmallVector<int64_t>>>
      functionAliasOverapprox;
  breaks_if_you_move_ops::ValueEquivalenceClasses aliasOverapprox;
  breaks_if_you_move_ops::ValueMap<Value> ownershipIndicator;
};

void Deallocator::setOwnershipIndicator(Value owned, Value indicator) {
  ownershipIndicator[owned] = indicator;
  aliasOverapprox.unionSets(owned, indicator);
}

Value Deallocator::findOwnershipIndicator(Value v) {
  if (llvm::isa_and_nonnull<memref::SubViewOp, memref::ViewOp,
                            memref::CollapseShapeOp, memref::ExpandShapeOp,
                            memref::TransposeOp, memref::ReinterpretCastOp>(
          v.getDefiningOp())) {
    return findOwnershipIndicator(v.getDefiningOp()->getOperand(0));
  }
  auto it = ownershipIndicator.find(v);
  if (it != ownershipIndicator.end()) return it->second;
  return {};
}

LogicalResult Deallocator::transformModuleOp(ModuleOp op) {
  LogicalResult result = success();
  op.walk([&](func::FuncOp funcOp) {
    if (failed(transformFuncOp(funcOp))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result;
}

// TODO(frgossen): Also allow passing ownership to functions.
LogicalResult Deallocator::transformFuncOp(func::FuncOp op) {
  // If we find an aliasing record for this function, it is already being
  // transformed. We might be hitting a cycle in the call graph here, in which
  // case this is a temorary aliasing overapproximation and may be refined
  // later.
  if (functionAliasOverapprox.find(op) != functionAliasOverapprox.end())
    return success();

  // Mark function as being processed and provide a valid overapproximation for
  // aliasing: every result may alias every argument.
  SmallVector<llvm::SmallVector<int64_t>> trivialOverapproximation;
  int numOwnershipResults = 0;
  auto allArgs = llvm::to_vector(llvm::seq<int64_t>(0, op.getNumArguments()));
  for (Type resultTy : op.getFunctionType().getResults()) {
    auto& resultAliasing = trivialOverapproximation.emplace_back();
    if (!llvm::isa<MemRefType>(resultTy)) continue;
    resultAliasing = allArgs;
    numOwnershipResults++;
  }
  trivialOverapproximation.append(numOwnershipResults, allArgs);
  functionAliasOverapprox[op] = trivialOverapproximation;

  if (op->getNumRegions() == 0) return success();

  // Transform function body.
  assert(op.getBody().getBlocks().size() == 1 &&
         "expect single block functions");
  Block& block = op.getBody().front();
  auto transformedBlock = transformBlock(block, /*ownsInputs=*/false);
  if (failed(transformedBlock)) return failure();
  if (!transformedBlock->released.empty()) {
    op->emitOpError("invalid realloc of memref");
    return failure();
  }

  // Update terminator and pass on the ownership indicator per escaping memref.
  auto returnOp = llvm::dyn_cast<func::ReturnOp>(block.getTerminator());
  returnOp->setOperands(returnOp.getNumOperands(), 0,
                        transformedBlock->acquired);
  op.setFunctionType(mlir::FunctionType::get(
      op.getContext(), block.getArgumentTypes(), returnOp.getOperandTypes()));

  // Refine function aliasing based on return values.
  SmallVector<llvm::SmallVector<int64_t>> refinedOverapproximation;
  for (Value result : returnOp.getOperands()) {
    auto& resultAliasing = refinedOverapproximation.emplace_back();
    for (auto [j, arg] : llvm::enumerate(op.getArguments())) {
      if (aliasOverapprox.isEquivalent(result, arg))
        resultAliasing.push_back(j);
    }
  }
  functionAliasOverapprox[op] = refinedOverapproximation;

  return success();
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
      setOwnershipIndicator(arg, newArg);
    }
  }

  TransformResult blockResult;
  for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
    auto opResult = transformOp(&op, ownedMemrefs);
    if (failed(opResult)) return failure();
    // Remove released memrefs.
    for (auto v : opResult->released) {
      auto owned = llvm::find(ownedMemrefs, v);
      // If we don't own the released value, pass the release on to the parent.
      if (owned == ownedMemrefs.end()) {
        if (!blockResult.released.insert(v).second) {
          block.getParentOp()->emitOpError("same value released twice");
          return failure();
        }
      } else {
        ownedMemrefs.erase(owned);
      }
    }
    ownedMemrefs.insert(opResult->acquired.begin(), opResult->acquired.end());
  }
  auto yieldedMemrefs = llvm::to_vector(
      llvm::make_filter_range(block.getTerminator()->getOperands(), isMemref));

  // Handle owned memrefs that don't alias with any yielded memref first.
  for (auto v : ownedMemrefs) {
    if (!llvm::any_of(yieldedMemrefs, [&](Value yielded) {
          return aliasOverapprox.isEquivalent(yielded, v);
        })) {
      // This owned memref does not escape, so we can put it in its own
      // retain and place it as early as possible.
      auto* insertionPoint = block.getTerminator();
      while (insertionPoint->getPrevNode() &&
             !doesAlias(insertionPoint->getPrevNode(), v, aliasOverapprox)) {
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
      aliasOverapprox.insert(v);
      result[aliasOverapprox.getLeaderValue(v)].push_back(v);
    }
    return result;
  };
  auto yieldedByLeader = groupByLeader(yieldedMemrefs);
  auto ownedByLeader = groupByLeader(ownedMemrefs);

  // Create one retain per equivalence class.
  ImplicitLocOpBuilder b(loc, block.getTerminator());
  auto null = b.create<NullOp>();
  blockResult.acquired =
      SmallVector<Value>(yieldedMemrefs.size(), null.getResult());
  for (auto [leader, yielded] : yieldedByLeader) {
    auto& ownedGroup = ownedByLeader[leader];
    if (ownedGroup.size() == 1 && yielded.size() == 1) {
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
      aliasOverapprox.unionSets(retained, result);
      blockResult.acquired[llvm::find(yieldedMemrefs, result) -
                           yieldedMemrefs.begin()] = retained;
    }
  }
  if (!llvm::is_contained(blockResult.acquired, null.getResult())) null.erase();
  return blockResult;
}

FailureOr<breaks_if_you_move_ops::ValueSet>
Deallocator::transformIfImplicitCapture(scf::IfOp op, TransformResult& ifResult,
                                        TransformResult& elseResult) {
  if (ifResult.released == elseResult.released) {
    return ifResult.released;
  }

  auto fixAcquiredAlloc = [&](Value v, Region& region,
                              TransformResult& result) -> LogicalResult {
    if (region.empty()) {
      op.emitOpError("cannot implicitly capture from an if without else");
      return failure();
    }
    auto* terminator = region.front().getTerminator();
    auto operands = terminator->getOperands();
    auto it = llvm::find_if(operands, [&](Value operand) {
      return findOwnershipIndicator(operand) == v;
    });
    if (it == operands.end()) {
      op.emitOpError("released value not yielded on other branch");
      return failure();
    }
    ownershipIndicator.erase(v);

    auto index = std::count_if(operands.begin(), it, isMemref);
    result.acquired[index] = v;
    return success();
  };

  for (auto v : ifResult.released) {
    if (!llvm::is_contained(elseResult.released, v)) {
      if (failed(fixAcquiredAlloc(v, op.getElseRegion(), elseResult)))
        return failure();
    }
  }
  for (auto v : elseResult.released) {
    if (!llvm::is_contained(ifResult.released, v)) {
      if (failed(fixAcquiredAlloc(v, op.getThenRegion(), ifResult)))
        return failure();
    }
  }

  breaks_if_you_move_ops::ValueSet released = ifResult.released;
  released.insert(elseResult.released.begin(), elseResult.released.end());
  return released;
}

FailureOr<TransformResult> Deallocator::transformOp(
    RegionBranchOpInterface op,
    const breaks_if_you_move_ops::ValueSet& ownedMemrefs) {
  SmallVector<int64_t> originalNumArgsByRegion;
  SmallVector<TransformResult> transformResultsByRegion;
  transformResultsByRegion.reserve(op->getNumRegions());

  bool mayImplicitlyCapture = llvm::isa<scf::IfOp>(op);
  for (auto [index, region] : llvm::enumerate(op->getRegions())) {
    assert(region.getBlocks().size() <= 1 &&
           "expected regions to have at most one block");
    auto edges = getSuccessorRegions(op, index);
    originalNumArgsByRegion.push_back(region.getNumArguments());

    auto& result = transformResultsByRegion.emplace_back();
    if (region.empty()) continue;

    // Transform the block and collect acquired/released memrefs.
    auto transformResultOrError = transformBlock(region.front());
    if (failed(transformResultOrError)) return failure();

    result = *std::move(transformResultOrError);  // NOLINT
    if (!result.released.empty() && !mayImplicitlyCapture) {
      // This error means that there's a realloc or free in a loop, and the op
      // defining the value is outside the loop. This is not valid. To fix
      // this, turn the argument of realloc/free into an iter arg.
      op.emitOpError(
          "can't implicitly capture across loop boundaries; use an "
          "explicit iter arg instead");
      return failure();
    }
  }

  breaks_if_you_move_ops::ValueSet released;
  if (llvm::any_of(transformResultsByRegion, [](const TransformResult& result) {
        return !result.released.empty();
      })) {
    auto releasedByIf = transformIfImplicitCapture(
        llvm::cast<scf::IfOp>(op.getOperation()), transformResultsByRegion[0],
        transformResultsByRegion[1]);
    if (failed(releasedByIf)) return failure();
    released = *std::move(releasedByIf);  // NOLINT
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
  // If we pass an owned memref to the loop and don't reuse it afterwards, we
  // can transfer ownership.
  for (auto operand : llvm::make_filter_range(operands, isMemref)) {
    auto isLastUse = [&]() {
      for (auto* candidate = op.getOperation(); candidate != nullptr;
           candidate = candidate->getNextNode()) {
        if (doesAlias(candidate, operand, aliasOverapprox,
                      /*considerOperands=*/candidate != op.getOperation())) {
          return false;
        }
      }
      return true;
    };

    Value ownershipIndicator = findOwnershipIndicator(operand);
    if (ownershipIndicator &&
        !llvm::is_contained(released, ownershipIndicator) &&
        llvm::is_contained(ownedMemrefs, ownershipIndicator) && isLastUse()) {
      // This is an alloc that is not used again, so we can pass ownership
      // to the loop.
      op->insertOperands(op->getNumOperands(), ownershipIndicator);
      released.insert(ownershipIndicator);
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
        aliasOverapprox.unionSets(pred, succ);
      }
    }
  };
  auto setMemrefAliases = [this](ValueRange a, ValueRange b) {
    for (auto [aa, bb] : llvm::zip(llvm::make_filter_range(a, isMemref), b)) {
      aliasOverapprox.unionSets(aa, bb);
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

// TODO(frgossen): Also allow passing ownership to functions.
FailureOr<TransformResult> Deallocator::transformOp(func::CallOp op) {
  ImplicitLocOpBuilder b(op.getLoc(), op);

  // Extend result types with ownership indicators.
  SmallVector<Type> newResultTys(op.getResultTypes());
  int64_t numMemrefResults = llvm::count_if(op.getResults(), isMemref);
  newResultTys.append(
      SmallVector<Type>(numMemrefResults, b.getType<OwnershipIndicatorType>()));
  auto newOp = b.create<func::CallOp>(op.getCalleeAttr(), newResultTys,
                                      op.getOperands());

  // Follow the call graph and process the callee first to get accurate aliasing
  // information.
  auto callee = llvm::cast<func::FuncOp>(
      op->getParentOfType<ModuleOp>().lookupSymbol(op.getCallee()));
  if (failed(transformFuncOp(callee))) return failure();

  // Update ownership indicators and aliasing.
  int64_t numResults = op.getNumResults();
  int64_t ownershipIndicatorIdx = numResults;
  for (auto [result, resultAliasing] :
       llvm::zip(newOp.getResults().take_front(numResults),
                 functionAliasOverapprox[callee])) {
    if (!isMemref(result)) continue;
    setOwnershipIndicator(result, newOp.getResult(ownershipIndicatorIdx++));
    for (int64_t i : resultAliasing) {
      aliasOverapprox.unionSets(result, op.getOperand(i));
    }
  }

  // Replace old op.
  op.replaceAllUsesWith(newOp.getResults().take_front(numResults));
  op.erase();

  // Collect ownership indicators.
  auto retained = newOp->getResults().drop_front(numResults);
  return TransformResult{{}, retained};
}

// Returns the set of values that are potentially owned by the op.
FailureOr<TransformResult> Deallocator::transformOp(
    Operation* op, const breaks_if_you_move_ops::ValueSet& ownedMemrefs) {
  if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(op)) {
    return transformOp(rbi, ownedMemrefs);
  }
  if (auto callOp = llvm::dyn_cast<func::CallOp>(op)) {
    return transformOp(callOp);
  }

  if (auto me = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
    TransformResult result;
    OpBuilder b(op->getContext());
    b.setInsertionPointAfter(op);

    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> allocs,
        frees;
    me.getEffects<MemoryEffects::Allocate>(allocs);
    me.getEffects<MemoryEffects::Free>(frees);
    if (!allocs.empty() || !frees.empty()) {
      for (const auto& alloc : allocs) {
        auto owned = b.create<OwnOp>(op->getLoc(), alloc.getValue());
        setOwnershipIndicator(alloc.getValue(), owned);
        result.acquired.push_back(owned);
      }
      for (const auto& free : frees) {
        auto ownershipIndicator = findOwnershipIndicator(free.getValue());
        if (!ownershipIndicator) {
          op->emitOpError("unable to find ownership indicator for operand");
          return failure();
        }
        result.released.insert(ownershipIndicator);
      }
      return result;
    }
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
        aliasOverapprox.unionSets(result, arg);
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
    ModuleOp moduleOp = getOperation();
    if (failed(Deallocator().transformModuleOp(moduleOp))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDeallocatePass() {
  return std::make_unique<DeallocatePass>();
}

}  // namespace deallocation
}  // namespace mlir
