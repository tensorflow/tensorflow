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

#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "deallocation/transforms/passes.h"
#include "deallocation/utils/util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace deallocation {
namespace {

// Finds the start of a memref use chain (e.g. subview of cast of alloc). Stops
// at block arguments and allocs. This assumes that hlo-deallocate was run
// previously and the invariants that it guarantees still hold: each ranked
// memref block argument should have a corresponding unranked ownership
// indicator argument.
Value rootAlloc(Value v) {
  if (auto bbarg = llvm::dyn_cast<BlockArgument>(v)) {
    if (v.getType().isa<UnrankedMemRefType>()) {
      return v;
    }

    auto memrefArgs = llvm::to_vector(llvm::make_filter_range(
        bbarg.getParentBlock()->getArguments(),
        [](BlockArgument arg) { return arg.getType().isa<MemRefType>(); }));
    auto unrankedMemrefArgs = llvm::to_vector(llvm::make_filter_range(
        bbarg.getParentBlock()->getArguments(), [](BlockArgument arg) {
          return arg.getType().isa<UnrankedMemRefType>();
        }));

    // Find the ownership indicator for the block argument.
    for (auto [memref, alloc] : llvm::zip(memrefArgs, unrankedMemrefArgs)) {
      if (memref == bbarg) {
        return alloc;
      }
    }

    // There may not be an ownership indicator, for example if this is a
    // function block argument.
    return v;
  }

  if (llvm::isa_and_present<memref::SubViewOp, memref::CastOp,
                            memref::ExpandShapeOp, memref::CollapseShapeOp,
                            memref::ReshapeOp, memref::ViewOp,
                            memref::ReinterpretCastOp, memref::TransposeOp>(
          v.getDefiningOp())) {
    return rootAlloc(v.getDefiningOp()->getOperand(0));
  }
  return v;
}

// Eliminates redundant owernship arguments for memrefs that are always owned
// by the block. This helps with hoisting and reuse.
void elideRedundantOwnershipArgs(RegionBranchOpInterface op) {
  bool isFor = llvm::isa<scf::ForOp>(op);
  if (!llvm::isa<scf::WhileOp>(op) && !isFor) {
    return;
  }

  SmallVector<size_t> resultIndices(op->getNumResults());
  std::iota(resultIndices.begin(), resultIndices.end(), 0);

  // Get the root allocs of all operands and arguments while the invariants are
  // still intact.
  DenseMap<Value, Value> rootAllocs;
  for (auto operand : op->getOperands()) {
    rootAllocs[operand] = rootAlloc(operand);
  }
  for (auto& region : op->getRegions()) {
    for (auto arg : region.getArguments()) {
      rootAllocs[arg] = rootAlloc(arg);
    }
    for (auto operand :
         region.getBlocks().front().getTerminator()->getOperands()) {
      rootAllocs[operand] = rootAlloc(operand);
    }
  }

  for (auto& region : op->getRegions()) {
    llvm::SmallVector<size_t> memrefArgIndices;
    llvm::SmallVector<size_t> ownershipArgIndices;

    llvm::SmallVector<Value> memrefArgs;
    llvm::SmallVector<Value> ownershipArgs;
    for (auto [index, arg] : llvm::enumerate(region.getArguments())) {
      if (arg.getType().isa<UnrankedMemRefType>()) {
        ownershipArgs.push_back(arg);
        ownershipArgIndices.push_back(index);
      } else if (arg.getType().isa<MemRefType>()) {
        memrefArgs.push_back(arg);
        memrefArgIndices.push_back(index);
      }
    }

    // Only proceed if this region has the standard form after
    // buffer-deallocation.
    if (memrefArgs.size() != ownershipArgs.size()) continue;

    llvm::SmallBitVector argsToDrop(ownershipArgs.size(), true);
    auto predecessors = getPredecessorRegions(op, region.getRegionNumber());
    for (const auto& pred : predecessors) {
      for (unsigned i = 0; i < argsToDrop.size(); ++i) {
        Value argRoot =
            rootAllocs[pred.getPredecessorOperand(memrefArgIndices[i])];
        Value ownerRoot =
            rootAllocs[pred.getPredecessorOperand(ownershipArgIndices[i])];
        bool same = argRoot && argRoot == ownerRoot;
        argsToDrop[i] = argsToDrop[i] && same;
      }
    }

    for (int64_t i = static_cast<int64_t>(argsToDrop.size()) - 1; i >= 0; --i) {
      if (!argsToDrop[i]) continue;

      for (auto& pred : predecessors) {
        if ((pred.predecessorOp == op && isFor) ||
            (pred.predecessorRegionIndex == 0 && !isFor)) {
          resultIndices[ownershipArgIndices[i] - pred.successorValueIndex] =
              memrefArgIndices[i] - pred.successorValueIndex;
        }
        pred.predecessorOp->eraseOperands(pred.predecessorOperandIndex +
                                          ownershipArgIndices[i] -
                                          pred.successorValueIndex);
      }
      // Cast to the right type.
      OpBuilder b(&region.getBlocks().front().front());
      region.getArgument(ownershipArgIndices[i])
          .replaceAllUsesWith(b.create<memref::CastOp>(
              region.getLoc(), ownershipArgs[i].getType(), memrefArgs[i]));
      region.eraseArgument(ownershipArgIndices[i]);
    }
  }

  auto newOp = moveRegionsToNewOpButKeepOldOp(op);
  SmallVector<Value> results;
  OpBuilder b(op.getContext());
  b.setInsertionPointAfter(newOp);
  for (auto [oldIndex, newIndex] : llvm::enumerate(resultIndices)) {
    if (oldIndex == newIndex) {
      results.push_back(newOp->getResult(newIndex));
    } else {
      results.push_back(b.create<memref::CastOp>(newOp.getLoc(),
                                                 op->getResultTypes()[oldIndex],
                                                 newOp->getResult(newIndex)));
    }
  }

  op->replaceAllUsesWith(results);
  op->erase();
}

void elideRedundantOwnershipArgs(Block& block) {
  block.walk(
      [](RegionBranchOpInterface rbi) { elideRedundantOwnershipArgs(rbi); });
  block.walk([](memref::DeallocOp dealloc) {
    dealloc.setOperand(rootAlloc(dealloc.getMemref()));
  });
}

SmallVector<Value> hoistAllocs(Operation* parent, Region& region,
                               SmallVector<Value> freeAllocs) {
  assert(region.hasOneBlock() && "expected the region to have a single block");
  // Hoist local allocs out of the loop.
  // TODO(jreiffers): Add some smarts here so we don't blow up the heap for
  // pathological inputs.

  SmallVector<Value> result;
  auto* op = &region.front().front();
  while (op) {
    auto alloc = llvm::dyn_cast<memref::AllocOp>(op);
    if (alloc && alloc.getDynamicSizes().empty()) {
      auto dealloc = llvm::find_if(op->getUsers(), [&](Operation* user) {
        return llvm::isa<memref::DeallocOp>(user) &&
               user->getParentRegion() == &region;
      });
      if (dealloc == op->getUsers().end()) {
        op = op->getNextNode();
        continue;
      }

      auto* reusable = llvm::find_if(freeAllocs, [&](Value free) {
        return free && free.getType() == alloc.getType();
      });
      if (reusable == freeAllocs.end()) {
        dealloc->moveAfter(parent);
        op = op->getNextNode();
        alloc->moveBefore(parent);
        result.push_back(alloc);
      } else {
        op->replaceAllUsesWith(ValueRange{*reusable});
        dealloc->erase();
        op = op->getNextNode();
        alloc->erase();
        *reusable = {};
      }
    } else {
      op = op->getNextNode();
    }
  }
  return result;
}

bool hoistAllocs(scf::WhileOp op) {
  auto beforeAllocs = hoistAllocs(op, op.getBefore(), {});
  auto afterAllocs = hoistAllocs(op, op.getAfter(), beforeAllocs);
  return !beforeAllocs.empty() || !afterAllocs.empty();
}

// Hoists allocs from while and for loops.
bool hoistAllocs(Block& block) {
  auto* op = &block.front();
  bool result = false;
  while (op) {
    for (auto& region : op->getRegions()) {
      if (!region.empty()) {
        assert(region.hasOneBlock());
        result |= hoistAllocs(region.front());
      }
    }

    if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
      result |= hoistAllocs(whileOp);
    }

    if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      result |= !hoistAllocs(forOp, forOp.getLoopBody(), {}).empty();
    }
    op = op->getNextNode();
  }
  return result;
}

template <typename T>
T findOp(Operation* start, std::function<bool(T)> predicate) {
  while (start) {
    if (T t = llvm::dyn_cast<T>(start)) {
      if (predicate(t)) return t;
    }
    start = start->getNextNode();
  }
  return {};
}

// Checks if v is used in [start; end)
bool hasUsesBetween(Operation* start, Operation* end, Value v) {
  while (start != end) {
    if (llvm::is_contained(start->getOperands(), v)) return true;
    for (auto& region : start->getRegions()) {
      if (hasUsesBetween(&region.front().front(), nullptr, v)) return true;
    }
    start = start->getNextNode();
  }
  return false;
}

std::pair<DenseMap<Type, SmallVector<memref::AllocOp>>,
          DenseMap<Type, SmallVector<memref::DeallocOp>>>
findAllocsAndDeallocs(Block& block) {
  DenseMap<Type, SmallVector<memref::AllocOp>> allocations;
  DenseMap<Type, SmallVector<memref::DeallocOp>> deallocations;

  for (auto& op : block) {
    if (auto alloc = llvm::dyn_cast<memref::AllocOp>(op)) {
      if (alloc.getDynamicSizes().empty()) {
        allocations[alloc.getResult().getType()].push_back(alloc);
      }
    } else if (auto dealloc = llvm::dyn_cast<memref::DeallocOp>(op)) {
      deallocations[dealloc.getMemref().getType()].push_back(dealloc);
    }
  }

  return {allocations, deallocations};
}

void doubleBuffer(Operation* op, memref::AllocOp alloc,
                  memref::DeallocOp dealloc) {
  assert(alloc->getParentRegion() == dealloc->getParentRegion());
  auto* region = alloc->getParentRegion();
  assert(region->hasOneBlock() && "expected the region to have a single block");

  // 1. Introduce a new bbarg to replace the alloc.
  Value arg = region->front().addArgument(alloc.getType(), alloc.getLoc());
  alloc.replaceAllUsesWith(arg);

  // 2. Move the alloc before the loop and add it to its operands.
  alloc->moveBefore(op);
  op->insertOperands(op->getNumOperands(), alloc.getResult());

  // 3. Yield the dealloc.
  region->front().getTerminator()->insertOperands(
      region->front().getTerminator()->getNumOperands(), dealloc.getMemref());

  // 4. Move the dealloc after the loop.
  dealloc->moveAfter(op);
}

RegionBranchOpInterface doubleBuffer(RegionBranchOpInterface op) {
  auto [allocations, deallocations] =
      findAllocsAndDeallocs(op->getRegion(op->getNumRegions() - 1).front());

  // If we have an argument that's deallocated, and a matching allocation that's
  // yielded, we can instead stash the deallocated buffer in an arg and use it
  // the next time.
  SmallVector<memref::DeallocOp> deallocsToFix;
  for (auto [type, allocs] : allocations) {
    for (auto [alloc, dealloc] : llvm::zip(allocs, deallocations[type])) {
      doubleBuffer(op, alloc, dealloc);
      deallocsToFix.push_back(dealloc);

      if (llvm::isa<scf::WhileOp>(op)) {
        auto& before = op->getRegion(0);
        // Forward the double buffered alloc from the before to the after
        // region.
        Value beforeArg = before.addArgument(alloc.getType(), op.getLoc());
        before.front().getTerminator()->insertOperands(
            before.front().getTerminator()->getNumOperands(), beforeArg);
      }
    }
  }

  if (deallocsToFix.empty()) return op;

  auto newOp = moveRegionsToNewOpButKeepOldOp(op);
  op->replaceAllUsesWith(newOp->getResults().take_front(op->getNumResults()));
  op.erase();

  for (auto [dealloc, result] :
       llvm::zip(deallocsToFix,
                 newOp->getResults().take_back(deallocsToFix.size()))) {
    dealloc.getMemrefMutable().assign(result);
  }
  return newOp;
}

bool doubleBuffer(Block& block) {
  auto* op = &block.front();
  bool result = false;
  while (op) {
    for (auto& region : op->getRegions()) {
      if (!region.empty()) {
        assert(region.hasOneBlock());
        result |= doubleBuffer(region.front());
      }
    }

    if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
      if (auto db = doubleBuffer(whileOp); db != op) {
        op = db;
        result = true;
      }
    }

    if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      if (auto db = doubleBuffer(forOp); db != op) {
        op = db;
        result = true;
      }
    }

    op = op->getNextNode();
  }
  return result;
}

bool reuseBuffers(Block& block) {
  auto* op = &block.front();
  bool result = false;
  while (op) {
    if (auto dealloc = llvm::dyn_cast<memref::DeallocOp>(op)) {
      // Try to find an alloc op with the same shape. Only check the next
      // alloc, so we don't increase the heap size in the meantime.
      // TODO(jreiffers): Can we be smarter here?
      auto alloc = findOp<memref::AllocOp>(
          op->getNextNode(), [](memref::AllocOp) { return true; });
      if (alloc && alloc.getDynamicSizes().empty() &&
          alloc->getResultTypes()[0] == dealloc.getMemref().getType()) {
        alloc.replaceAllUsesWith(dealloc.getMemref());
        op = alloc->getNextNode();
        dealloc.erase();
        alloc.erase();
        result = true;
        continue;
      }
    }

    if (auto copy = llvm::dyn_cast_or_null<memref::CopyOp>(op)) {
      auto alloc = llvm::dyn_cast_or_null<memref::AllocOp>(
          copy.getTarget().getDefiningOp());
      auto dealloc =
          llvm::dyn_cast_or_null<memref::DeallocOp>(copy->getNextNode());
      if (alloc && dealloc &&
          alloc.getType() == dealloc.getMemref().getType() &&
          !hasUsesBetween(/*start=*/alloc, /*end=*/copy, alloc)) {
        // %a = alloc
        // (some IR not using %a)
        // copy %b, %a
        // dealloc %b
        alloc.replaceAllUsesWith(dealloc.getMemref());
        op = dealloc->getNextNode();
        copy->erase();
        alloc->erase();
        dealloc->erase();
        result = true;
        continue;
      }
    }

    // Reuse may get rid of allocs entirely, so run it before attempting double
    // buffering.
    for (auto& region : op->getRegions()) {
      if (!region.empty()) {
        assert(region.hasOneBlock());
        result |= reuseBuffers(region.front());
      }
    }

    op = op->getNextNode();
  }
  return result;
}

void promoteToStack(memref::DeallocOp dealloc) {
  auto* alloc = dealloc.getMemref().getDefiningOp();
  OpBuilder b(alloc);
  auto alloca = b.create<memref::AllocaOp>(
      alloc->getLoc(), alloc->getResultTypes()[0].cast<MemRefType>());
  alloc->replaceAllUsesWith(ValueRange{alloca.getResult()});
  alloc->erase();
  dealloc->erase();
}

bool simplifyLoopDeallocs(Block& block) {
  // Tries to transform:
  //   %a:n = scf.while (...)
  //   dealloc %a#i
  //   dealloc %a#j

  // Into:
  //   %a:n = scf.while (...)
  //   dealloc %alloc_i
  //   dealloc %alloc_j

  // This enables more buffer promotion/reuse.
  bool result = false;
  for (auto& op : block.getOperations()) {
    auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(op);
    if (!rbi) continue;

    struct ResultInfo {
      memref::DeallocOp dealloc;
    };

    llvm::DenseSet<Value> operands{rbi->getOperands().begin(),
                                   rbi->getOperands().end()};
    llvm::DenseMap<Value, ResultInfo> results;

    for (auto result : rbi->getResults()) {
      if (result.use_empty()) {
        results[result] = {};
      } else if (result.hasOneUse()) {
        if (auto dealloc =
                llvm::dyn_cast<memref::DeallocOp>(*result.getUsers().begin())) {
          if (dealloc->getParentRegion() == block.getParent()) {
            results[result] = {dealloc};
          }
        }
      }
    }

    breaks_if_you_move_ops::ValueEquivalenceClasses eq;
    auto getAliases = [&](std::optional<unsigned> index) {
      for (const auto& edge : getSuccessorRegions(rbi, index)) {
        for (auto [pred, succ] : llvm::zip(edge.getPredecessorOperands(),
                                           edge.getSuccessorValues())) {
          eq.unionSets(pred, succ);
        }
      }
    };

    getAliases(std::nullopt);
    for (auto& region : rbi->getRegions()) {
      getAliases(region.getRegionNumber());
    }

    for (auto it = eq.begin(), e = eq.end(); it != e; ++it) {
      if (!it->isLeader()) continue;

      breaks_if_you_move_ops::ValueSet equivalentOperands;
      llvm::SmallVector<memref::DeallocOp> deallocs;
      bool failed = false;
      for (auto member = eq.member_begin(it);
           !failed && member != eq.member_end(); ++member) {
        if (operands.contains(*member)) {
          equivalentOperands.insert(*member);
        } else if (auto result = results.find(*member);
                   result != results.end()) {
          if (result->second.dealloc) {
            deallocs.push_back(result->second.dealloc);
          }
        } else if (auto bbarg = llvm::dyn_cast<BlockArgument>(*member)) {
          if (bbarg.getParentRegion()->getParentOp() != rbi) {
            failed = true;
          }
        } else {
          failed = true;
        }
      }

      if (equivalentOperands.size() == deallocs.size() && !failed) {
        // If all results are unused, we can just deallocate them in any
        // order:
        for (auto [dealloc, operand] :
             llvm::zip(deallocs, llvm::to_vector(equivalentOperands))) {
          dealloc.setOperand(operand);
          result = true;
        }
      }
    }
  }
  return result;
}

void promoteBuffers(Block& block) {
  for (auto* op = &block.front(); op;) {
    auto alloc = llvm::dyn_cast<memref::AllocOp>(op);
    op = op->getNextNode();

    if (alloc) {
      // TODO(jreiffers): Add size heuristic.
      if (!alloc.getMemref().getType().hasStaticShape()) continue;

      auto dealloc = llvm::find_if(alloc->getUsers(), [&](Operation* user) {
        return user->getBlock() == &block && llvm::isa<memref::DeallocOp>(user);
      });

      if (dealloc != alloc->getUsers().end()) {
        if (op == *dealloc) {
          op = op->getNextNode();
          dealloc->erase();
          alloc->erase();
        } else {
          promoteToStack(llvm::cast<memref::DeallocOp>(*dealloc));
        }
      }
    }
  }
}

#define GEN_PASS_DEF_BUFFERREUSEPASS
#include "deallocation/transforms/passes.h.inc"

struct BufferReusePass : public impl::BufferReusePassBase<BufferReusePass> {
  void runOnOperation() override {
    bool result;
    auto& block = getOperation().getBody().front();
    // This assumes invariants that it breaks, so it can only be run once.
    elideRedundantOwnershipArgs(block);
    do {
      // Eliminate dead code.
      (void)applyPatternsAndFoldGreedily(getOperation(), {});
      result = hoistAllocs(block);
      result |= reuseBuffers(block);
      result |= doubleBuffer(block);
      result |= simplifyLoopDeallocs(block);
    } while (result);
    promoteBuffers(block);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}

}  // namespace deallocation
}  // namespace mlir
