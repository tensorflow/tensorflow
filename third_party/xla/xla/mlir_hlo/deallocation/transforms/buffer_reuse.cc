/* Copyright 2023 The OpenXLA Authors.

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

#include <functional>
#include <memory>
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace deallocation {
namespace {

SmallVector<Value> hoistAllocs(Operation* parent, Region& region,
                               SmallVector<Value> freeAllocs) {
  if (region.empty()) return freeAllocs;
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
        alloc->replaceAllUsesWith(ValueRange{*reusable});
        dealloc->erase();
        op = op->getNextNode();
        alloc->erase();
        result.push_back(*reusable);
        *reusable = {};
      }
    } else {
      op = op->getNextNode();
    }
  }
  // Return remaining free allocs.
  for (auto reusable : freeAllocs) {
    if (reusable) result.push_back(reusable);
  }
  return result;
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

    if (auto rbi = llvm::dyn_cast<RegionBranchOpInterface>(op)) {
      SmallVector<Value> hoistedAllocs;
      for (auto& region : op->getRegions()) {
        hoistedAllocs = hoistAllocs(rbi, region, std::move(hoistedAllocs));
      }
      result |= !hoistedAllocs.empty();
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

// Recursively checks if `pred` holds for any op in [start; end)
bool anyMatchBetween(Operation* start, Operation* end,
                     const std::function<bool(Operation*)>& predicate) {
  while (start != end) {
    if (predicate(start)) return true;
    for (auto& region : start->getRegions()) {
      if (!region.empty() &&
          anyMatchBetween(&region.front().front(), nullptr, predicate)) {
        return true;
      }
    }
    start = start->getNextNode();
  }
  return false;
}

// Recursively checks if `v` is used in [start; end)
bool hasUsesBetween(Operation* start, Operation* end, Value v) {
  return anyMatchBetween(start, end, [v](Operation* op) {
    return llvm::is_contained(op->getOperands(), v);
  });
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
  SmallVector<memref::DeallocOp> deallocsToFix;
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    auto [allocations, deallocations] =
        findAllocsAndDeallocs(op->getRegion(i).front());

    // If we have an argument that's deallocated, and a matching allocation
    // that's yielded, we can instead stash the deallocated buffer in an arg and
    // use it the next time.
    for (auto [type, allocs] : allocations) {
      for (auto [alloc, dealloc] : llvm::zip(allocs, deallocations[type])) {
        doubleBuffer(op, alloc, dealloc);
        deallocsToFix.push_back(dealloc);

        if (llvm::isa<scf::WhileOp>(op)) {
          auto& otherRegion = op->getRegion(1 - i);
          // Forward the double buffered alloc.
          Value arg = otherRegion.addArgument(alloc.getType(), op.getLoc());
          otherRegion.front().getTerminator()->insertOperands(
              otherRegion.front().getTerminator()->getNumOperands(), arg);
        }
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

    if (llvm::isa<scf::ForOp, scf::WhileOp>(op)) {
      if (auto db = doubleBuffer(cast<RegionBranchOpInterface>(op)); db != op) {
        op = db;
        result = true;
      }
    }

    op = op->getNextNode();
  }
  return result;
}

bool isRestrictBbArg(Value value) {
  auto bbarg = llvm::dyn_cast<BlockArgument>(value);
  auto func =
      llvm::dyn_cast<func::FuncOp>(value.getParentBlock()->getParentOp());
  if (!bbarg || !func) return false;
  auto isRestrict = func.getArgAttrOfType<BoolAttr>(bbarg.getArgNumber(),
                                                    "deallocation.restrict");
  return isRestrict && isRestrict.getValue();
}

void eliminateCopies(Block& block, Block& root) {
  auto* op = &block.front();
  while (op) {
    for (auto& region : op->getRegions()) {
      if (!region.empty()) {
        assert(region.hasOneBlock());
        eliminateCopies(region.front(), root);
      }
    }

    auto copy = llvm::dyn_cast_or_null<memref::CopyOp>(op);
    op = op->getNextNode();

    auto dealloc = llvm::dyn_cast_or_null<memref::DeallocOp>(op);
    if (!copy || !dealloc ||
        copy.getTarget().getType() != copy.getSource().getType() ||
        dealloc.getMemref() != copy.getSource()) {
      continue;
    }

    auto sourceAlloc = llvm::dyn_cast_or_null<memref::AllocOp>(
        copy.getSource().getDefiningOp());
    if (!sourceAlloc) continue;

    bool targetIsFirstUseOfRestrictBbArg =
        isRestrictBbArg(copy.getTarget()) &&
        !hasUsesBetween(&root.front(), copy, copy.getTarget());
    if (!targetIsFirstUseOfRestrictBbArg) {
      auto targetAlloc = llvm::dyn_cast_or_null<memref::AllocOp>(
          copy.getTarget().getDefiningOp());
      bool targetIsFirstUseOfAlloc =
          targetAlloc && !hasUsesBetween(targetAlloc, copy, targetAlloc);

      // If the source was used before the definition of the target, or the
      // target was used before the copy, this transformation is unsafe.
      if (!targetIsFirstUseOfAlloc ||
          hasUsesBetween(&root.front(), targetAlloc, sourceAlloc)) {
        continue;
      }
    }

    // (no use of %b)
    // %a = alloc or %a is a bbarg with `restrict`.
    // %b = alloc
    // (no use of %a)
    // copy %b, %a
    // dealloc %b
    copy.getSource().replaceAllUsesWith(copy.getTarget());
    op = dealloc->getNextNode();
    copy->erase();
    dealloc->erase();
    sourceAlloc->erase();
  }
}

enum class BufferReuseMode {
  // Only reuse buffers if between a `dealloc` and `alloc` there are no further
  // `alloc`s that might later become a candidate for buffer reuse.
  CONSERVATIVE,
  // Also reuse buffers if there are intermediate ops between `dealloc` and
  // `alloc`. This may extend live-ranges of buffers (e.g. if the intermediate
  // op contains a region), which may destroy reuse opportunities.
  AGGRESSIVE
};

bool reuseBuffers(Block& block, BufferReuseMode mode) {
  auto* op = &block.front();
  bool result = false;
  while (op) {
    if (auto dealloc = llvm::dyn_cast<memref::DeallocOp>(op)) {
      memref::AllocOp alloc = findOp<memref::AllocOp>(
          op->getNextNode(), [](memref::AllocOp) { return true; });

      // In conservative mode, don't reuse buffers if there is a candidate alloc
      // in between the dealloc/alloc pair that might still be matched with this
      // dealloc. If we extend the live-range of the buffer past this alloc,
      // this will prevent further reuse.
      if (!alloc || (mode == BufferReuseMode::CONSERVATIVE &&
                     anyMatchBetween(dealloc, alloc, [&](Operation* op) {
                       return llvm::isa<memref::AllocOp>(op) &&
                              op->getResultTypes().front() == alloc.getType();
                     }))) {
        op = op->getNextNode();
        continue;
      }

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

    for (auto& region : op->getRegions()) {
      if (!region.empty()) {
        assert(region.hasOneBlock());
        result |= reuseBuffers(region.front(), mode);
      }
    }

    op = op->getNextNode();
  }
  return result;
}

void promoteToStack(memref::DeallocOp dealloc) {
  auto alloc = dealloc.getMemref().getDefiningOp<memref::AllocOp>();
  OpBuilder b(alloc);
  auto alloca = b.create<memref::AllocaOp>(
      alloc->getLoc(), mlir::cast<MemRefType>(alloc->getResultTypes()[0]),
      alloc.getAlignmentAttr());
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
    auto getAliases = [&](RegionBranchPoint point) {
      for (const auto& edge : getSuccessorRegions(rbi, point)) {
        for (auto [pred, succ] : llvm::zip(edge.getPredecessorOperands(),
                                           edge.getSuccessorValues())) {
          eq.unionSets(pred, succ);
        }
      }
    };

    getAliases(RegionBranchPoint::parent());
    for (auto& region : rbi->getRegions()) {
      getAliases(region);
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
  // TODO(jreiffers): Use byte sizes instead.
  int64_t remainingAllowedStackUse = 1 << 12;
  for (auto* op = &block.front(); op;) {
    auto alloc = llvm::dyn_cast<memref::AllocOp>(op);
    op = op->getNextNode();

    if (alloc) {
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
          int64_t numElements = alloc.getMemref().getType().getNumElements();
          if (remainingAllowedStackUse >= numElements) {
            remainingAllowedStackUse -= numElements;
            promoteToStack(llvm::cast<memref::DeallocOp>(*dealloc));
          }
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
    // Copy elimination requires small live-ranges to work well. We only extend
    // live ranges afterwards, so running it more than once doesn't help.
    eliminateCopies(block, /*root=*/block);
    do {
      // Eliminate dead code.
      (void)applyPatternsAndFoldGreedily(getOperation(), {});
      // Only coalesce dealloc/alloc pairs that are immediate neighbors, to
      // make sure we don't accidentally extend the live range of a buffer.
      result = reuseBuffers(block, BufferReuseMode::CONSERVATIVE);
      // Make sure we rerun buffer reuse after every intermediate step.
      result |= hoistAllocs(block) || doubleBuffer(block) ||
                simplifyLoopDeallocs(block);
    } while (result);
    // Now we can also coalesce distant dealloc/alloc pairs.
    reuseBuffers(block, BufferReuseMode::AGGRESSIVE);
    promoteBuffers(block);
    (void)applyPatternsAndFoldGreedily(getOperation(), {});
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}

}  // namespace deallocation
}  // namespace mlir
