//===- MemRefDataFlowOpt.cpp - MemRef DataFlow Optimization pass ------ -*-===//
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
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO(mlir-team): In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'for'/'if' statements is available.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

#define DEBUG_TYPE "memref-dataflow-opt"

using namespace mlir;

namespace {

// The store to load forwarding relies on three conditions:
//
// 1) there has to be a dependence from the store to the load satisfied at the
// block immediately within the innermost common surrounding loop of the load op
// and the store op, and such a dependence should associate with a single  load
// location for a given source store iteration.
//
// 2) the store op should dominate the load op,
//
// 3) among all candidate store op's that satisfy (1) and (2), if there exists a
// store op that postdominates all those that satisfy (1), such a store op is
// provably the last writer to the particular memref location being loaded from
// by the load op, and its store value can be forwarded to the load.
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - condition (1) and (3) are precise and necessary, while
// condition (2) is a sufficient one but not necessary (since it doesn't reason
// about loops that are guaranteed to execute at least one).
//
// TODO(mlir-team): more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO(mlir-team): do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct MemRefDataFlowOpt : public FunctionPass, InstWalker<MemRefDataFlowOpt> {
  explicit MemRefDataFlowOpt() : FunctionPass(&MemRefDataFlowOpt::passID) {}

  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(Function *f) override { return success(); }
  PassResult runOnMLFunction(Function *f) override;

  void visitOperationInst(OperationInst *opInst);

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value *, 4> memrefsToErase;

  static char passID;
};

} // end anonymous namespace

char MemRefDataFlowOpt::passID = 0;

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
FunctionPass *mlir::createMemRefDataFlowOptPass() {
  return new MemRefDataFlowOpt();
}

// This is a straightforward implementation not optimized for speed. Optimize
// this in the future if needed.
void MemRefDataFlowOpt::visitOperationInst(OperationInst *opInst) {
  OperationInst *lastWriteStoreOp = nullptr;

  auto loadOp = opInst->dyn_cast<LoadOp>();
  if (!loadOp)
    return;

  OperationInst *loadOpInst = opInst;

  // First pass over the use list to get minimum number of surrounding
  // loops common between the load op and the store op, with min taken across
  // all store ops.
  SmallVector<OperationInst *, 8> storeOps;
  unsigned minSurroundingLoops = getNestingDepth(*loadOpInst);
  for (InstOperand &use : loadOp->getMemRef()->getUses()) {
    auto storeOp = cast<OperationInst>(use.getOwner())->dyn_cast<StoreOp>();
    if (!storeOp)
      continue;
    auto *storeOpInst = storeOp->getInstruction();
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOpInst, *storeOpInst);
    minSurroundingLoops = std::min(nsLoops, minSurroundingLoops);
    storeOps.push_back(storeOpInst);
  }

  // 1. Check if there is a dependence satisfied at depth equal to the depth
  // of the loop body of the innermost common surrounding loop of the storeOp
  // and loadOp.
  // The list of store op candidates for forwarding - need to satisfy the
  // conditions listed at the top.
  SmallVector<OperationInst *, 8> fwdingCandidates;
  // Store ops that have a dependence into the load (even if they aren't
  // forwarding candidates). Each fwding candidate will be checked for a
  // post-dominance on these. 'fwdingCandidates' are a subset of depSrcStores.
  SmallVector<OperationInst *, 8> depSrcStores;
  for (auto *storeOpInst : storeOps) {
    MemRefAccess srcAccess, destAccess;
    getMemRefAccess(storeOpInst, &srcAccess);
    getMemRefAccess(loadOpInst, &destAccess);
    FlatAffineConstraints dependenceConstraints;
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOpInst, *storeOpInst);
    // Dependences at loop depth <= minSurroundingLoops do NOT matter.
    for (unsigned d = nsLoops + 1; d > minSurroundingLoops; d--) {
      if (!checkMemrefAccessDependence(srcAccess, destAccess, d,
                                       &dependenceConstraints,
                                       /*dependenceComponents=*/nullptr))
        continue;
      depSrcStores.push_back(storeOpInst);
      // Check if this store is a candidate for forwarding; we only forward if
      // the dependence from the store is carried by the *body* of innermost
      // common surrounding loop. As an example this filters out cases like:
      // for %i0
      //   for %i1
      //     %idx = affine_apply (d0) -> (d0 + 1) (%i0)
      //     store %A[%idx]
      //     load %A[%i0]
      //
      if (d != nsLoops + 1)
        break;

      // 2. The store has to dominate the load op to be candidate. This is not
      // strictly a necessary condition since dominance isn't a prerequisite for
      // a memref element store to reach a load, but this is sufficient and
      // reasonably powerful in practice.
      if (!dominates(*storeOpInst, *loadOpInst))
        break;

      // Finally, forwarding is only possible if the load touches a single
      // location in the memref across the enclosing loops *not* common with the
      // store. This is filtering out cases like:
      // for (i ...)
      //   a [i] = ...
      //   for (j ...)
      //      ... = a[j]
      MemRefRegion region;
      getMemRefRegion(loadOpInst, nsLoops, &region);
      if (!region.getConstraints()->isRangeOneToOne(
              /*start=*/0, /*limit=*/loadOp->getMemRefType().getRank()))
        break;

      // After all these conditions, we have a candidate for forwarding!
      fwdingCandidates.push_back(storeOpInst);
      break;
    }
  }

  // Note: this can implemented in a cleaner way with postdominator tree
  // traversals. Consider this for the future if needed.
  for (auto *storeOpInst : fwdingCandidates) {
    // 3. Of all the store op's that meet the above criteria, the store
    // that postdominates all 'depSrcStores' (if such a store exists) is the
    // unique store providing the value to the load, i.e., provably the last
    // writer to that memref loc.
    if (llvm::all_of(depSrcStores, [&](OperationInst *depStore) {
          return postDominates(*storeOpInst, *depStore);
        })) {
      lastWriteStoreOp = storeOpInst;
      break;
    }
  }
  // TODO: optimization for future: those store op's that are determined to be
  // postdominated above can actually be recorded and skipped on the 'i' loop
  // iteration above --- since they can never post dominate everything.

  if (!lastWriteStoreOp)
    return;

  // Perform the actual store to load forwarding.
  Value *storeVal = lastWriteStoreOp->cast<StoreOp>()->getValueToStore();
  loadOp->getResult()->replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp->getMemRef());
  loadOp->erase();
}

PassResult MemRefDataFlowOpt::runOnMLFunction(Function *f) {
  memrefsToErase.clear();

  // Walk all load's and perform load/store forwarding.
  walk(f);

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canononicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto *memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    OperationInst *defInst = memref->getDefiningInst();
    if (!defInst || !cast<OperationInst>(defInst)->isa<AllocOp>())
      // TODO(mlir-team): if the memref was returned by a 'call' instruction, we
      // could still erase it if the call has no side-effects.
      continue;
    if (std::any_of(memref->use_begin(), memref->use_end(),
                    [&](InstOperand &use) {
                      auto *ownerInst = cast<OperationInst>(use.getOwner());
                      return (!ownerInst->isa<StoreOp>() &&
                              !ownerInst->isa<DeallocOp>());
                    }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto it = memref->use_begin(), e = memref->use_end(); it != e;) {
      auto &use = *(it++);
      cast<OperationInst>(use.getOwner())->erase();
    }
    defInst->erase();
  }

  // This function never leaves the IR in an invalid state.
  return success();
}

static PassRegistration<MemRefDataFlowOpt>
    pass("memref-dataflow-opt", "Perform store/load forwarding for memrefs");
