/* Copyright 2024 The OpenXLA Authors.

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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"
#include "xla/hlo/analysis/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using mlir::MLIRContext;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;
namespace mv = ::mlir::vector;

#define GEN_PASS_DEF_FUSELOOPSPASS
#include "xla/backends/gpu/codegen/transforms/passes.h.inc"

bool LoopsUseSameDimOps(LoopOp& loop1, LoopOp& loop2) {
  for (auto [dim1, dim2] : llvm::zip(loop1.getDims(), loop2.getDims())) {
    if (dim1.getDefiningOp() != dim2.getDefiningOp()) {
      return false;
    }
  }
  return true;
}

bool LoopsHaveTheSameDomain(LoopOp& loop1, LoopOp& loop2) {
  auto map1 = loop1.getIndexingMap();
  auto map2 = loop2.getIndexingMap();
  if (map1.GetDimVarsCount() != map2.GetDimVarsCount() ||
      map1.GetRangeVarsCount() != map2.GetRangeVarsCount() ||
      map1.GetConstraintsCount() != map2.GetConstraintsCount()) {
    return false;
  }
  for (auto [d1, d2] : llvm::zip(map1.GetDimVars(), map2.GetDimVars())) {
    if (d1 != d2) return false;
  }
  for (auto [r1, r2] : llvm::zip(map1.GetRangeVars(), map2.GetRangeVars())) {
    if (r1 != r2) return false;
  }
  if (map1.GetConstraints() != map2.GetConstraints()) return false;

  // Check dimensions come from the same op. This is technically not a
  // requirement and could be modified to handle different dim args.
  return LoopsUseSameDimOps(loop1, loop2);
}

// Check that the loops:
// 1. insert and extract from the same location within each iteration,
// 2. use all their IVs (so we don't overwrite the values in another iteration),
// 3. all indices are IVs (so they are confirmed injective).
bool IndicesAreEqualAndInjective(int64_t iv_count, mv::InsertOp insert,
                                 mv::ExtractOp extract) {
  auto insert_indices = insert.getDynamicPosition();
  auto extract_indices = extract.getDynamicPosition();
  if (insert_indices.size() != extract_indices.size()) {
    return false;
  }
  if (insert_indices.size() != iv_count) {
    return false;
  }

  SmallVector<bool> matched_indices(iv_count, false);
  for (auto [in, ex] : llvm::zip(insert_indices, extract_indices)) {
    auto in_arg = mlir::dyn_cast<mlir::BlockArgument>(in);
    auto ex_arg = mlir::dyn_cast<mlir::BlockArgument>(ex);
    if (!in_arg || !ex_arg || in_arg.getArgNumber() != ex_arg.getArgNumber()) {
      return false;
    }
    // Check #3 - all indices are IVs.
    if (in_arg.getArgNumber() >= iv_count) {
      return false;
    }
    matched_indices[in_arg.getArgNumber()] = true;
  }
  // If there is a loop IV that we didn't use in the insert op, then don't
  // match. It's possible that we overwrite the value on a subsequent iteration
  // so the loops cannot be fused.
  return llvm::all_of(matched_indices, [](bool matched) { return matched; });
}

bool LoopDominatesLoop(LoopOp dominator /*lastloop*/, LoopOp dominatee) {
  mlir::DominanceInfo dom;
  return llvm::all_of(dominatee.getResults(), [&](Value result) {
    return llvm::all_of(result.getUsers(), [&](Operation* user) {
      return dom.properlyDominates(dominator, user,
                                   /*enclosingOpOk*/ false);
    });
  });
}

// Fuse insert_loop and extract_loop into a single loop, and remove the
// vector.insert and vector.extract ops.
void FuseExtractInsertLoopPair(MLIRContext* mlir_context, LoopOp insert_loop,
                               LoopOp extract_loop, mv::InsertOp insert,
                               mv::ExtractOp extract) {
  mlir::IRRewriter rewriter(mlir_context);
  rewriter.setInsertionPointAfter(extract_loop);
  // Create a new map that has the results of both loops.
  // map = (d0...dn)[s0...sn] ->
  //    (insert_loop_results..., extract_loop_results...)
  auto insert_loop_map = insert_loop.getIndexingMap();
  auto extract_loop_map = extract_loop.getIndexingMap();
  auto map = insert_loop_map.GetAffineMap();
  for (auto res : extract_loop_map.GetAffineMap().getResults()) {
    map = map.insertResult(res, map.getNumResults());
  }
  IndexingMap new_map(map, insert_loop_map.GetDimVars(),
                      insert_loop_map.GetRangeVars(),
                      /*rt_vars=*/{}, insert_loop_map.GetConstraints());

  auto new_loop =
      rewriter.create<LoopOp>(insert_loop.getLoc(), new_map,
                              insert_loop.getDims(), extract_loop.getInits());

  // Make the loops independent of the vector.insert/extract & erase.
  auto vector_cst = insert_loop.getInits().back();
  insert_loop->replaceAllUsesWith(ValueRange(vector_cst));
  extract_loop->replaceAllUsesWith(new_loop.getResults());
  extract.replaceAllUsesWith(insert.getSource());
  auto insert_loop_yield =
      mlir::dyn_cast<YieldOp>(insert_loop.getRegion().front().back());
  rewriter.eraseOp(insert_loop_yield);
  rewriter.eraseOp(extract);
  rewriter.eraseOp(insert);

  // Map old loop arguments to new loop arguments.
  // new_args = [s0...sn, insert_loop_results..., extract_loop_results...,
  //   extract_inits...]
  auto new_args = new_loop.getRegion().front().getArguments();
  auto range_vars = new_args.take_front(new_map.GetRangeVarsCount());
  new_args = new_args.drop_front(range_vars.size());
  auto in_loop_results = new_args.take_front(insert_loop_map.GetNumResults());
  new_args = new_args.drop_front(in_loop_results.size());
  auto ex_loop_results = new_args.take_front(extract_loop_map.GetNumResults());
  auto extract_inits = new_args.take_back(extract_loop.getInits().size());

  // old_insert_args = [s0...sn, insert_loop_results..., vector_cst]
  SmallVector<Value> old_insert_args;
  old_insert_args.append(range_vars.begin(), range_vars.end());
  old_insert_args.append(in_loop_results.begin(), in_loop_results.end());
  old_insert_args.push_back(vector_cst);

  // old_insert_args = [s0...sn, extract_loop_results..., extract_inits...]
  SmallVector<Value> old_extract_args;
  old_extract_args.append(range_vars.begin(), range_vars.end());
  old_extract_args.append(ex_loop_results.begin(), ex_loop_results.end());
  old_extract_args.append(extract_inits.begin(), extract_inits.end());

  // Merge the loops: first insert, then extract.
  rewriter.mergeBlocks(&insert_loop.getRegion().front(),
                       &new_loop.getRegion().front(), old_insert_args);
  rewriter.mergeBlocks(&extract_loop.getRegion().front(),
                       &new_loop.getRegion().front(), old_extract_args);
  rewriter.eraseOp(insert_loop);
  rewriter.eraseOp(extract_loop);
}

// Fuse loops that have the same map, same dim variables, & can be rewritten as
// a single loop, each stacked on top of the next.
void FuseIndependentLoops(MLIRContext* mlir_context,
                          SmallVector<LoopOp>& loops) {
  auto last_loop = loops.back();
  auto map = last_loop.getIndexingMap();
  mlir::IRRewriter rewriter(mlir_context);
  rewriter.setInsertionPointAfter(last_loop);

  SmallVector<Value> inits;
  SmallVector<Value> results;
  for (auto loop : loops) {
    inits.append(loop.getInits().begin(), loop.getInits().end());
    auto yield_op = loop.getBody()->getTerminator();
    auto yields = yield_op->getOperands();
    results.append(yields.begin(), yields.end());
    yield_op->erase();
  }
  auto new_loop = rewriter.create<LoopOp>(last_loop.getLoc(), map,
                                          last_loop.getDims(), inits);

  auto new_args = new_loop.getRegion().front().getArguments();
  int common_args_count = map.GetRangeVarsCount() + map.GetNumResults();
  auto common_args = new_args.take_front(common_args_count);
  auto init_args = new_args.drop_front(common_args_count);
  auto new_results = new_loop.getResults();

  for (auto loop : loops) {
    int num_results = loop.getNumResults();
    loop->replaceAllUsesWith(new_results.take_front(num_results));
    new_results = new_results.drop_front(num_results);
    SmallVector<Value> old_args(common_args);
    auto old_inits = init_args.take_front(num_results);
    old_args.append(old_inits.begin(), old_inits.end());
    init_args = init_args.drop_front(num_results);

    rewriter.mergeBlocks(&loop.getRegion().front(),
                         &new_loop.getRegion().front(), old_args);
    rewriter.eraseOp(loop);
  }
  rewriter.setInsertionPointToEnd(new_loop.getBody());
  rewriter.create<YieldOp>(new_loop.getLoc(), results);
}

void FuseSameMapLoopsIfPossible(MLIRContext* mlir_context,
                                SmallVector<LoopOp>& loops) {
  if (loops.size() < 2) return;
  auto last_loop = loops.back();
  loops.pop_back();
  SmallVector<LoopOp> eligible_loops;
  for (auto loop : loops) {
    if (LoopDominatesLoop(/*dominator=*/last_loop, /*dominatee=*/loop) &&
        LoopsUseSameDimOps(last_loop, loop)) {
      eligible_loops.push_back(loop);
    }
  }
  eligible_loops.push_back(last_loop);

  if (eligible_loops.size() < 2) return;
  FuseIndependentLoops(mlir_context, eligible_loops);
}

void FuseExtractIfPossible(MLIRContext* mlir_context, mv::ExtractOp extract) {
  // Check that it has the following pattern:
  // %insert_loop = { %insert = vector.insert ... }
  // %extract_loop = { %extract = vector.extract %insert_loop }
  auto extract_loop = extract->getParentOfType<LoopOp>();
  if (!extract_loop) return;
  if (!extract.getVector().getDefiningOp()) return;
  auto insert_loop =
      mlir::dyn_cast<LoopOp>(extract.getVector().getDefiningOp());
  if (!insert_loop) return;
  SmallVector<mv::InsertOp> inserts;
  // If necessary, the insert_loop result size constraint may be relaxed.
  if (insert_loop.getResults().size() != 1) return;
  for (auto user : insert_loop.getRegionIterArgs().back().getUsers()) {
    if (auto insert = mlir::dyn_cast<mv::InsertOp>(user)) {
      inserts.push_back(insert);
    }
  }
  if (inserts.size() != 1) return;
  auto insert = inserts.front();

  // Check that the vector isn't being used anywhere else so it can be
  // removed entirely; we already know from above it's being used by
  // extract so it should have exactly one use.
  if (!insert_loop.getResult(0).hasOneUse()) return;

  if (!LoopsHaveTheSameDomain(insert_loop, extract_loop)) return;
  // Only fuse loops if we are extracting from the same position that we are
  // inserting into on each iteration.
  if (!IndicesAreEqualAndInjective(insert_loop.getNumInductionVars(), insert,
                                   extract)) {
    return;
  }

  // All requirements have been met: fuse loops.
  FuseExtractInsertLoopPair(mlir_context, insert_loop, extract_loop, insert,
                            extract);
}

struct FuseLoopsPass : public impl::FuseLoopsPassBase<FuseLoopsPass> {
  void runOnOperation() override {
    auto mlir_context = &getContext();

    SmallVector<mv::ExtractOp> extracts;
    getOperation()->walk([&](Operation* op) -> void {
      if (auto extract = mlir::dyn_cast<mv::ExtractOp>(op)) {
        extracts.push_back(extract);
      }
    });
    for (auto extract : extracts) {
      FuseExtractIfPossible(mlir_context, extract);
    }

    // Fuse loops with the same map & that do not affect each other.
    mlir::DenseMap<mlir::Attribute, SmallVector<LoopOp>> loops_by_map;
    getOperation()->walk([&](Operation* op) -> void {
      if (auto loop = mlir::dyn_cast<LoopOp>(op)) {
        loops_by_map[loop.getIndexingMapAttr()].push_back(loop);
      }
    });
    for (auto [_, loops] : loops_by_map) {
      FuseSameMapLoopsIfPossible(mlir_context, loops);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFuseLoopsPass() {
  return std::make_unique<FuseLoopsPass>();
}

}  // namespace gpu
}  // namespace xla
