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
#include <utility>

#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_PEELLOOPSPASS
#include "xla/backends/gpu/codegen/transforms/passes.h.inc"

using mlir::Location;
using mlir::OpBuilder;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;

struct PeelLoop : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      LoopOp loop_op, PatternRewriter& rewriter) const override {
    int64_t cumulative_loop_size = 1;

    // Compute the list of indexing maps. The last element is the "peeled" or
    // "main" loop. Everything else is a "tail" loop.
    auto indexing_map = loop_op.getIndexingMap();
    // TODO(b/358274367): Remove the simplify call once we have `is_simplified`
    // field and a canonicalization pattern to simplify indexing map in
    // xla_gpu.loop.
    indexing_map.Simplify();
    SmallVector<IndexingMap> indexing_maps{indexing_map};
    for (int sym_index = indexing_map.GetSymbolCount() - 1;
         sym_index >= 0 && cumulative_loop_size < 64; --sym_index) {
      IndexingMap indexing_map = indexing_maps.back();
      auto& bound = indexing_map.GetSymbolBound(sym_index);
      cumulative_loop_size *= bound.GetLoopTripCount();
      if (!indexing_map.IsSymbolConstrained(sym_index) ||
          bound.upper == bound.lower) {
        continue;
      }
      // Create peeled indexing map.
      IndexingMap peeled_map = indexing_map;
      --peeled_map.GetMutableSymbolBound(sym_index).upper;
      peeled_map.Simplify();

      // If the symbol is still constrained, peeling does not help.
      if (peeled_map.IsSymbolConstrained(sym_index)) continue;

      // Create remainder indexing map.
      IndexingMap tail_map = indexing_map;
      tail_map.GetMutableSymbolBound(sym_index).lower = bound.upper;
      tail_map.Simplify();

      VLOG(5) << "Peeled indexing map\n"
              << ToString(indexing_map) << "into\n"
              << ToString(peeled_map) << "and\n"
              << ToString(tail_map) << "\n";
      indexing_maps.pop_back();
      indexing_maps.push_back(tail_map);
      indexing_maps.push_back(peeled_map);
    }

    if (indexing_maps.size() == 1) {
      return rewriter.notifyMatchFailure(loop_op,
                                         "No range variables to peel.");
    }

    // Create chained loops from the list of indexing maps.
    Location loc = loop_op.getLoc();
    SmallVector<Value, 4> inits = loop_op.getInits();
    for (const auto& indexing_map : llvm::reverse(indexing_maps)) {
      if (indexing_map.IsKnownEmpty()) continue;
      auto tail_loop = rewriter.create<LoopOp>(
          loc, indexing_map, loop_op.getDims(), inits,
          [&](OpBuilder& nested_b, Location nested_loc, ValueRange ivs,
              ValueRange map_results, ValueRange iter_args) {
            OpBuilder::InsertionGuard guard(nested_b);
            mlir::IRMapping mapping;
            mapping.map(loop_op.getInductionVars(), ivs);
            mapping.map(loop_op.getIndexingMapResults(), map_results);
            mapping.map(loop_op.getRegionIterArgs(), iter_args);
            for (auto& op : loop_op.getBody()->getOperations()) {
              nested_b.clone(op, mapping);
            }
          });
      inits = tail_loop.getResults();
    }
    rewriter.replaceOp(loop_op, inits);
    return mlir::success();
  }
};

struct PeelLoopsPass : public impl::PeelLoopsPassBase<PeelLoopsPass> {
  void runOnOperation() override {
    auto func = getOperation();
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<PeelLoop>(mlir_context);
    if (mlir::failed(
            mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreatePeelLoopsPass() {
  return std::make_unique<PeelLoopsPass>();
}

}  // namespace gpu
}  // namespace xla
