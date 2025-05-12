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

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla::gpu {

namespace {

#define GEN_PASS_DEF_LOWERXLASHAREDPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

struct LowerForall : mlir::OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ForallOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);

    mlir::Block* old_block = op.getBody();
    if (old_block == nullptr) {
      return mlir::failure();
    }
    mlir::IRMapping mapping;
    for (const auto& [idx, workgroup] : llvm::enumerate(
             llvm::zip(op.getInductionVars(), op.getWorkgroupDims()))) {
      auto& [argument, size] = workgroup;
      // xla.range is a closed interval so we subtract 1 from the size which is
      // half-open.
      int64_t upper_range = mlir::cast<mlir::IntegerAttr>(size).getInt() - 1;
      auto thread_id = b.create<mlir::gpu::ThreadIdOp>(
          static_cast<mlir::gpu::Dimension>(idx));
      thread_id->setAttr("xla.range", b.getIndexArrayAttr({0, upper_range}));
      mapping.map(argument, thread_id);
    }

    for (auto [region_arg, operand] :
         llvm::zip(op.getRegionOutputArgs(), op.getOutputArgs())) {
      mapping.map(region_arg, operand);
    }

    for (mlir::Operation& inner_op : old_block->without_terminator()) {
      b.clone(inner_op, mapping);
    }

    // Get the results from the original loop's yield operation, remapped
    mlir::SmallVector<mlir::Value> new_results;
    for (mlir::Value terminator_operand :
         old_block->getTerminator()->getOperands()) {
      new_results.push_back(mapping.lookupOrDefault(terminator_operand));
    }

    // Replace the forOp with the new results
    rewriter.replaceOp(op, new_results);

    return mlir::success();
  }
};

struct LowerXlaSharedPass final
    : public impl::LowerXlaSharedPassBase<LowerXlaSharedPass> {
 public:
  void runOnOperation() final {
    mlir::RewritePatternSet lower_shared_patterns(&getContext());
    lower_shared_patterns.add<LowerForall>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(lower_shared_patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerXlaSharedPass() {
  return std::make_unique<LowerXlaSharedPass>();
}

}  // namespace xla::gpu
