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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/ir/xla_dialect.h.inc"

namespace xla::gpu {

namespace {

#define GEN_PASS_DEF_LOWERXLASHAREDPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

struct LowerForall : mlir::OpRewritePattern<mlir::scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForallOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);

    mlir::Block* old_block = op.getBody();
    if (old_block == nullptr) {
      return mlir::failure();
    }

    if (!op.getDynamicUpperBound().empty()) {
      return rewriter.notifyMatchFailure(op, "Bounds must be static");
    }

    if (!op.isNormalized()) {
      return rewriter.notifyMatchFailure(op, "Bounds must be normalized");
    }

    mlir::SmallVector<mlir::Value> new_args;
    for (const auto& [idx, size] : llvm::enumerate(op.getStaticUpperBound())) {
      // xla.range is a closed interval so we subtract 1 from the size which is
      // half-open.
      int64_t upper_range = size - 1;
      auto thread_id = b.create<mlir::gpu::ThreadIdOp>(
          static_cast<mlir::gpu::Dimension>(idx));
      thread_id->setAttr("xla.range", b.getIndexArrayAttr({0, upper_range}));
      new_args.push_back(thread_id);
    }

    mlir::Operation::operand_range shared_outputs = op.getOutputs();
    new_args.append(shared_outputs.begin(), shared_outputs.end());

    mlir::scf::InParallelOp terminator = op.getTerminator();
    mlir::SmallVector<mlir::Value> new_results;
    for (mlir::Operation& yielding_op : terminator.getYieldingOps()) {
      new_results.push_back(
          mlir::cast<mlir::tensor::ParallelInsertSliceOp>(yielding_op)
              .getSource());
    }

    rewriter.inlineBlockBefore(old_block, op, new_args);
    rewriter.replaceOp(op, new_results);
    rewriter.eraseOp(terminator);

    return mlir::success();
  }
};

struct LowerXlaSharedPass final
    : public impl::LowerXlaSharedPassBase<LowerXlaSharedPass> {
 public:
  void runOnOperation() final {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerForall>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
