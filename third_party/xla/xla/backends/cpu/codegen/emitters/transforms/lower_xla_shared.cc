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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
#include "xla/codegen/emitters/ir/xla_dialect.h.inc"

namespace xla::cpu {

namespace {

#define GEN_PASS_DEF_LOWERXLASHAREDPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

struct LowerForall : mlir::OpRewritePattern<mlir::scf::ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForallOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder builder(loc, rewriter);

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

    size_t num_dims = op.getStaticUpperBound().size();
    llvm::SmallVector<mlir::Value, 3> lbs(
        num_dims, builder.create<mlir::arith::ConstantIndexOp>(0));
    llvm::SmallVector<mlir::Value, 3> steps(
        num_dims, builder.create<mlir::arith::ConstantIndexOp>(1));
    llvm::SmallVector<mlir::Value, 3> ubs;
    for (int64_t size : op.getStaticUpperBound()) {
      ubs.push_back(builder.create<mlir::arith::ConstantIndexOp>(size));
    }

    mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, op.getOutputs(),
        [&op, old_block](mlir::OpBuilder& nested_builder,
                         mlir::Location nested_loc, mlir::ValueRange ivs,
                         mlir::ValueRange iter_args) {
          mlir::ImplicitLocOpBuilder nested_b(nested_loc, nested_builder);
          mlir::IRMapping mapping;
          for (const auto& [old_arg, iv] :
               llvm::zip(old_block->getArguments(), ivs)) {
            mapping.map(old_arg, iv);
          }
          for (const auto& [region_arg, operand] :
               llvm::zip(op.getRegionOutArgs(), iter_args)) {
            mapping.map(region_arg, operand);
          }

          for (mlir::Operation& inner_op : old_block->without_terminator()) {
            nested_b.clone(inner_op, mapping);
          }

          mlir::scf::InParallelOp terminator = op.getTerminator();
          mlir::SmallVector<mlir::Value> new_results;
          for (mlir::Operation& yielding_op : terminator.getYieldingOps()) {
            new_results.push_back(mapping.lookupOrDefault(
                mlir::cast<mlir::tensor::ParallelInsertSliceOp>(yielding_op)
                    .getSource()));
          }

          return new_results;
        });

    rewriter.replaceOp(op, loop_nest.results);
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

}  // namespace xla::cpu
