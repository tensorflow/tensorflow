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
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla::cpu {

namespace {

#define GEN_PASS_DEF_LOWERXLASHAREDPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

struct LowerForall : mlir::OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ForallOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder builder(loc, rewriter);

    mlir::Block* old_block = op.getBody();
    if (old_block == nullptr) {
      return mlir::failure();
    }

    size_t num_dims = op.getWorkgroupDimsAttr().size();
    llvm::SmallVector<mlir::Value, 3> lbs(
        num_dims, builder.create<mlir::arith::ConstantIndexOp>(0));
    llvm::SmallVector<mlir::Value, 3> steps(
        num_dims, builder.create<mlir::arith::ConstantIndexOp>(1));
    llvm::SmallVector<mlir::Value, 3> ubs;
    for (const mlir::Attribute& attr : op.getWorkgroupDimsAttr()) {
      ubs.push_back(builder.create<mlir::arith::ConstantIndexOp>(
          mlir::cast<mlir::IntegerAttr>(attr).getInt()));
    }

    mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
        builder, loc, lbs, ubs, steps, op.getOutputArgs(),
        [&op, old_block](mlir::OpBuilder& nested_builder,
                         mlir::Location nested_loc, mlir::ValueRange ivs,
                         mlir::ValueRange iter_args) {
          mlir::ImplicitLocOpBuilder nested_b(nested_loc, nested_builder);
          mlir::IRMapping mapping;
          for (const auto& [old_arg, iv] :
               llvm::zip(old_block->getArguments(), ivs)) {
            mapping.map(old_arg, iv);
          }
          for (auto [region_arg, operand] :
               llvm::zip(op.getRegionOutputArgs(), iter_args)) {
            mapping.map(region_arg, operand);
          }

          for (mlir::Operation& inner_op : old_block->without_terminator()) {
            nested_b.clone(inner_op, mapping);
          }

          // Get the results from the original loop's yield operation, remapped
          mlir::SmallVector<mlir::Value> new_results;
          for (mlir::Value terminator_operand :
               old_block->getTerminator()->getOperands()) {
            new_results.push_back(mapping.lookupOrDefault(terminator_operand));
          }
          return new_results;
        });

    rewriter.replaceOp(op, loop_nest.results);
    return mlir::success();
  }
};

struct LowerWorkgroupId : mlir::OpRewritePattern<WorkgroupIdOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      WorkgroupIdOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);

    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (func == nullptr) {
      return rewriter.notifyMatchFailure(op, "No parent func found.");
    }
    if (func.getArguments().empty()) {
      return rewriter.notifyMatchFailure(op, "Parent func has no operands.");
    }

    mlir::Value first_arg = func.getArgument(0);
    if (!mlir::isa<CallFrameType>(first_arg.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Parent func first arg is not a call frame.");
    }

    auto thread_id =
        b.create<ThreadIdOp>(b.getIndexType(), first_arg, op.getDimension());
    rewriter.replaceOp(op, thread_id);
    return mlir::success();
  }
};

struct LowerXlaSharedPass final
    : public impl::LowerXlaSharedPassBase<LowerXlaSharedPass> {
 public:
  void runOnOperation() final {
    mlir::RewritePatternSet lower_shared_patterns(&getContext());
    lower_shared_patterns.add<LowerForall, LowerWorkgroupId>(&getContext());
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

}  // namespace xla::cpu
