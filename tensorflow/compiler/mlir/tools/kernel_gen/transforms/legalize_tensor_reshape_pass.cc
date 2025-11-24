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

#include <cassert>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace kernel_gen {

#define GEN_PASS_DEF_LEGALIZETENSORRESHAPEPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

namespace {

struct LegalizeTensorReshapePattern
    : public OpRewritePattern<tensor::ReshapeOp> {
  explicit LegalizeTensorReshapePattern(MLIRContext* context)
      : OpRewritePattern<tensor::ReshapeOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, op.getResultType(),
                                                        op.getOperands());
    return success();
  }
};

}  // namespace

struct LegalizeTensorReshapePass
    : public impl::LegalizeTensorReshapePassBase<LegalizeTensorReshapePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LegalizeTensorReshapePattern>(ctx);
    GreedyRewriteConfig config;
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace kernel_gen
}  // namespace mlir
