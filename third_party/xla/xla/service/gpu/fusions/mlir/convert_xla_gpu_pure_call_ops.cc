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
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_CONVERTPURECALLOPSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

struct RewriteCall : mlir::OpRewritePattern<PureCallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      PureCallOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getResultTypes(), op.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class ConvertPureCallOpsPass
    : public impl::ConvertPureCallOpsPassBase<ConvertPureCallOpsPass> {
 public:
  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteCall>(ctx);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateConvertPureCallOpsPass() {
  return std::make_unique<ConvertPureCallOpsPass>();
}

}  // namespace gpu
}  // namespace xla
