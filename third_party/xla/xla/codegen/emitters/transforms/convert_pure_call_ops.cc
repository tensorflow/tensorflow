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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_CONVERTPURECALLOPSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

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
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateConvertPureCallOpsPass() {
  return std::make_unique<ConvertPureCallOpsPass>();
}

}  // namespace emitters
}  // namespace xla
