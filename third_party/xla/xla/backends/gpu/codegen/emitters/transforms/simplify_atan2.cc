/* Copyright 2026 The OpenXLA Authors.

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

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_SIMPLIFYATAN2PASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

class SimplifyAtan2Pattern
    : public mlir::OpRewritePattern<mlir::math::Atan2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Atan2Op op, mlir::PatternRewriter& rewriter) const override {
    mlir::Value rhs = op.getRhs();

    mlir::APFloat const_val(0.0);
    if (!mlir::matchPattern(rhs, mlir::m_ConstantFloat(&const_val))) {
      return rewriter.notifyMatchFailure(op, "RHS is not a constant float");
    }

    if (!const_val.isExactlyValue(1.0)) {
      return rewriter.notifyMatchFailure(op, "RHS is not 1.0");
    }

    rewriter.replaceOpWithNewOp<mlir::math::AtanOp>(op, op.getType(),
                                                    op.getLhs());
    return mlir::success();
  }
};

class SimplifyAtan2Pass
    : public impl::SimplifyAtan2PassBase<SimplifyAtan2Pass> {
 public:
  using SimplifyAtan2PassBase::SimplifyAtan2PassBase;
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyAtan2Pattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace gpu
}  // namespace xla
