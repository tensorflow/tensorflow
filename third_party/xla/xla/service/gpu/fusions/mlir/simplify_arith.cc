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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_SIMPLIFYARITHPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

Interval::ComparisonResult EvaluateCmpI(mlir::arith::CmpIPredicate pred,
                                        Interval lhs, int64_t rhs) {
  switch (pred) {
    case mlir::arith::CmpIPredicate::eq:
      return lhs == rhs;
    case mlir::arith::CmpIPredicate::ne:
      return lhs != rhs;
    case mlir::arith::CmpIPredicate::slt:
    case mlir::arith::CmpIPredicate::ult:
      return lhs < rhs;
    case mlir::arith::CmpIPredicate::sle:
    case mlir::arith::CmpIPredicate::ule:
      return lhs <= rhs;
    case mlir::arith::CmpIPredicate::sgt:
    case mlir::arith::CmpIPredicate::ugt:
      return lhs > rhs;
    case mlir::arith::CmpIPredicate::sge:
    case mlir::arith::CmpIPredicate::uge:
      return lhs >= rhs;
  }
}

struct RewriteCmpI : mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::CmpIOp op, mlir::PatternRewriter& rewriter) const override {
    // We don't need to support constants on the LHS, since comparisons are
    // canonicalized to have them on the RHS.
    auto rhs = mlir::getConstantIntValue(op.getRhs());
    auto lhs = GetRange(op.getLhs());
    if (lhs && rhs) {
      Interval::ComparisonResult result =
          EvaluateCmpI(op.getPredicate(), *lhs, *rhs);
      if (result != std::nullopt) {
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(
            op, *result, rewriter.getI1Type());
        return mlir::success();
      }
    }
    // TODO(jreiffers): Consider supporting ranges on the RHS as well.
    return rewriter.notifyMatchFailure(op, "not a constant result");
  }
};

class SimplifyArithPass
    : public impl::SimplifyArithPassBase<SimplifyArithPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteCmpI>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSimplifyArithPass() {
  return std::make_unique<SimplifyArithPass>();
}

}  // namespace gpu
}  // namespace xla
