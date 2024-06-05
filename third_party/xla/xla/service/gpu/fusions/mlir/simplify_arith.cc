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
#include <algorithm>
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
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_SIMPLIFYARITHPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

Interval::ComparisonResult EvaluateCmpI(mlir::arith::CmpIPredicate pred,
                                        Interval lhs, Interval rhs) {
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
    auto rhs = GetRange(op.getRhs());
    auto lhs = GetRange(op.getLhs());
    if (!lhs || !rhs) {
      return rewriter.notifyMatchFailure(op, "failed to deduce input ranges");
    }
    Interval::ComparisonResult result =
        EvaluateCmpI(op.getPredicate(), *lhs, *rhs);
    if (result != std::nullopt) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(
          op, *result, rewriter.getI1Type());
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(op, "not a constant result");
  }
};

struct RewriteMaxSi : mlir::OpRewritePattern<mlir::arith::MaxSIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::MaxSIOp op, mlir::PatternRewriter& rewriter) const override {
    auto lhs = GetRange(op.getLhs());
    auto rhs = GetRange(op.getRhs());
    if (!lhs || !rhs) {
      return rewriter.notifyMatchFailure(op, "failed to deduce input ranges");
    }
    if (auto lhs_ge_rhs = *lhs >= *rhs; lhs_ge_rhs == true) {
      rewriter.replaceOp(op, op.getLhs());
    } else if (auto rhs_ge_lhs = *rhs >= *lhs; rhs_ge_lhs == true) {
      rewriter.replaceOp(op, op.getRhs());
    } else {
      return rewriter.notifyMatchFailure(op, "not equal to lhs or rhs");
    }
    return mlir::success();
  }
};

struct RewriteMinSi : mlir::OpRewritePattern<mlir::arith::MinSIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::MinSIOp op, mlir::PatternRewriter& rewriter) const override {
    auto lhs = GetRange(op.getLhs());
    auto rhs = GetRange(op.getRhs());
    if (!lhs || !rhs) {
      return rewriter.notifyMatchFailure(op, "failed to deduce input ranges");
    }
    if (auto lhs_le_rhs = *lhs <= *rhs; lhs_le_rhs == true) {
      rewriter.replaceOp(op, op.getLhs());
    } else if (auto rhs_le_lhs = *rhs <= *lhs; rhs_le_lhs == true) {
      rewriter.replaceOp(op, op.getRhs());
    } else {
      return rewriter.notifyMatchFailure(op, "not equal to lhs or rhs");
    }
    return mlir::success();
  }
};

void AnnotateRanges(mlir::func::FuncOp func) {
  func->walk([](mlir::Operation* op) {
    if (op->getNumResults() != 1) {
      return;
    }

    auto result = op->getResult(0);
    if (GetRange(result).has_value()) {
      return;
    }

    auto get_range = [](mlir::Value value) -> Interval {
      auto range = GetRange(value);
      if (range) {
        return *range;
      }
      return {std::numeric_limits<int64_t>::min(),
              std::numeric_limits<int64_t>::max()};
    };

    std::optional<Interval> out_range = std::nullopt;
    if (mlir::isa<mlir::arith::MaxSIOp, mlir::arith::MinSIOp,
                  mlir::arith::AddIOp, mlir::arith::MulIOp>(op)) {
      auto lhs_range = get_range(op->getOperand(0));
      auto rhs_range = get_range(op->getOperand(1));
      if (mlir::isa<mlir::arith::MaxSIOp>(op)) {
        out_range = lhs_range.max(rhs_range);
      } else if (mlir::isa<mlir::arith::MinSIOp>(op)) {
        out_range = lhs_range.min(rhs_range);
      } else if (mlir::isa<mlir::arith::AddIOp>(op)) {
        out_range = lhs_range + rhs_range;
      } else {
        out_range = lhs_range * rhs_range;
      }
    }

    if (out_range) {
      mlir::OpBuilder b(op);
      op->setAttr("xla.range",
                  b.getIndexArrayAttr({out_range->lower, out_range->upper}));
    }
  });
}

class SimplifyArithPass
    : public impl::SimplifyArithPassBase<SimplifyArithPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    AnnotateRanges(getOperation());
    patterns.add<RewriteCmpI, RewriteMaxSi, RewriteMinSi>(&getContext());
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
