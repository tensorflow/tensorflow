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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using mlir::AffineBinaryOpExpr;
using mlir::AffineConstantExpr;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::ImplicitLocOpBuilder;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir::affine::AffineApplyOp;

namespace arith = mlir::arith;

#define GEN_PASS_DEF_SIMPLIFYAFFINEPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

Value EvaluateExpression(ImplicitLocOpBuilder& b, AffineExpr expr,
                         unsigned dim_count, ValueRange operands) {
  if (auto bin_op = mlir::dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = EvaluateExpression(b, bin_op.getLHS(), dim_count, operands);
    auto rhs = EvaluateExpression(b, bin_op.getRHS(), dim_count, operands);
    switch (expr.getKind()) {
      case AffineExprKind::Add:
        return b.create<arith::AddIOp>(lhs, rhs);
      case AffineExprKind::Mul:
        return b.create<arith::MulIOp>(lhs, rhs);
      case AffineExprKind::Mod:
        return b.create<arith::RemUIOp>(lhs, rhs);
      case AffineExprKind::FloorDiv:
        return b.create<arith::DivUIOp>(lhs, rhs);
      default:
        ABSL_UNREACHABLE();
    }
  }
  switch (expr.getKind()) {
    case AffineExprKind::Constant:
      return b.create<arith::ConstantIndexOp>(
          mlir::cast<AffineConstantExpr>(expr).getValue());
    case AffineExprKind::DimId:
      return operands[mlir::cast<AffineDimExpr>(expr).getPosition()];
    case AffineExprKind::SymbolId:
      return operands[dim_count +
                      mlir::cast<AffineSymbolExpr>(expr).getPosition()];
    default:
      ABSL_UNREACHABLE();
  }
}

bool IsLoweringSupported(AffineExpr expr, RangeEvaluator& range_evaluator) {
  auto bin_op = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin_op) {
    return true;
  }
  // Mod and div can be lowered if their LHS is >= 0 and their RHS is a
  // constant.
  if (expr.getKind() == AffineExprKind::Mod ||
      expr.getKind() == AffineExprKind::FloorDiv) {
    if (!range_evaluator.IsAlwaysPositiveOrZero(bin_op.getLHS()) ||
        !range_evaluator.ComputeExpressionRange(bin_op.getRHS()).IsPoint()) {
      return false;
    }
  }
  if (expr.getKind() == AffineExprKind::CeilDiv) {
    return false;
  }
  return IsLoweringSupported(bin_op.getLHS(), range_evaluator) &&
         IsLoweringSupported(bin_op.getRHS(), range_evaluator);
}

struct RewriteAffineApply : OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::affine::AffineApplyOp op,
                                PatternRewriter& rewriter) const override {
    AffineMap affine_map = op.getAffineMap();
    std::vector<DimVar> dim_ranges(affine_map.getNumDims());
    std::vector<RangeVar> symbol_ranges(affine_map.getNumSymbols());

    for (int i = 0; i < affine_map.getNumInputs(); ++i) {
      if (auto range = GetRange(op->getOperand(i))) {
        if (i >= dim_ranges.size()) {
          symbol_ranges[i - dim_ranges.size()] = RangeVar{*range};
        } else {
          dim_ranges[i] = DimVar{*range};
        }
      } else {
        return rewriter.notifyMatchFailure(op, "failed to deduce range");
      }
    }

    IndexingMap indexing_map(affine_map, std::move(dim_ranges),
                             std::move(symbol_ranges),
                             /*rt_vars=*/{});
    indexing_map.Simplify(GetIndexingMapForInstruction);
    auto result_expr = indexing_map.GetAffineMap().getResult(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    RangeEvaluator range_evaluator = indexing_map.GetRangeEvaluator();
    if (!IsLoweringSupported(result_expr, range_evaluator)) {
      return rewriter.notifyMatchFailure(op,
                                         "unable to lower the affine apply");
    }
    b.setInsertionPoint(op);
    auto result = EvaluateExpression(
        b, result_expr, indexing_map.GetDimensionCount(), op->getOperands());
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RewriteApplyIndexingOp : OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp op,
                                PatternRewriter& rewriter) const override {
    auto indexing_map = op.getIndexingMap();
    indexing_map.Simplify(GetIndexingMapForInstruction);
    auto affine_map = indexing_map.GetAffineMap();
    int64_t dim_count = indexing_map.GetDimensionCount();
    auto operands = op->getOperands();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    RangeEvaluator range_evaluator = indexing_map.GetRangeEvaluator();

    b.setInsertionPoint(op);
    SmallVector<Value, 4> results;
    results.reserve(affine_map.getNumResults());
    for (AffineExpr result_expr : affine_map.getResults()) {
      // If the expression cannot be lowered, we convert it to affine.apply,
      // since it supports more expression types.
      results.push_back(
          IsLoweringSupported(result_expr, range_evaluator)
              ? EvaluateExpression(b, result_expr, dim_count, operands)
              : b.create<AffineApplyOp>(affine_map, operands));
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct SimplifyAffinePass
    : public impl::SimplifyAffinePassBase<SimplifyAffinePass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteAffineApply, RewriteApplyIndexingOp>(ctx);
    mlir::GreedyRewriteConfig config;
    // There's no point simplifying more than once.
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::optional<Interval> GetRange(mlir::Value value) {
  auto attr_to_range = [](mlir::Attribute attr) -> std::optional<Interval> {
    if (!attr) {
      return std::nullopt;
    }
    auto values = llvm::to_vector(
        mlir::cast<mlir::ArrayAttr>(attr).getAsValueRange<mlir::IntegerAttr>());
    return {{values[0].getSExtValue(), values[1].getSExtValue()}};
  };

  if (value.getDefiningOp()) {
    return attr_to_range(value.getDefiningOp()->getAttr("xla.range"));
  }

  auto bbarg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!bbarg) {
    return std::nullopt;
  }

  auto parent = bbarg.getParentBlock()->getParentOp();
  if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
    return attr_to_range(func_op.getArgAttr(bbarg.getArgNumber(), "xla.range"));
  }

  if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
    llvm::APInt lb, ub;
    if (mlir::matchPattern(for_op.getLowerBound(), mlir::m_ConstantInt(&lb)) &&
        mlir::matchPattern(for_op.getUpperBound(), mlir::m_ConstantInt(&ub))) {
      return {{lb.getSExtValue(), ub.getSExtValue() - 1}};
    }
  }
  return std::nullopt;
}

std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass() {
  return std::make_unique<SimplifyAffinePass>();
}

}  // namespace gpu
}  // namespace xla
