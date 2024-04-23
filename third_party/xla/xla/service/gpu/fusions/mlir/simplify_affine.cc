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

#include "absl/base/optimization.h"
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
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_SIMPLIFYAFFINEPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

std::optional<Interval> GetRange(mlir::Value value) {
  auto attr_to_range = [](mlir::Attribute attr) -> std::optional<Interval> {
    if (!attr) {
      return std::nullopt;
    }
    auto values = llvm::to_vector(
        attr.cast<mlir::ArrayAttr>().getAsValueRange<mlir::IntegerAttr>());
    return {{values[0].getSExtValue(), values[1].getSExtValue()}};
  };

  if (value.getDefiningOp()) {
    return attr_to_range(value.getDefiningOp()->getAttr("xla.range"));
  }

  auto bbarg = value.dyn_cast<mlir::BlockArgument>();
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

namespace {

class SimplifyAffinePass
    : public impl::SimplifyAffinePassBase<SimplifyAffinePass> {
 public:
  void runOnOperation() override;
};

struct RewriteAffineApply
    : mlir::OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::affine::AffineApplyOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto affine_map = op.getAffineMap();
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

    IndexingMap map(op.getAffineMap(), dim_ranges, symbol_ranges,
                    /*rt_vars=*/{});
    map.Simplify(GetIndexingMapForInstruction);
    auto expr = map.GetAffineMap().getResult(0);

    RangeEvaluator range_evaluator(map.GetDimensionBounds(),
                                   map.GetSymbolBounds(), op->getContext());
    std::function<bool(mlir::AffineExpr)> can_be_lowered;
    bool fits_32_bits = true;
    can_be_lowered = [&](mlir::AffineExpr expr) {
      auto range = range_evaluator.ComputeExpressionRange(expr);
      fits_32_bits &= range.upper < std::numeric_limits<int32_t>::max();

      auto bin_op = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
      if (!bin_op) {
        return true;
      }

      // Mod and div can be lowered if their LHS is >= 0 and their RHS is a
      // constant.
      if (expr.getKind() == mlir::AffineExprKind::Mod ||
          expr.getKind() == mlir::AffineExprKind::FloorDiv) {
        if (!range_evaluator.IsAlwaysPositiveOrZero(bin_op.getLHS()) ||
            !range_evaluator.ComputeExpressionRange(bin_op.getRHS())
                 .IsPoint()) {
          return false;
        }
      }
      if (expr.getKind() == mlir::AffineExprKind::CeilDiv) {
        return false;
      }

      return can_be_lowered(bin_op.getLHS()) && can_be_lowered(bin_op.getRHS());
    };

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    if (!can_be_lowered(expr)) {
      auto range = range_evaluator.ComputeExpressionRange(expr);
      op->setAttr("xla.range", b.getIndexArrayAttr({range.lower, range.upper}));
      return rewriter.notifyMatchFailure(op,
                                         "unable to lower the affine apply");
    }

    std::function<mlir::Value(mlir::AffineExpr)> lower;

    auto int_ty = fits_32_bits ? b.getI32Type() : b.getI64Type();
    b.setInsertionPoint(op);
    lower = [&](mlir::AffineExpr expr) -> mlir::Value {
      if (auto bin_op = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        auto lhs = lower(bin_op.getLHS());
        auto rhs = lower(bin_op.getRHS());
        switch (expr.getKind()) {
          case mlir::AffineExprKind::Add:
            return b.create<mlir::arith::AddIOp>(lhs, rhs);
          case mlir::AffineExprKind::Mul:
            return b.create<mlir::arith::MulIOp>(lhs, rhs);
          case mlir::AffineExprKind::Mod:
            return b.create<mlir::arith::RemUIOp>(lhs, rhs);
          case mlir::AffineExprKind::FloorDiv:
            return b.create<mlir::arith::DivUIOp>(lhs, rhs);
          default:
            ABSL_UNREACHABLE();
        }
      }

      switch (expr.getKind()) {
        case mlir::AffineExprKind::Constant:
          return b.create<mlir::arith::ConstantIntOp>(
              mlir::cast<mlir::AffineConstantExpr>(expr).getValue(), int_ty);
        case mlir::AffineExprKind::DimId:
          return b.create<mlir::arith::IndexCastUIOp>(
              int_ty, op.getDimOperands()[mlir::cast<mlir::AffineDimExpr>(expr)
                                              .getPosition()]);
        case mlir::AffineExprKind::SymbolId:
          return b.create<mlir::arith::IndexCastUIOp>(
              int_ty,
              op.getSymbolOperands()[mlir::cast<mlir::AffineSymbolExpr>(expr)
                                         .getPosition()]);
        default:
          ABSL_UNREACHABLE();
      }
    };

    auto result = lower(map.GetAffineMap().getResult(0));
    auto result_range =
        range_evaluator.ComputeExpressionRange(map.GetAffineMap().getResult(0));
    rewriter
        .replaceOpWithNewOp<mlir::arith::IndexCastUIOp>(op, b.getIndexType(),
                                                        result)
        ->setAttr("xla.range", b.getIndexArrayAttr(
                                   {result_range.lower, result_range.upper}));
    return mlir::success();
  }
};

void SimplifyAffinePass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<RewriteAffineApply>(&getContext());
  mlir::GreedyRewriteConfig config;
  // There's no point simplifying more than once.
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(patterns), config))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass() {
  return std::make_unique<SimplifyAffinePass>();
}

}  // namespace gpu
}  // namespace xla
