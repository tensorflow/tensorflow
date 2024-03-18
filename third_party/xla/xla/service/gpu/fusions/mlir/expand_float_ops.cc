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
#include <type_traits>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Math/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

template <typename OpTy, mlir::arith::CmpFPredicate pred>
struct RewriteToCmpSelect : public mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  RewriteToCmpSelect(mlir::MLIRContext* context, bool include_f32)
      : mlir::OpRewritePattern<OpTy>(context), include_f32(include_f32) {}

  mlir::LogicalResult matchAndRewrite(
      OpTy op, mlir::PatternRewriter& rewriter) const override {
    if (op.getType().isF32() && !include_f32) {
      return rewriter.notifyMatchFailure(op, "not rewriting f32 min/max");
    }

    auto lhs_is_nan = rewriter.create<mlir::arith::CmpFOp>(
        op.getLoc(), mlir::arith::CmpFPredicate::UNE, op.getLhs(), op.getLhs());
    auto rhs_is_not_nan = rewriter.create<mlir::arith::CmpFOp>(
        op.getLoc(), mlir::arith::CmpFPredicate::OEQ, op.getRhs(), op.getRhs());

    auto return_lhs = rewriter
                          .create<mlir::arith::CmpFOp>(op.getLoc(), pred,
                                                       op.getLhs(), op.getRhs())
                          .getResult();

    // logic: isNaN(lhs) || (!isNan(rhs) && return_lhs) ? lhs : rhs
    return_lhs = rewriter.create<mlir::arith::OrIOp>(
        op.getLoc(), lhs_is_nan,
        rewriter.create<mlir::arith::AndIOp>(op.getLoc(), rhs_is_not_nan,
                                             return_lhs));

    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, op.getResult().getType(), return_lhs, op.getLhs(), op.getRhs());
    return mlir::success();
  }

  bool include_f32;
};

struct RewriteErf32Pattern : public mlir::OpRewritePattern<mlir::math::ErfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ErfOp op, mlir::PatternRewriter& rewriter) const override {
    namespace ma = mlir::arith;
    if (!op.getType().isF32()) {
      return rewriter.notifyMatchFailure(op, "not an f32 erf");
    }

    static const std::array<float, 5> kAlpha{
        0.00022905065861350646f, 0.0034082910107109506f, 0.050955695062380861f,
        0.18520832239976145f, 1.128379143519084f};

    static const std::array<float, 7> kBeta{-1.1791602954361697e-7,
                                            0.000023547966471313185f,
                                            0.0010179625278914885f,
                                            0.014070470171167667f,
                                            0.11098505178285362f,
                                            0.49746925110067538f,
                                            1.0f};

    // We clamp x to be within [-c;c] where c = erfinv(1-2^-23), outside of
    // which x should be +/-1.
    constexpr float kErfInvOneMinusHalfULP = 3.7439211627767994f;

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto c = [&](float v) -> mlir::Value {
      return b.create<ma::ConstantFloatOp>(llvm::APFloat(v),
                                           rewriter.getF32Type());
    };

    auto poly = [&](auto x, auto coefficients) -> mlir::Value {
      auto r = c(coefficients[0]);
      for (int i = 1; i < coefficients.size(); ++i) {
        r = b.create<mlir::math::FmaOp>(r, x, c(coefficients[i]));
      }
      return r;
    };

    mlir::Value x = op.getOperand();
    x = b.create<ma::MaximumFOp>(x, c(-kErfInvOneMinusHalfULP));
    x = b.create<ma::MinimumFOp>(x, c(kErfInvOneMinusHalfULP));
    mlir::Value x2 = b.create<ma::MulFOp>(x, x);

    rewriter.replaceOpWithNewOp<ma::DivFOp>(
        op, b.create<ma::MulFOp>(x, poly(x2, kAlpha)), poly(x2, kBeta));

    return mlir::success();
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteToCmpSelect<mlir::arith::MinimumFOp,
                                    mlir::arith::CmpFPredicate::OLE>>(
        &getContext(), /*include_f32=*/pre_ampere_);
    patterns.add<RewriteToCmpSelect<mlir::arith::MaximumFOp,
                                    mlir::arith::CmpFPredicate::OGE>>(
        &getContext(), /*include_f32=*/pre_ampere_);
    mlir::populatePolynomialApproximateTanhPattern(patterns);
    patterns.add<RewriteErf32Pattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass(bool pre_ampere) {
  return createExpandFloatOpsPass(ExpandFloatOpsPassOptions{pre_ampere});
}

}  // namespace gpu
}  // namespace xla
