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
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"

namespace xla::emitters {

#define GEN_PASS_DEF_SAFEINTEGERARITHMETICPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

inline mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Type type, mlir::TypedAttr value) {
  if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type)) {
    value = mlir::SplatElementsAttr::get(shaped_type, value);
  }
  return mlir::arith::ConstantOp::create(builder, type, value);
}

inline mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Type type, mlir::APInt value) {
  return GetConstantOrSplat(
      builder, type,
      builder.getIntegerAttr(mlir::getElementTypeOrSelf(type), value));
}

template <typename OpT, bool is_unsigned>
mlir::LogicalResult RewriteToSafeDiv(OpT op, mlir::PatternRewriter& rewriter) {
  if (!op->hasAttr("xla.guard_ub")) {
    return rewriter.notifyMatchFailure(op, "already safe");
  }

  mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  mlir::Type element_type = mlir::getElementTypeOrSelf(op.getType());
  auto int_type = mlir::cast<mlir::IntegerType>(element_type);
  mlir::Value minusOne = GetConstantOrSplat(
      builder, op.getType(), mlir::APInt::getAllOnes(int_type.getWidth()));
  mlir::Value smin =
      GetConstantOrSplat(builder, op.getType(),
                         mlir::APInt::getSignedMinValue(int_type.getWidth()));
  mlir::Value safe_div = mlir::mhlo::impl::makeSafeIntDiv<mlir::arith::DivUIOp,
                                                          mlir::arith::DivSIOp>(
      builder, is_unsigned, op.getLhs(), op.getRhs(),
      /*returnedOnZero=*/minusOne,
      /*returnedOnSignedOverflow=*/smin);

  rewriter.replaceOp(op, safe_div);
  return mlir::success();
}

template <typename OpT, bool is_unsigned>
mlir::LogicalResult RewriteToSafeRem(OpT op, mlir::PatternRewriter& rewriter) {
  if (!op->hasAttr("xla.guard_ub")) {
    return rewriter.notifyMatchFailure(op, "already safe");
  }

  mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  mlir::Type element_type = mlir::getElementTypeOrSelf(op.getType());
  auto int_type = mlir::cast<mlir::IntegerType>(element_type);
  auto zero = GetConstantOrSplat(builder, op.getType(),
                                 mlir::APInt::getZero(int_type.getWidth()));
  mlir::Value safe_div = mlir::mhlo::impl::makeSafeIntDiv<mlir::arith::RemUIOp,
                                                          mlir::arith::RemSIOp>(
      builder, is_unsigned, op.getLhs(), op.getRhs(),
      /*returnedOnZero=*/op.getLhs(),
      /*returnedOnSignedOverflow=*/zero);

  rewriter.replaceOp(op, safe_div);
  return mlir::success();
}

class SafeIntegerArithmeticPass
    : public impl::SafeIntegerArithmeticPassBase<SafeIntegerArithmeticPass> {
 public:
  using SafeIntegerArithmeticPassBase::SafeIntegerArithmeticPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.add(RewriteToSafeDiv<mlir::arith::DivUIOp, /*is_unsigned=*/true>);
    patterns.add(RewriteToSafeDiv<mlir::arith::DivSIOp, /*is_unsigned=*/false>);

    patterns.add(RewriteToSafeRem<mlir::arith::RemUIOp, /*is_unsigned=*/true>);
    patterns.add(RewriteToSafeRem<mlir::arith::RemSIOp, /*is_unsigned=*/false>);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSafeIntegerArithmeticPass() {
  return std::make_unique<SafeIntegerArithmeticPass>();
}

}  // namespace xla::emitters
