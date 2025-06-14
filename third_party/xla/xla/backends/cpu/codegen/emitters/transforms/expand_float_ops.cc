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
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/implicit_arith_op_builder.h"

namespace xla::cpu {

#define GEN_PASS_DECL_EXPANDFLOATOPSPASS
#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

// Emit a f32 to bf16 conversion.
// This has no special logic for nans but it works correctly for silent nans
// (exponent all high & msb of fraction set high) - which are the only ones that
// LLVM really supports anyway.
// We don't want to add any explicit checks for nan as that would result in a
// select instruction which makes auto-vectorization much harder, when we
// implement vectorization at the mlir level we can revisit this.
// See Eigen::BFLoat16.h (float_to_bfloat16_rtne) for more details.
mlir::Value EmitF32ToBF16(mlir::Value in, mlir::ImplicitLocOpBuilder& b) {
  emitters::ImplicitArithOpBuilder i32{
      b.create<mlir::arith::BitcastOp>(b.getI32Type(), in), &b};
  // Round to nearest - even on tie.
  // (Could depend on arith::RoundingModeAttr if desired)
  emitters::ImplicitArithOpBuilder lsb = i32 >> 16 & 1;
  emitters::ImplicitArithOpBuilder rounding_bias = lsb + 0x7fff;
  emitters::ImplicitArithOpBuilder unbiased_i32 = i32 + rounding_bias;
  mlir::Value i16 = b.create<ma::TruncIOp>(b.getI16Type(), unbiased_i32 >> 16);
  return b.create<ma::BitcastOp>(b.getType<mlir::BFloat16Type>(), i16);
}

mlir::Value EmitBF16ToF32(mlir::Value in, mlir::ImplicitLocOpBuilder& b) {
  mlir::Value i16 = b.create<ma::BitcastOp>(b.getI16Type(), in);
  emitters::ImplicitArithOpBuilder i32(
      b.create<ma::ExtUIOp>(b.getI32Type(), i16), &b);
  return b.create<ma::BitcastOp>(b.getType<mlir::Float32Type>(), i32 << 16);
}

struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::Float32Type>(src.getType()) &&
        mlir::isa<mlir::BFloat16Type>(dst_ty)) {
      rewriter.replaceOp(op, EmitF32ToBF16(src, builder));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Not f32 -> bf16");
  }
};

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::BFloat16Type>(src.getType()) &&
        mlir::isa<mlir::Float32Type>(dst_ty)) {
      rewriter.replaceOp(op, EmitBF16ToF32(src, builder));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Not bf16 -> f32");
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteTruncFPattern, RewriteExtFPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass() {
  return std::make_unique<ExpandFloatOpsPass>();
}

}  // namespace xla::cpu
