/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO dialect to Affine dialect.

#include "absl/memory/memory.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace xla_lhlo {
namespace {

// Builds an affine loop nest iterating from zeros to "upper_bounds" with unit
// steps, and populates the body of the innermost loop using "body_builder".
static void BuildBoundedAffineLoopNest(
    OpBuilder& builder, Location location, ArrayRef<int64_t> upper_bounds,
    function_ref<void(OpBuilder&, Location, ValueRange)> body_builder) {
  SmallVector<int64_t, 3> lower_bounds(upper_bounds.size(), /*Value=*/0);
  SmallVector<int64_t, 3> steps(upper_bounds.size(), /*Value=*/1);
  buildAffineLoopNest(builder, location, lower_bounds, upper_bounds, steps,
                      body_builder);
}

struct DotOpConverter : public OpRewritePattern<DotOp> {
  using OpRewritePattern<DotOp>::OpRewritePattern;

  // Supports only rank-2 tensors for LHS and RHS.
  LogicalResult matchAndRewrite(DotOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    MemRefType lhs_type = lhs.getType().cast<MemRefType>();
    MemRefType rhs_type = rhs.getType().cast<MemRefType>();
    Type element_type = lhs_type.getElementType();
    ArrayRef<int64_t> shape_lhs = lhs_type.getShape();
    ArrayRef<int64_t> shape_rhs = rhs_type.getShape();

    if ((lhs_type.getRank() != 2) || (rhs_type.getRank() != 2)) {
      return failure();
    }

    LogicalResult map_status = success();
    auto body_builder = [&](OpBuilder& builder, Location loc, ValueRange ivs) {
      SmallVector<Value, 2> lhs_indices{ivs[0], ivs[2]},
          rhs_indices{ivs[2], ivs[1]}, result_indices{ivs[0], ivs[1]};

      auto l = builder.create<AffineLoadOp>(loc, lhs, lhs_indices);
      auto r = builder.create<AffineLoadOp>(loc, rhs, rhs_indices);
      auto result =
          rewriter.create<AffineLoadOp>(loc, op.output(), result_indices);
      Value op_result = xla_lhlo::XlaOpToStdScalarOp::map<DotOp>(
          op, element_type, {l, r, result}, &builder);
      map_status = success(op_result != nullptr);
      if (failed(map_status)) return;
      builder.create<AffineStoreOp>(loc, op_result, op.output(),
                                    result_indices);
    };

    BuildBoundedAffineLoopNest(rewriter, op.getLoc(),
                               {shape_lhs[0], shape_rhs[1], shape_rhs[0]},
                               body_builder);
    if (failed(map_status)) return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LhloOpTy>
struct BinaryOpConverter : public OpRewritePattern<LhloOpTy> {
  using OpRewritePattern<LhloOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LhloOpTy op,
                                PatternRewriter& rewriter) const override {
    const auto& lhs = op.lhs();
    const auto& rhs = op.rhs();
    const auto& lhs_type = lhs.getType().template cast<MemRefType>();
    const auto& rhs_type = rhs.getType().template cast<MemRefType>();
    const auto& element_type = lhs_type.getElementType();

    if (lhs_type.getShape() != rhs_type.getShape()) {
      return failure();
    }

    LogicalResult map_status = success();
    auto body_builder = [&](OpBuilder& builder, Location loc,
                            ValueRange induction_vars) {
      auto l = builder.create<AffineLoadOp>(loc, lhs, induction_vars);
      auto r = builder.create<AffineLoadOp>(loc, rhs, induction_vars);
      Value op_result = xla_lhlo::XlaOpToStdScalarOp::map<LhloOpTy>(
          op, element_type, {l, r}, &builder);
      map_status = success(op_result != nullptr);
      if (failed(map_status)) return;
      rewriter.create<AffineStoreOp>(loc, op_result, op.out(), induction_vars);
    };

    BuildBoundedAffineLoopNest(rewriter, op.getLoc(), lhs_type.getShape(),
                               body_builder);
    if (failed(map_status)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLHLOToAffineConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      BinaryOpConverter<xla_lhlo::AddOp>,
      BinaryOpConverter<xla_lhlo::AndOp>,
      BinaryOpConverter<xla_lhlo::DivOp>,
      BinaryOpConverter<xla_lhlo::MaxOp>,
      BinaryOpConverter<xla_lhlo::MinOp>,
      BinaryOpConverter<xla_lhlo::MulOp>,
      BinaryOpConverter<xla_lhlo::SubOp>,
      DotOpConverter>(context);
  // clang-format on
}

struct LhloLegalizeToAffine
    : public PassWrapper<LhloLegalizeToAffine, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateLHLOToAffineConversionPattern(func.getContext(), &patterns);
    applyPatternsAndFoldGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLegalizeToAffinePass() {
  return absl::make_unique<LhloLegalizeToAffine>();
}

static PassRegistration<LhloLegalizeToAffine> legalize_pass(
    "lhlo-legalize-to-affine", "Legalize from LHLO dialect to affine dialect");

}  // namespace xla_lhlo
}  // namespace mlir
