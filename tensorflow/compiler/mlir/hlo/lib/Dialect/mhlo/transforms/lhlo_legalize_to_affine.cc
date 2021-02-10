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

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {
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

    // We don't currently support batching dimensions, or multiple contraction
    // dimensions.
    mhlo::DotDimensionNumbers dot_dimension_numbers =
        op.dot_dimension_numbers();
    if (dot_dimension_numbers.lhs_batching_dimensions().size() > 0 ||
        dot_dimension_numbers.rhs_batching_dimensions().size() > 0)
      return failure();
    if (dot_dimension_numbers.lhs_contracting_dimensions().size() != 1 ||
        *dot_dimension_numbers.lhs_contracting_dimensions().begin() != 1 ||
        dot_dimension_numbers.rhs_contracting_dimensions().size() != 1 ||
        *dot_dimension_numbers.rhs_contracting_dimensions().begin() != 0) {
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
      Value op_result = lmhlo::HloOpToStdScalarOp::map<DotOp>(
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
      Value op_result = lmhlo::HloOpToStdScalarOp::map<LhloOpTy>(
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
      BinaryOpConverter<lmhlo::AddOp>,
      BinaryOpConverter<lmhlo::AndOp>,
      BinaryOpConverter<lmhlo::DivOp>,
      BinaryOpConverter<lmhlo::MaxOp>,
      BinaryOpConverter<lmhlo::MinOp>,
      BinaryOpConverter<lmhlo::MulOp>,
      BinaryOpConverter<lmhlo::SubOp>,
      DotOpConverter>(context);
  // clang-format on
}

struct LhloLegalizeToAffinePass
    : public PassWrapper<LhloLegalizeToAffinePass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect>();
  }
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateLHLOToAffineConversionPattern(func.getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLhloLegalizeToAffinePass() {
  return std::make_unique<LhloLegalizeToAffinePass>();
}

}  // namespace lmhlo
}  // namespace mlir
