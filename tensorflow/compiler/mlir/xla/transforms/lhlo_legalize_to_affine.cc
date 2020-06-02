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
#include "mlir/Dialect/StandardOps/IR/Ops.h"   // from @llvm-project
#include "mlir/IR/Attributes.h"                // from @llvm-project
#include "mlir/IR/Location.h"                  // from @llvm-project
#include "mlir/IR/MLIRContext.h"               // from @llvm-project
#include "mlir/IR/PatternMatch.h"              // from @llvm-project
#include "mlir/IR/StandardTypes.h"             // from @llvm-project
#include "mlir/Pass/Pass.h"                    // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace xla_lhlo {
namespace {

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
    SmallVector<Value, 4> lhs_indices, rhs_indices, result_indices;
    const auto& loc = op.getLoc();

    // Create the canonical ijk form of matmul.
    auto forOp = rewriter.create<AffineForOp>(loc, 0, shape_lhs[0]);
    lhs_indices.push_back(forOp.getInductionVar());
    result_indices.push_back(forOp.getInductionVar());

    rewriter.setInsertionPointToStart(forOp.getBody());
    forOp = rewriter.create<AffineForOp>(loc, 0, shape_rhs.back());
    result_indices.push_back(forOp.getInductionVar());
    rhs_indices.resize(2);
    rhs_indices[1] = forOp.getInductionVar();

    rewriter.setInsertionPointToStart(forOp.getBody());
    forOp = rewriter.create<AffineForOp>(loc, 0, shape_rhs.front());
    lhs_indices.push_back(forOp.getInductionVar());
    rhs_indices[0] = forOp.getInductionVar();

    // Construct the innermost loop body.
    rewriter.setInsertionPointToStart(forOp.getBody());
    auto l = rewriter.create<AffineLoadOp>(loc, lhs, lhs_indices);
    auto r = rewriter.create<AffineLoadOp>(loc, rhs, rhs_indices);
    auto result =
        rewriter.create<AffineLoadOp>(loc, op.output(), result_indices);
    Value op_result = xla_lhlo::XlaOpToStdScalarOp::map<DotOp>(
        op, element_type, {l, r, result}, &rewriter);
    if (op_result == nullptr) {
      return failure();
    }
    rewriter.create<AffineStoreOp>(loc, op_result, op.output(), result_indices);
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
    const auto& shape = lhs_type.getShape();
    SmallVector<Value, 4> induction_vars;
    const auto loc = op.getLoc();
    for (int i = 0; i < shape.size(); ++i) {
      auto forOp = rewriter.create<AffineForOp>(loc, 0, shape[i]);
      induction_vars.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    auto l = rewriter.create<AffineLoadOp>(loc, lhs, induction_vars);
    auto r = rewriter.create<AffineLoadOp>(loc, rhs, induction_vars);
    Value opResult = xla_lhlo::XlaOpToStdScalarOp::map<LhloOpTy>(
        op, element_type, {l, r}, &rewriter);
    if (opResult == nullptr) {
      return failure();
    }
    rewriter.create<AffineStoreOp>(loc, opResult, op.out(), induction_vars);
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
