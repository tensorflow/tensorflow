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
#include "mlir/Dialect/AffineOps/AffineOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_lhlo_to_scalar_op.h"

namespace mlir {
namespace xla_lhlo {
namespace {

template <typename LhloOp>
struct BinaryOpConverter : public OpRewritePattern<LhloOp> {
  using OpRewritePattern<LhloOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(LhloOp op,
                                     PatternRewriter& rewriter) const override {
    const auto& lhs = op.lhs();
    const auto& rhs = op.rhs();
    const auto& lhs_type = lhs->getType().template cast<MemRefType>();
    const auto& rhs_type = rhs->getType().template cast<MemRefType>();
    const auto& element_type = lhs_type.getElementType();

    if (lhs_type.getShape() != rhs_type.getShape()) {
      return this->matchFailure();
    }
    const auto& shape = lhs_type.getShape();
    SmallVector<Value, 4> induction_vars;
    const auto loc = op.getLoc();
    for (int i = 0; i < shape.size(); ++i) {
      auto forOp = rewriter.create<AffineForOp>(loc, 0, shape[i]);
      induction_vars.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    auto l = rewriter.create<LoadOp>(loc, lhs, induction_vars);
    auto r = rewriter.create<LoadOp>(loc, rhs, induction_vars);
    Operation* result = MapLhloOpToStdScalarOp<LhloOp>(
        llvm::cast<LhloOp>(op), element_type, {l, r}, rewriter);
    if (result == nullptr) {
      return this->matchFailure();
    }
    rewriter.create<StoreOp>(loc, result->getResult(0), op.out(),
                             induction_vars);
    rewriter.eraseOp(op);
    return this->matchSuccess();
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
      BinaryOpConverter<xla_lhlo::SubOp>>(context);
  // clang-format on
}

struct LhloLegalizeToAffine : public FunctionPass<LhloLegalizeToAffine> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateLHLOToAffineConversionPattern(func.getContext(), &patterns);
    applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToAffinePass() {
  return absl::make_unique<LhloLegalizeToAffine>();
}

static PassRegistration<LhloLegalizeToAffine> legalize_pass(
    "lhlo-legalize-to-affine", "Legalize from LHLO dialect to affine dialect");

}  // namespace xla_lhlo
}  // namespace mlir
