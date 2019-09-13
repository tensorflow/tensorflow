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
#include "mlir/Dialect/AffineOps/AffineOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {
namespace {

template <typename LHLO_BinaryOp>
struct ScalarOp;

template <>
struct ScalarOp<xla_lhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<xla_lhlo::AndOp> {
  using IOp = ::mlir::AndOp;
};
template <>
struct ScalarOp<xla_lhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::DivISOp;
};
template <>
struct ScalarOp<xla_lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<xla_lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};
template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename LHLO_BinaryOp>
Value* GetBinaryOp(Type element_type, Location loc, Value* lhs, Value* rhs,
                   OpBuilder b) {
  if (element_type.isa<IntegerType>()) {
    return b.create<ScalarIOp<LHLO_BinaryOp>>(loc, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    return b.create<ScalarFOp<LHLO_BinaryOp>>(loc, lhs, rhs);
  }
  return nullptr;
}

template <>
Value* GetBinaryOp<xla_lhlo::AndOp>(Type element_type, Location loc, Value* lhs,
                                    Value* rhs, OpBuilder b) {
  return element_type.isa<IntegerType>()
             ? b.create<ScalarIOp<xla_lhlo::AndOp>>(loc, lhs, rhs)
             : nullptr;
}

template <>
Value* GetBinaryOp<xla_lhlo::MinOp>(Type element_type, Location loc, Value* lhs,
                                    Value* rhs, OpBuilder b) {
  if (element_type.isa<IntegerType>()) {
    auto lhs_lt_rhs = b.create<CmpIOp>(loc, CmpIPredicate::SLT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_lt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_lt_rhs = b.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_lt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
Value* GetBinaryOp<xla_lhlo::MaxOp>(Type element_type, Location loc, Value* lhs,
                                    Value* rhs, OpBuilder b) {
  if (element_type.isa<IntegerType>()) {
    auto lhs_gt_rhs = b.create<CmpIOp>(loc, CmpIPredicate::SGT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_gt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_gt_rhs = b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
    return b.create<::mlir::SelectOp>(loc, lhs_gt_rhs, lhs, rhs);
  }
  return nullptr;
}

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
    SmallVector<Value*, 4> induction_vars;
    const auto loc = op.getLoc();
    for (int i = 0; i < shape.size(); ++i) {
      auto forOp = rewriter.create<AffineForOp>(loc, 0, shape[i]);
      induction_vars.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    auto l = rewriter.create<LoadOp>(loc, lhs, induction_vars);
    auto r = rewriter.create<LoadOp>(loc, rhs, induction_vars);
    auto result = GetBinaryOp<LhloOp>(element_type, loc, l, r, rewriter);
    if (result == nullptr) {
      return this->matchFailure();
    }
    rewriter.create<StoreOp>(loc, result, op.out(), induction_vars);
    rewriter.replaceOp(op, {});
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

std::unique_ptr<FunctionPassBase> createLegalizeToAffinePass() {
  return absl::make_unique<LhloLegalizeToAffine>();
}

static PassRegistration<LhloLegalizeToAffine> legalize_pass(
    "lhlo-legalize-to-affine", "Legalize from LHLO dialect to affine dialect");

}  // namespace xla_lhlo
}  // namespace mlir
