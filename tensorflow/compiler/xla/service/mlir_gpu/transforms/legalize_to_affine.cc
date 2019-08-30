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

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::AffineForOp;
using ::mlir::CmpFOp;
using ::mlir::CmpFPredicate;
using ::mlir::CmpIOp;
using ::mlir::CmpIPredicate;
using ::mlir::FloatType;
using ::mlir::IntegerType;
using ::mlir::MLIRContext;
using ::mlir::Operation;
using ::mlir::OwningRewritePatternList;
using ::mlir::PatternRewriter;
using ::mlir::Value;

namespace lhlo = ::mlir::xla_lhlo;

template <typename LHLO_BinaryOp>
struct ScalarOp;

template <>
struct ScalarOp<lhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<lhlo::AndOp> {
  using IOp = ::mlir::AndOp;
};
template <>
struct ScalarOp<lhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::DivISOp;
};
template <>
struct ScalarOp<lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};
template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename LHLO_BinaryOp>
Value* GetBinaryOp(::mlir::Type element_type, ::mlir::Location loc, Value* lhs,
                   Value* rhs, ::mlir::OpBuilder b) {
  if (element_type.isa<::mlir::IntegerType>()) {
    return b.create<ScalarIOp<LHLO_BinaryOp>>(loc, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    return b.create<ScalarFOp<LHLO_BinaryOp>>(loc, lhs, rhs);
  }
  return nullptr;
}

template <>
Value* GetBinaryOp<lhlo::AndOp>(::mlir::Type element_type, ::mlir::Location loc,
                                Value* lhs, Value* rhs, ::mlir::OpBuilder b) {
  return element_type.isa<::mlir::IntegerType>()
             ? b.create<ScalarIOp<lhlo::AndOp>>(loc, lhs, rhs)
             : nullptr;
}

template <>
Value* GetBinaryOp<lhlo::MinOp>(::mlir::Type element_type, ::mlir::Location loc,
                                Value* lhs, Value* rhs, ::mlir::OpBuilder b) {
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
Value* GetBinaryOp<lhlo::MaxOp>(::mlir::Type element_type, ::mlir::Location loc,
                                Value* lhs, Value* rhs, ::mlir::OpBuilder b) {
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
struct BinaryOpConverter : public ::mlir::RewritePattern {
  explicit BinaryOpConverter(const string& opname, ::mlir::MLIRContext* context)
      : RewritePattern(opname, {}, 1, context), opname(opname) {}

  ::mlir::PatternMatchResult matchAndRewrite(
      Operation* op, PatternRewriter& rewriter) const override {
    auto binary_op = ::mlir::cast<LhloOp>(op);

    const auto& lhs = binary_op.lhs();
    const auto& rhs = binary_op.rhs();
    const auto& lhs_type = lhs->getType().template cast<::mlir::MemRefType>();
    const auto& rhs_type = rhs->getType().template cast<::mlir::MemRefType>();
    const auto& element_type = lhs_type.getElementType();

    if (lhs_type.getShape() != rhs_type.getShape()) {
      return matchFailure();
    }
    const auto& shape = lhs_type.getShape();
    ::mlir::SmallVector<::mlir::Value*, 4> induction_vars;
    const auto loc = rewriter.getUnknownLoc();
    for (int i = 0; i < shape.size(); ++i) {
      auto forOp = rewriter.create<AffineForOp>(loc, 0, shape[i]);
      induction_vars.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    auto l = rewriter.create<::mlir::LoadOp>(loc, lhs, induction_vars);
    auto r = rewriter.create<::mlir::LoadOp>(loc, rhs, induction_vars);
    auto result = GetBinaryOp<LhloOp>(element_type, loc, l, r, rewriter);
    if (result == nullptr) {
      return matchFailure();
    }
    rewriter.create<::mlir::StoreOp>(loc, result, binary_op.out(),
                                     induction_vars);
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
  const std::string opname;
};

void AppendBinaryOpsPatterns(MLIRContext* context,
                             OwningRewritePatternList* patterns) {
  patterns->insert<BinaryOpConverter<lhlo::AddOp>>("xla_lhlo.add", context);
  patterns->insert<BinaryOpConverter<lhlo::AndOp>>("xla_lhlo.and", context);
  patterns->insert<BinaryOpConverter<lhlo::DivOp>>("xla_lhlo.div", context);
  patterns->insert<BinaryOpConverter<lhlo::MaxOp>>("xla_lhlo.max", context);
  patterns->insert<BinaryOpConverter<lhlo::MinOp>>("xla_lhlo.min", context);
  patterns->insert<BinaryOpConverter<lhlo::MulOp>>("xla_lhlo.mul", context);
  patterns->insert<BinaryOpConverter<lhlo::SubOp>>("xla_lhlo.sub", context);
}

struct LegalizeToAffine : public ::mlir::FunctionPass<LegalizeToAffine> {
  void runOnFunction() override {
    ::mlir::OwningRewritePatternList patterns;
    auto func = getFunction();
    AppendBinaryOpsPatterns(func.getContext(), &patterns);
    applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPassBase> createLegalizeAffinePass() {
  return absl::make_unique<LegalizeToAffine>();
}

static ::mlir::PassRegistration<LegalizeToAffine> legalize_pass(
    "lhlo-legalize-to-affine", "Legalize from LHLO dialect to affine dialect");

}  // namespace mlir_gpu
}  // namespace xla
