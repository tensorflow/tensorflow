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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xla::cpu {

#define GEN_PASS_DECL_EXPANDFLOATOPSPASS
#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

// Get a constant value, if the type is a vector, splat the value to the vector
// type.
mlir::Value GetConst(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                     mlir::TypedAttr value) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    value =
        mlir::SplatElementsAttr::get(mlir::cast<mlir::ShapedType>(type), value);
  }
  return mlir::arith::ConstantOp::create(b, type, value);
}

mlir::Value EmitBF16ToF32(mlir::Type dst_ty, mlir::Value in,
                          mlir::ImplicitLocOpBuilder& b) {
  auto get_type = [&](mlir::Type element_type) -> mlir::Type {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(in.getType())) {
      return vector_type.clone(element_type);
    }
    return element_type;
  };

  mlir::Type i16_type = get_type(b.getI16Type());
  mlir::Type i32_type = get_type(b.getI32Type());

  mlir::Value i16 = ma::BitcastOp::create(b, i16_type, in);
  mlir::Value i32 = ma::ExtUIOp::create(b, i32_type, i16);

  mlir::TypedAttr shift_value = b.getI32IntegerAttr(16);
  mlir::Value shift_const = GetConst(b, i32_type, shift_value);

  mlir::Value i32_shl = mlir::arith::ShLIOp::create(b, i32, shift_const);
  return ma::BitcastOp::create(b, dst_ty, i32_shl);
}

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = op.getType();

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::BFloat16Type>(
            mlir::getElementTypeOrSelf(src.getType())) &&
        mlir::isa<mlir::Float32Type>(mlir::getElementTypeOrSelf(dst_ty))) {
      rewriter.replaceOp(op, EmitBF16ToF32(dst_ty, src, builder));
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
    patterns.add<RewriteExtFPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
