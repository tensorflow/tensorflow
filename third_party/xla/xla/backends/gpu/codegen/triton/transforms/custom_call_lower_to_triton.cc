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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_CUSTOMCALLLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

namespace {

absl::StatusOr<ttir::ScaleDotElemType> GetScaleDotElemType(Type value) {
  auto type = getElementTypeOrSelf(value);
  if (type == mlir::Float8E4M3FNType::get(value.getContext())) {
    return ttir::ScaleDotElemType::E4M3;
  }
  if (type == mlir::Float8E5M2Type::get(value.getContext())) {
    return ttir::ScaleDotElemType::E5M2;
  }
  if (type == mlir::Float4E2M1FNType::get(value.getContext())) {
    return ttir::ScaleDotElemType::E2M1;
  }
  if (type == mlir::BFloat16Type::get(value.getContext())) {
    return ttir::ScaleDotElemType::BF16;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported type: ", ::xla::llvm_ir::DumpToString(type)));
}

mlir::Value Bitcast(mlir::PatternRewriter& rewriter, Location loc,
                    mlir::Value value, mlir::Type new_type) {
  auto value_type = value.getType();
  value_type = mlir::dyn_cast<ShapedType>(value_type).clone(new_type);
  return mlir::arith::BitcastOp::create(rewriter, loc, value_type, value);
}

}  // namespace

class CustomScaledDotCallToTriton
    : public mlir::OpRewritePattern<::mlir::func::CallOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      ::mlir::func::CallOp call_op,
      mlir::PatternRewriter& rewriter) const override {
    if (call_op.getCallee().str() !=
        ::xla::gpu::triton::kScaledDotFunctionName) {
      return rewriter.notifyMatchFailure(
          call_op, absl::StrCat("callee attribute is not ",
                                ::xla::gpu::triton::kScaledDotFunctionName,
                                " but ", call_op.getCallee().str()));
    }

    auto operands = call_op.getOperands();
    auto lhs = operands[0];
    auto rhs = operands[1];
    auto acc = operands[2];
    // operands[3] is the lhs_scale, operands[4] is the rhs_scale, but we don't
    // define them here as they are only used conditionally.

    auto lhs_dot_elem_type_or_status = GetScaleDotElemType(lhs.getType());
    auto rhs_dot_elem_type_or_status = GetScaleDotElemType(rhs.getType());

    if (!lhs_dot_elem_type_or_status.ok()) {
      return rewriter.notifyMatchFailure(
          call_op, lhs_dot_elem_type_or_status.status().ToString());
    }
    if (!rhs_dot_elem_type_or_status.ok()) {
      return rewriter.notifyMatchFailure(
          call_op, rhs_dot_elem_type_or_status.status().ToString());
    }

    auto lhs_dot_elem_type = *lhs_dot_elem_type_or_status;
    auto rhs_dot_elem_type = *rhs_dot_elem_type_or_status;

    mlir::Value lhs_scale;
    // make type with the same shape as the scale but with i8 type
    if (lhs_dot_elem_type != ttir::ScaleDotElemType::BF16) {
      lhs_scale = Bitcast(rewriter, call_op.getLoc(), operands[3],
                          rewriter.getI8Type());
    }

    mlir::Value rhs_scale;
    if (rhs_dot_elem_type != ttir::ScaleDotElemType::BF16) {
      rhs_scale = Bitcast(rewriter, call_op.getLoc(), operands[4],
                          rewriter.getI8Type());
      rhs_scale = ttir::TransOp::create(rewriter, call_op.getLoc(), rhs_scale,
                                        mlir::ArrayRef<int32_t>{1, 0});
    }

    bool fast_math =
        call_op->getAttrOfType<mlir::BoolAttr>("fast_math").getValue();

    rewriter.replaceOpWithNewOp<ttir::DotScaledOp>(
        call_op, acc.getType(), lhs, rhs, acc, lhs_scale, rhs_scale,
        lhs_dot_elem_type, rhs_dot_elem_type, fast_math);

    return mlir::success();
  }
};

class CustomCallLowerToTritonPass
    : public impl::CustomCallLowerToTritonPassBase<
          CustomCallLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<CustomScaledDotCallToTriton>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateCustomCallLowerToTritonPass() {
  return std::make_unique<CustomCallLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
