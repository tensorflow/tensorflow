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

#include <iterator>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_XTILELOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

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

class LowerDotScaled
    : public mlir::OpRewritePattern<::xla::xtile::DotScaledOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::DotScaledOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (std::distance(op->getUsers().begin(), op->getUsers().end()) != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dot op must have exactly one user in order to be lowered to "
          "triton.");
    }

    mlir::Operation* add_op = dyn_cast<arith::AddFOp>(*op->getUsers().begin());
    if (!add_op) {
      add_op = dyn_cast<arith::AddIOp>(*op->getUsers().begin());
    }

    if (!add_op) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dot op must be consumed by an AddOp in order to be convertible to "
          "triton dot.");
    }

    // Accumulator is the operand of add that is not the dot operation.
    auto accumulator = add_op->getOperand(1) == op ? add_op->getOperand(0)
                                                   : add_op->getOperand(1);

    auto lhs_dot_elem_type_or_status =
        GetScaleDotElemType(op.getLhs().getType());
    auto rhs_dot_elem_type_or_status =
        GetScaleDotElemType(op.getRhs().getType());

    if (!lhs_dot_elem_type_or_status.ok() ||
        !rhs_dot_elem_type_or_status.ok()) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          absl::StrCat(
              "Failed to get dot element type for lhs or rhs.\nLhs status: ",
              lhs_dot_elem_type_or_status.status().message(), "\nRhs status: ",
              rhs_dot_elem_type_or_status.status().message()));
    }

    auto lhs_dot_elem_type = lhs_dot_elem_type_or_status.value();
    auto rhs_dot_elem_type = rhs_dot_elem_type_or_status.value();

    auto triton_dot_scaled_op = ttir::DotScaledOp::create(
        rewriter, op.getLoc(), accumulator.getType(), op.getLhs(), op.getRhs(),
        accumulator, op.getLhsScale(), op.getRhsScale(), lhs_dot_elem_type,
        rhs_dot_elem_type, op.getFastMath(), op.getLhsKPack(),
        op.getRhsKPack());

    rewriter.replaceAllOpUsesWith(add_op, op.getResult());
    rewriter.replaceOp(op, triton_dot_scaled_op);
    return mlir::success();
  }
};

class XTileLowerToTritonPass
    : public impl::XTileLowerToTritonPassBase<XTileLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerDotScaled>(mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateXTileLowerToTritonPass() {
  return std::make_unique<XTileLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
