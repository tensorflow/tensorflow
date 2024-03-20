/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <array>
#include <cstdint>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_DEFERACTIVATIONTRANSPOSEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using ::mlir::stablehlo::AddOp;
using ::mlir::stablehlo::TransposeOp;

class RewriteAddWithActivationTranspose : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult match(AddOp op) const override {
    // Only supports the case for 2D convolution.
    const Value lhs = op.getOperand(0);
    if (!HasRankOf(lhs, /*rank=*/4)) return failure();

    const Value rhs = op.getOperand(1);
    Operation* rhs_op = rhs.getDefiningOp();
    if (rhs_op == nullptr || !rhs_op->hasTrait<OpTrait::ConstantLike>()) {
      return failure();
    }

    // Match LHS permutation that converts: NHWC -> NCHW.
    auto transpose_op = dyn_cast_or_null<TransposeOp>(lhs.getDefiningOp());
    if (transpose_op == nullptr) {
      return failure();
    }

    return success(transpose_op.getPermutation() ==
                   ArrayRef<int64_t>(kDesiredLhsPermutation));
  }

  void rewrite(AddOp op, PatternRewriter& rewriter) const override {
    auto lhs_transpose_op = cast<TransposeOp>(op.getOperand(0).getDefiningOp());
    Value lhs_input = lhs_transpose_op.getOperand();

    Value rhs_input = op.getOperand(1);

    // NCHW -> NHWC for the right-hand side, to match the operand's shape.
    auto rhs_transpose_op = rewriter.create<TransposeOp>(
        op.getLoc(), /*operand=*/rhs_input,
        rewriter.getDenseI64ArrayAttr(kRhsPermutation));

    auto add_op =
        rewriter.create<AddOp>(op.getLoc(), lhs_input, rhs_transpose_op);

    // NHWC -> NCHW for the output, to match the shapes of `op`'s users.
    auto output_transpose_op = rewriter.create<TransposeOp>(
        op.getLoc(), /*operand=*/add_op.getResult(),
        rewriter.getDenseI64ArrayAttr(kOutputPermutation));

    rewriter.replaceAllUsesWith(op.getResult(), output_transpose_op);
  }

 private:
  // Permutation representing NHWC -> NCHW for the activation (LHS), used for
  // matching the pattern.
  static constexpr std::array<int64_t, 4> kDesiredLhsPermutation = {0, 3, 1, 2};

  // Permutation representing NCHW -> NHWC for the RHS, newly inserted after the
  // conversion.
  static constexpr std::array<int64_t, 4> kRhsPermutation = {0, 2, 3, 1};
  // Permutation representing NHWC -> NCHW for the output, newly inserted after
  // the conversion.
  static constexpr std::array<int64_t, 4> kOutputPermutation = {0, 3, 1, 2};
};

}  // namespace

class DeferActivationTransposePass
    : public impl::DeferActivationTransposePassBase<
          DeferActivationTransposePass> {
 private:
  void runOnOperation() override;
};

void DeferActivationTransposePass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<RewriteAddWithActivationTranspose>(&ctx);
  if (failed(applyPatternsAndFoldGreedily(func_op, std::move(patterns)))) {
    func_op->emitWarning() << "Failed to converge patterns: " << getArgument();
  }
}

}  // namespace mlir::quant::stablehlo
