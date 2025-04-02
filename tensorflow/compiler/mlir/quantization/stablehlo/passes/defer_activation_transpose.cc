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
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/permutation.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_DEFERACTIVATIONTRANSPOSEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using ::mlir::stablehlo::AddOp;
using ::mlir::stablehlo::BroadcastInDimOp;
using ::mlir::stablehlo::MaxOp;
using ::mlir::stablehlo::TransposeOp;

// Returns `success()` if `op` is a `TransposeOp` with permutation attribute
// equivalent to `permuation`.
LogicalResult IsTransposeOpWithPermuation(Operation* /*absl_nullable*/ op,
                                          const ArrayRef<int64_t> permutation) {
  auto transpose_op = dyn_cast_or_null<TransposeOp>(op);
  return success(transpose_op != nullptr && transpose_op.getPermutation() ==
                                                ArrayRef<int64_t>(permutation));
}

// Convenience function to create a `TransposeOp` with a given `permutation`.
// The Location is set as `input`'s loc.
TransposeOp CreateTransposeOp(Value input, const ArrayRef<int64_t> permutation,
                              PatternRewriter& rewriter) {
  return rewriter.create<TransposeOp>(
      input.getLoc(), input, rewriter.getDenseI64ArrayAttr(permutation));
}

// Defers the transpose of the left-hand side (LHS) to the right-hand side and
// the result of a binary operation. In detail, this rewrites the
// `op(transpose(%rhs), %lhs)` to `transpose(op(%rhs, transpose(%lhs)))`. The
// LHS transpose permutation must be a NCHW->NHWC permutation.
template <typename OpT>
void DeferRhsTransposeForBinaryOp(OpT op, PatternRewriter& rewriter) {
  auto transpose_op = cast<TransposeOp>(op.getOperand(0).getDefiningOp());
  Value lhs_pre_transpose = transpose_op.getOperand();

  // NCHW -> NHWC for the right-hand side, to match the operand's shape.
  Value rhs = op.getOperand(1);
  TransposeOp rhs_transpose_op = CreateTransposeOp(
      /*input=*/rhs, kNchwToNhwcPermutation, rewriter);

  auto new_binary_op =
      rewriter.create<OpT>(op.getLoc(), lhs_pre_transpose, rhs_transpose_op);

  // NHWC -> NCHW for the output, to match the shapes of `op`'s users.
  TransposeOp output_transpose_op = CreateTransposeOp(
      /*input=*/new_binary_op, kNhwcToNchwPermutation, rewriter);

  rewriter.replaceAllUsesWith(op.getResult(), output_transpose_op);
}

// "Climbs up" the `op` if `op` is a `BraodcastInDimOp` and returns the defining
// op of its operand. Returns `op` otherwise. May return `nullptr` when the
// `BroadcastInDimOp`'s operand is a block argument.
Operation* /*absl_nullable*/ SkipUpwardsOptionalBroadcastInDimOp(
    Operation* /*absl_nonnull*/ op) {
  if (auto broadcast_in_dim_op = dyn_cast_or_null<BroadcastInDimOp>(op);
      broadcast_in_dim_op != nullptr) {
    return broadcast_in_dim_op.getOperand().getDefiningOp();
  }
  return op;
}

class DeferActivationTransposeForAddOp : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    // Only supports the case for 2D convolution.
    const Value lhs = op.getOperand(0);
    if (!HasRankOf(lhs, /*rank=*/4)) return failure();

    const Value rhs = op.getOperand(1);
    Operation* rhs_op = rhs.getDefiningOp();
    if (rhs_op == nullptr) return failure();

    // Ignore the optional `BroadcastInDimOp` in between the constant and RHS.
    rhs_op = SkipUpwardsOptionalBroadcastInDimOp(rhs_op);

    if (rhs_op == nullptr || !rhs_op->hasTrait<OpTrait::ConstantLike>()) {
      return failure();
    }

    // Match LHS permutation that converts: NHWC -> NCHW.
    if (IsTransposeOpWithPermuation(lhs.getDefiningOp(), kNhwcToNchwPermutation)
            .failed()) {
      return failure();
    }

    DeferRhsTransposeForBinaryOp(op, rewriter);
    return success();
  }
};

// Rewrites the `reduce_window(transpose(%activation), %init_value)` patterns to
// `transpose(reduce_window(%activation), %init_value)`, deferring the transpose
// to the result. The reduce function should be equivalent to
// `stablehlo.maximum`, representing max pooling.
class DeferActivationTransposeForMaxPoolReduceWindowOp
    : public OpRewritePattern<mlir::stablehlo::ReduceWindowOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceWindowOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(MatchMaxPoolReduceWindowOp(op))) return failure();

    // Match only when the lhs is connected to a transpose.
    // Only supports the case commonly appearing for 2D convolutions.
    Value lhs = op.getOperand(0);
    if (!HasRankOf(lhs, /*rank=*/4)) return failure();

    // Match input permutation that converts: NHWC -> NCHW.
    if (IsTransposeOpWithPermuation(lhs.getDefiningOp(), kNhwcToNchwPermutation)
            .failed()) {
      return failure();
    }

    // Pushes the transpose op at the input to the result.
    auto transpose_op = cast<TransposeOp>(op.getOperand(0).getDefiningOp());

    const auto result_type = mlir::cast<TensorType>(op.getResult(0).getType());
    const SmallVector<int64_t> new_result_shape =
        Permute<int64_t>(result_type.getShape(), kNchwToNhwcPermutation);

    const TensorType new_result_type =
        result_type.cloneWith(new_result_shape, result_type.getElementType());

    // Create a new `stablehlo.reduce_window` with all relevant attributes
    // permutated to match the new operand & result type.
    auto new_reduce_window_op =
        rewriter.create<mlir::stablehlo::ReduceWindowOp>(
            op.getLoc(), new_result_type, transpose_op.getOperand(),
            /*init_value=*/op.getOperand(1),
            /*window_dimensions=*/
            PermuteI64ArrayAttr(rewriter, op.getWindowDimensions(),
                                kNchwToNhwcPermutation),
            /*window_strides=*/
            PermuteI64ArrayAttr(rewriter, op.getWindowStrides(),
                                kNchwToNhwcPermutation),
            /*base_dilations=*/
            PermuteI64ArrayAttr(rewriter, op.getBaseDilations(),
                                kNchwToNhwcPermutation),
            /*window_dilations=*/
            PermuteI64ArrayAttr(rewriter, op.getWindowDilations(),
                                kNchwToNhwcPermutation),
            /*padding=*/DenseIntElementsAttr(nullptr));

    // Clone the reduce body. It is not affected by the permutation.
    IRMapping mapping;
    op.getBody().cloneInto(&new_reduce_window_op.getBody(), mapping);

    // Introduce a transpose to the result to match the shapes of `op`'s uses.
    TransposeOp result_transpose_op = CreateTransposeOp(
        /*input=*/new_reduce_window_op.getResult(0), kNhwcToNchwPermutation,
        rewriter);

    rewriter.replaceAllUsesWith(op.getResult(0), result_transpose_op);
    return success();
  }

 private:
  // Permutes `array_attr` with `permutation`. The number of elements in
  // `array_attr` and `permutation` must be equal. Returns a null attribute
  // if `array_attr` is null.
  DenseI64ArrayAttr PermuteI64ArrayAttr(
      PatternRewriter& rewriter,
      const std::optional<ArrayRef<int64_t>> array_attr,
      const ArrayRef<int64_t> permutation) const {
    if (!array_attr.has_value()) return DenseI64ArrayAttr(nullptr);

    return rewriter.getDenseI64ArrayAttr(
        Permute<int64_t>(array_attr.value(), permutation));
  }

  LogicalResult MatchMaxPoolReduceWindowOp(
      mlir::stablehlo::ReduceWindowOp op) const {
    // TODO: b/321099943 - Support explicit padding.
    if (HasPadding(op)) return failure();

    // Check that the reduce-window body is a max operation.
    return success(IsMaxFunction(op.getBody().front()));
  }

  // Whether `block` semantically corresponds to a `stablehlo.maximum` op.
  bool IsMaxFunction(Block& block) const {
    if (block.getNumArguments() != 2) return false;

    auto return_op = cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
    if (return_op.getNumOperands() != 1) return false;

    auto max_op = dyn_cast_or_null<MaxOp>(
        return_op.getOperands().front().getDefiningOp());
    if (!max_op) return false;

    return (max_op.getLhs() == block.getArgument(0)) &&
           (max_op.getRhs() == block.getArgument(1));
  }

  // Whether `op` has the `padding` attribute (which is optional).
  bool HasPadding(mlir::stablehlo::ReduceWindowOp op) const {
    return op.getPadding() != std::nullopt;
  }
};

// Rewrites `maximum(transpose(%rhs), %lhs)` patterns to
// `transpose(maximum(%rhs, transpose(%lhs)))`.
class DeferActivationTransposeForMaxOp : public OpRewritePattern<MaxOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getOperand(0);
    if (!HasRankOf(input, /*rank=*/4)) return failure();

    const Value max_value = op.getOperand(1);
    Operation* max_value_op = max_value.getDefiningOp();
    if (max_value_op == nullptr ||
        !max_value_op->hasTrait<OpTrait::ConstantLike>()) {
      return failure();
    }

    if (IsTransposeOpWithPermuation(input.getDefiningOp(),
                                    kNhwcToNchwPermutation)
            .failed()) {
      return failure();
    }
    DeferRhsTransposeForBinaryOp(op, rewriter);
    return success();
  }
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
  patterns.add<DeferActivationTransposeForAddOp,
               DeferActivationTransposeForMaxPoolReduceWindowOp,
               DeferActivationTransposeForMaxOp>(&ctx);
  if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
    func_op->emitWarning() << "Failed to converge patterns: " << getArgument();
  }
}

}  // namespace mlir::quant::stablehlo
