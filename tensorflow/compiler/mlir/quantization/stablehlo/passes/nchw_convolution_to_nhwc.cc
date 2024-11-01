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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/uniform_quantized_types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/permutation.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_NCHWCONVOLUTIONTONHWCPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

using ::mlir::stablehlo::ConvDimensionNumbersAttr;

class NchwConvolutionToNhwcPass
    : public impl::NchwConvolutionToNhwcPassBase<NchwConvolutionToNhwcPass> {
 private:
  void runOnOperation() override;
};

// Rewrites NCHW convolution to NHWC.
// * Src dimension numbers: [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
// * Dst dimension numbers: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
class RewriteNchwConvolutionToNhwc
    : public OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
 public:
  using OpRewritePattern<mlir::stablehlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult match(mlir::stablehlo::ConvolutionOp op) const override {
    // Handles 2D convolutions only.
    if (!HasRankOf(op.getOperand(0), /*rank=*/4) ||
        !HasRankOf(op.getOperand(1), /*rank=*/4)) {
      return failure();
    }

    if (!IsOpNotQuantized(op)) return failure();

    const ConvDimensionNumbersAttr dimension_nums = op.getDimensionNumbers();
    return success(MatchInputDimensionNumbers(dimension_nums) &&
                   MatchKernelDimensionNumbers(dimension_nums) &&
                   MatchOutputDimensionNumbers(dimension_nums));
  }

  void rewrite(mlir::stablehlo::ConvolutionOp op,
               PatternRewriter& rewriter) const override {
    // Transpose the input tensor: [b, f, 0, 1] => [b, 0, 1, f]
    Value input = op->getOperand(0);
    const TensorType new_input_tensor_type = GetTransposedTensorType(
        mlir::cast<TensorType>(input.getType()), kNchwToNhwcPermutation);

    auto input_transpose_op = rewriter.create<mlir::stablehlo::TransposeOp>(
        op.getLoc(), /*resultType0=*/new_input_tensor_type, /*operand=*/input,
        rewriter.getDenseI64ArrayAttr(kNchwToNhwcPermutation));

    // Transpose the filter tensor: [o, i, 0, 1] => [0, 1, i, o]
    Value filter = op->getOperand(1);
    const TensorType new_filter_tensor_type = GetTransposedTensorType(
        mlir::cast<TensorType>(filter.getType()), kOihwToHwioPermutation);

    auto filter_transpose_op = rewriter.create<mlir::stablehlo::TransposeOp>(
        op.getLoc(), /*resultType0=*/new_filter_tensor_type, /*operand=*/filter,
        rewriter.getDenseI64ArrayAttr(kOihwToHwioPermutation));

    // [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
    const auto new_dimension_nums = rewriter.getAttr<ConvDimensionNumbersAttr>(
        /*inputBatchDimension=*/0, /*inputFeatureDimension=*/3,
        /*inputSpatialDimensions=*/SmallVector<int64_t>{1, 2},
        /*kernelInputFeatureDimension=*/2, /*kernelOutputFeatureDimension=*/3,
        /*kernelSpatialDimensions=*/SmallVector<int64_t>{0, 1},
        /*outputBatchDimension=*/0, /*outputFeatureDimension=*/3,
        /*outputSpatialDimensions=*/SmallVector<int64_t>{1, 2});

    // Determine the shape of the output tensor: [b, f, 0, 1] => [b, 0, 1, f]
    auto output_tensor_type =
        mlir::cast<TensorType>(op->getResult(0).getType());
    const TensorType new_conv_output_tensor_type =
        GetTransposedTensorType(output_tensor_type, kNchwToNhwcPermutation);

    // window_strides, padding, lhs_dilation, rhs_dilation, window_reversal are
    // reused without modification because the ordering of spatial dimensions
    // is not modified (i.e. before: [b, f, 0, 1], after: [b, 0, 1, f] => the
    // spatial dimension is still ordered as {0, 1}).
    auto new_convolution_op = rewriter.create<mlir::stablehlo::ConvolutionOp>(
        op.getLoc(), /*resultType0=*/new_conv_output_tensor_type,
        /*lhs=*/input_transpose_op,
        /*rhs=*/filter_transpose_op,
        /*window_strides=*/op.getWindowStridesAttr(),
        /*padding=*/op.getPaddingAttr(),
        /*lhs_dilation=*/op.getLhsDilationAttr(),
        /*rhs_dilation=*/op.getRhsDilationAttr(),
        /*window_reversal=*/op.getWindowReversalAttr(),
        /*dimension_numbers=*/new_dimension_nums,
        /*feature_group_count=*/op.getFeatureGroupCountAttr(),
        /*batch_group_count=*/op.getBatchGroupCountAttr(),
        /*precision_config=*/op.getPrecisionConfigAttr());

    // Transpose the output of the `ConvolutionOp` back to the original op's
    // output shape so that users' shapes match.
    // [b, 0, 1, f] => [b, f, 0, 1]
    auto output_transpose_op = rewriter.create<mlir::stablehlo::TransposeOp>(
        new_convolution_op.getLoc(), /*resultType0=*/output_tensor_type,
        /*operand=*/new_convolution_op,
        rewriter.getDenseI64ArrayAttr(kNhwcToNchwPermutation));

    rewriter.replaceAllUsesWith(op, output_transpose_op);
  }

 private:
  // Matches input dimensions corresponding to: [b, f, 0, 1].
  bool MatchInputDimensionNumbers(
      const ConvDimensionNumbersAttr dimension_numbers) const {
    return dimension_numbers.getInputBatchDimension() == 0 &&
           dimension_numbers.getInputFeatureDimension() == 1 &&
           dimension_numbers.getInputSpatialDimensions() ==
               ArrayRef<int64_t>{2, 3};
  }

  // Matches kernel dimensions corresponding to: [o, i, 0, 1].
  bool MatchKernelDimensionNumbers(
      const ConvDimensionNumbersAttr dimension_numbers) const {
    return dimension_numbers.getKernelInputFeatureDimension() == 1 &&
           dimension_numbers.getKernelOutputFeatureDimension() == 0 &&
           dimension_numbers.getKernelSpatialDimensions() ==
               ArrayRef<int64_t>{2, 3};
  }

  // Matches output dimensions corresponding to: [b, f, 0, 1].
  bool MatchOutputDimensionNumbers(
      const ConvDimensionNumbersAttr dimension_numbers) const {
    return dimension_numbers.getOutputBatchDimension() == 0 &&
           dimension_numbers.getOutputFeatureDimension() == 1 &&
           dimension_numbers.getOutputSpatialDimensions() ==
               ArrayRef<int64_t>{2, 3};
  }

  // Returns a new tensor type with the shape transposed according to the
  // permutation. The rank of `type` and the size of `permutation` must be
  // equal.
  TensorType GetTransposedTensorType(
      const TensorType type, const ArrayRef<int64_t> permutation) const {
    const SmallVector<int64_t> after_shape =
        Permute<int64_t>(type.getShape(), permutation);
    return type.cloneWith(after_shape, type.getElementType());
  }
};

}  // namespace

void NchwConvolutionToNhwcPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<RewriteNchwConvolutionToNhwc>(&ctx);

  if (failed(applyPatternsAndFoldGreedily(func_op, std::move(patterns)))) {
    func_op.emitError() << "Failed to run NchwConvolutionToNhwcPass.";
    signalPassFailure();
  }
}

}  // namespace mlir::quant::stablehlo
