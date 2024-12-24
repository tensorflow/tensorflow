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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/fft.h"

#include <stdbool.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// Convert a DenseIntElementsAttr to a vector of int64_t.
std::vector<int64_t> ConvertI64DenseIntAttr(DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

// Returns true if the fft op is a supported rfft op.
bool IsSupportedRfftOp(mhlo::FftOp fft_op) {
  const auto fft_type = llvm::StringSwitch<std::optional<mhlo::FftType>>(
                            mlir::mhlo::stringifyFftType(fft_op.getFftType()))
                            .Case("FFT", mhlo::FftType::FFT)
                            .Case("RFFT", mhlo::FftType::RFFT)
                            .Case("IFFT", mhlo::FftType::IFFT)
                            .Case("IRFFT", mhlo::FftType::IRFFT)
                            .Default(std::nullopt);
  if (!fft_type || *fft_type != mhlo::FftType::RFFT) {
    return false;
  }

  const std::vector<int64_t> fft_lengths =
      ConvertI64DenseIntAttr(fft_op.getFftLength());
  const std::vector<int64_t> input_shape =
      mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();
  if (fft_lengths.back() != input_shape.back()) {
    return false;
  }

  auto input_type =
      mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
  if (!input_type || input_type.getRank() != 3) return false;

  return true;
}

// Convert rfft to rfft2d.
// The transformation pattern looks like below:
//
//    input     fft_len
//     \      /
//     rfft
//
//     ||
//     \/
//
//   input       fft_len
//    \            /
//   expand_dim    concat with [1] at the front
//      \         /
//     rfft_2d
//       |
//     squeeze
class Convert1DFftTo2DFftOp : public OpRewritePattern<mhlo::FftOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::FftOp fft_op,
                                PatternRewriter& rewriter) const final {
    const auto fft_type = llvm::StringSwitch<std::optional<mhlo::FftType>>(
                              mlir::mhlo::stringifyFftType(fft_op.getFftType()))
                              .Case("FFT", mhlo::FftType::FFT)
                              .Case("RFFT", mhlo::FftType::RFFT)
                              .Case("IFFT", mhlo::FftType::IFFT)
                              .Case("IRFFT", mhlo::FftType::IRFFT)
                              .Default(std::nullopt);
    if (!fft_type || !(*fft_type == mhlo::FftType::RFFT ||
                       *fft_type == mhlo::FftType::IRFFT)) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft type.");
    }

    const auto fft_lengths =
        llvm::to_vector(fft_op.getFftLength().getValues<int64_t>());
    if (fft_lengths.size() != 1) {
      return rewriter.notifyMatchFailure(
          fft_op, "Can only lower a single fft dimension");
    }

    const int fft_len = fft_lengths.back();
    const std::vector<int64_t> input_shape =
        mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();
    if (fft_len != input_shape.back()) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft length.");
    }

    auto input_type =
        mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
    if (!input_type || input_type.getRank() != 2)
      return rewriter.notifyMatchFailure(fft_op, "Unsupported input type.");

    auto output_type = mlir::cast<ShapedType>(fft_op.getResult().getType());
    const std::vector<int64_t> output_shape = output_type.getShape();

    // Replace the expand_dims op with a reshape op:
    auto expanded_input_type = mlir::RankedTensorType::get(
        {input_shape[0], 1, input_shape[1]}, input_type.getElementType());
    auto reshape_op = rewriter.create<mhlo::ReshapeOp>(
        fft_op.getLoc(), expanded_input_type, fft_op.getOperand());

    // Create a new fft_length attribute for the 3D FFT.
    SmallVector<int64_t, 3> new_fft_lengths = {1, fft_len};
    auto new_fft_lengths_attr = rewriter.getI64TensorAttr(new_fft_lengths);

    // Create a new mhlo.fft op with the expanded input and fft_length.
    auto new_fft_type = mlir::RankedTensorType::get(
        {output_shape[0], 1, output_shape[1]}, output_type.getElementType());
    auto new_fft = rewriter.create<mhlo::FftOp>(
        fft_op.getLoc(), new_fft_type, reshape_op.getResult(), *fft_type,
        new_fft_lengths_attr);

    // Squeeze the output dimensions back to 2D.
    auto squeeze_op = rewriter.create<mhlo::ReshapeOp>(
        fft_op.getLoc(), output_type, new_fft.getResult());

    rewriter.replaceOp(fft_op, squeeze_op.getResult());
    return success();
  }
};

class LegalizeRfftOp : public OpConversionPattern<mhlo::FftOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp fft_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    const auto fft_type = llvm::StringSwitch<std::optional<mhlo::FftType>>(
                              mlir::mhlo::stringifyFftType(fft_op.getFftType()))
                              .Case("FFT", mhlo::FftType::FFT)
                              .Case("RFFT", mhlo::FftType::RFFT)
                              .Case("IFFT", mhlo::FftType::IFFT)
                              .Case("IRFFT", mhlo::FftType::IRFFT)
                              .Default(std::nullopt);
    if (!fft_type || *fft_type != mhlo::FftType::RFFT) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft type.");
    }

    auto fft_lengths =
        llvm::to_vector(fft_op.getFftLength().getValues<int64_t>());
    const std::vector<int64_t> input_shape =
        mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();
    if (fft_lengths.back() != input_shape.back()) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft length.");
    }

    auto input_type =
        mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
    if (!input_type || input_type.getRank() != 3)
      return rewriter.notifyMatchFailure(fft_op, "Unsupported input type.");

    auto output_type = mlir::cast<ShapedType>(fft_op.getResult().getType());

    llvm::SmallVector<int32_t, 2> fft_lengths_i32{fft_lengths.begin(),
                                                  fft_lengths.end()};
    auto fft_len_f32_attr = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(2, rewriter.getIntegerType(32)),
        fft_lengths_i32);

    auto fft_len_const =
        rewriter.create<arith::ConstantOp>(fft_op.getLoc(), fft_len_f32_attr);
    auto tfl_rfft2d = rewriter.create<TFL::RFFT2dOp>(
        fft_op.getLoc(), output_type, fft_op.getOperand(), fft_len_const);

    rewriter.replaceOp(fft_op, tfl_rfft2d.getResult());

    return success();
  }
};

// Returns true if the fft op is a legal fft op.
bool IsLegalFftOp(mhlo::FftOp fft_op) { return !IsSupportedRfftOp(fft_op); }

}  // namespace

void PopulateLegalizeFftPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                                 ConversionTarget& target) {
  patterns.add<LegalizeRfftOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::FftOp>(IsLegalFftOp);
}

void PopulatePrepareFftPatterns(MLIRContext* ctx, RewritePatternSet& patterns) {
  patterns.add<Convert1DFftTo2DFftOp>(ctx);
}

}  // namespace mlir::odml
