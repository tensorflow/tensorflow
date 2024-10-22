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
#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"
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
  if (fft_lengths.size() != 1) {
    return false;
  }

  const int fft_len = fft_lengths.back();
  const std::vector<int64_t> input_shape =
      mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();
  if (fft_len != input_shape.back()) {
    return false;
  }

  auto input_type =
      mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
  if (!input_type || input_type.getRank() != 2) return false;

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

    // Expanded inputs.
    // Insert at -2 location.
    auto one_ele_type = mlir::RankedTensorType::get(
        llvm::ArrayRef<int64_t>{1}, rewriter.getIntegerType(32));
    auto minus_two = TFL::CreateConstOpWithSingleValue(
        &rewriter, fft_op.getLoc(), one_ele_type, -2);

    SmallVector<int64_t, 4> expanded_input_shape;
    SmallVector<int64_t, 4> expanded_output_shape;
    int expanded_rank = input_type.getRank() + 1;
    int r = 0;
    for (int i = 0; i < expanded_rank; ++i) {
      if (i == expanded_rank - 2) {
        expanded_input_shape.push_back(1);
        expanded_output_shape.push_back(1);
      } else {
        expanded_input_shape.push_back(input_type.getDimSize(r));
        expanded_output_shape.push_back(output_type.getDimSize(r));
        r++;
      }
    }

    auto expaned_input_type = mlir::RankedTensorType::get(
        expanded_input_shape, input_type.getElementType());
    TFL::ExpandDimsOp expanded_input = rewriter.create<TFL::ExpandDimsOp>(
        fft_op.getLoc(), expaned_input_type, fft_op.getOperand(),
        minus_two->getResult());

    // Expanded fft_len.
    auto one_attr = mlir::DenseIntElementsAttr::get(one_ele_type, {1});

    auto one = rewriter.create<arith::ConstantOp>(fft_op.getLoc(), one_attr);
    auto fft_len_attr =
        mlir::DenseIntElementsAttr::get(one_ele_type, {fft_len});
    auto fft_len_const =
        rewriter.create<arith::ConstantOp>(fft_op.getLoc(), fft_len_attr);

    auto expanded_fft_len_type = mlir::RankedTensorType::get(
        llvm::ArrayRef<int64_t>{2}, rewriter.getIntegerType(32));

    TFL::ConcatenationOp expanded_fft_len =
        rewriter.create<TFL::ConcatenationOp>(
            fft_op.getLoc(), expanded_fft_len_type,
            llvm::SmallVector<Value, 2>({one, fft_len_const}),
            /*axis*/ 0, /*fused_activation_function*/ "NONE");

    // Insert the rfft_2d.
    auto rfft2d_out_type = mlir::RankedTensorType::get(
        expanded_output_shape, output_type.getElementType());
    auto rfft2d = rewriter.create<TFL::RFFT2dOp>(
        fft_op.getLoc(), rfft2d_out_type, expanded_input.getResult(),
        expanded_fft_len.getResult());

    // Insert the squeeze op.
    auto squeeze_dim = rewriter.getI64ArrayAttr({-2});
    auto squeeze = rewriter.create<TFL::SqueezeOp>(
        fft_op.getLoc(), output_type, rfft2d.getResult(), squeeze_dim);

    rewriter.replaceOp(fft_op, squeeze.getResult());
    return success();
  }
};

// Returns true if the fft op is a legal fft op.
bool IsLegalFftOp(mhlo::FftOp fft_op) { return !IsSupportedRfftOp(fft_op); }

}  // namespace

void PopulateFftPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                         ConversionTarget& target) {
  patterns.add<LegalizeRfftOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::FftOp>(IsLegalFftOp);
}

}  // namespace mlir::odml
