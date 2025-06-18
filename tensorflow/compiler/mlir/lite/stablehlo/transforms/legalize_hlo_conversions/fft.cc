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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "mhlo/IR/hlo_ops.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
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

  if (fft_lengths.size() > 2) return false;  // Only support 2D FFT.

  // Check if the trailing input shape matches the fft_lengths.
  const std::vector<int64_t> input_shape =
      mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();
  return std::equal(input_shape.end() - fft_lengths.size(), input_shape.end(),
                    fft_lengths.begin(), fft_lengths.end());
}

// Returns a tensor of the dimension size of the input tensor. Result of
// mhlo::GetDimensionSizeOp is always a scalar value, but we need a tensor to
// concatenate with other dimension sizes.
Value GetDimensionSizeTensor(OpBuilder& rewriter, Location loc, Value input,
                             int64_t dim) {
  auto size_scalar = rewriter.create<mhlo::GetDimensionSizeOp>(loc, input, dim);
  return rewriter.create<mhlo::ReshapeOp>(
      loc, RankedTensorType::get({1}, rewriter.getI32Type()), size_scalar);
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
class ConvertNDFftTo2DFftOp : public OpRewritePattern<mhlo::FftOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::FftOp fft_op,
                                PatternRewriter& rewriter) const final {
    if (!IsSupportedRfftOp(fft_op)) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft op.");
    }

    const auto fft_lengths =
        llvm::to_vector(fft_op.getFftLength().getValues<int64_t>());
    if (fft_lengths.size() != 1) {
      return rewriter.notifyMatchFailure(
          fft_op, "Can only lower a single fft dimension");
    }

    auto input_type =
        mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
    const std::vector<int64_t> input_shape =
        input_type
            ? input_type.getShape()
            : mlir::cast<ShapedType>(fft_op.getOperand().getType()).getShape();

    Value fft_operand = fft_op.getOperand();
    auto output_type = mlir::cast<ShapedType>(fft_op.getResult().getType());

    // Create a new fft_length attribute for the 2D FFT.
    SmallVector<int64_t, 3> new_fft_lengths = {1, fft_lengths.back()};
    auto new_fft_lengths_attr = rewriter.getI64TensorAttr(new_fft_lengths);

    bool is_dynamic_shape = !input_type || !input_type.hasStaticShape();

    // Input can have a single trivial batch dim next to the fft dimension, in
    // which case we don't need to expand the input.
    if (input_shape[input_shape.size() - 2] != 1) {
      const std::vector<int64_t> output_shape = output_type.getShape();

      // [a, b, c, d, e] -> [a, b, c, d, 1, e]
      SmallVector<int64_t, 6> expanded_input_shape{input_shape.begin(),
                                                   input_shape.end() - 1};
      expanded_input_shape.push_back(1);
      expanded_input_shape.push_back(input_shape.back());
      auto expanded_input_type = tensorflow::GetTypeFromTFTensorShape(
          expanded_input_shape, input_type.getElementType());

      // Dynamic shape needs to be handled separately as mhlo::ReshapeOp does
      // not support dynamic shape.
      if (is_dynamic_shape) {
        // Programmatically-
        // 1. Get the dimensions of the input tensor and create shape vector.
        // 2. Insert a 1 as the penultimate dimension size.
        // 3. Concatenate the dimension sizes to create a new SHAPE tensor.
        SmallVector<Value> expanded_input_shape_values;
        for (int i = 0; i < input_shape.size() - 1; ++i) {
          expanded_input_shape_values.push_back(GetDimensionSizeTensor(
              rewriter, fft_op.getLoc(), fft_operand, i));
        }
        expanded_input_shape_values.push_back(rewriter.create<mhlo::ConstantOp>(
            fft_op.getLoc(), rewriter.getI32TensorAttr({1})));
        expanded_input_shape_values.push_back(GetDimensionSizeTensor(
            rewriter, fft_op.getLoc(), fft_operand, input_shape.size() - 1));

        auto expanded_input_shape_tensor = rewriter.create<mhlo::ConcatenateOp>(
            fft_op.getLoc(),
            RankedTensorType::get(
                {static_cast<int64_t>(expanded_input_shape_values.size())},
                rewriter.getI32Type()),
            expanded_input_shape_values, 0);

        // Create a new mhlo.dynamic_reshape op with the expanded input and
        // expanded input shape. SHAPE tensor is created in the previous step.
        fft_operand = rewriter.create<mhlo::DynamicReshapeOp>(
            fft_op.getLoc(), expanded_input_type, fft_operand,
            expanded_input_shape_tensor);
      } else {
        fft_operand = rewriter.create<mhlo::ReshapeOp>(
            fft_op.getLoc(), expanded_input_type, fft_operand);
      }

      SmallVector<int64_t, 6> new_output_shape = {output_shape.begin(),
                                                  output_shape.end() - 1};
      new_output_shape.push_back(1);
      new_output_shape.push_back(output_shape.back());
      // Create a new mhlo.fft op with the expanded input and fft_length.
      output_type = mlir::RankedTensorType::get(new_output_shape,
                                                output_type.getElementType());
    }

    auto new_fft =
        rewriter.create<mhlo::FftOp>(fft_op.getLoc(), output_type, fft_operand,
                                     fft_op.getFftType(), new_fft_lengths_attr);

    if (input_shape[input_shape.size() - 2] != 1) {
      // Squeeze the output dimensions back to 2D.
      if (is_dynamic_shape) {
        SmallVector<Value> output_shape_values;
        for (int i = 0; i < new_fft.getResult().getType().getShape().size() - 2;
             ++i) {
          output_shape_values.push_back(GetDimensionSizeTensor(
              rewriter, fft_op.getLoc(), new_fft.getResult(), i));
        }
        output_shape_values.push_back(GetDimensionSizeTensor(
            rewriter, fft_op.getLoc(), new_fft.getResult(),
            new_fft.getResult().getType().getShape().size() - 1));

        auto shape_tensor = rewriter.create<mhlo::ConcatenateOp>(
            fft_op.getLoc(),
            RankedTensorType::get(
                {static_cast<int64_t>(output_shape_values.size())},
                rewriter.getI32Type()),
            output_shape_values, 0);
        auto squeeze_op = rewriter.create<mhlo::DynamicReshapeOp>(
            fft_op.getLoc(), fft_op.getResult().getType(), new_fft.getResult(),
            shape_tensor);
        rewriter.replaceOp(fft_op, squeeze_op.getResult());
      } else {
        auto squeeze_op = rewriter.create<mhlo::ReshapeOp>(
            fft_op.getLoc(), fft_op.getResult().getType(), new_fft.getResult());
        rewriter.replaceOp(fft_op, squeeze_op.getResult());
      }
    } else {
      rewriter.replaceOp(fft_op, new_fft.getResult());
    }

    return success();
  }
};

class LegalizeRfftOp : public OpConversionPattern<mhlo::FftOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp fft_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!IsSupportedRfftOp(fft_op)) {
      return rewriter.notifyMatchFailure(fft_op, "Unsupported fft op.");
    }

    auto input_type =
        mlir::dyn_cast_or_null<RankedTensorType>(fft_op.getOperand().getType());
    if (!input_type)
      return rewriter.notifyMatchFailure(fft_op, "Unsupported input type.");

    const auto fft_lengths =
        llvm::to_vector(fft_op.getFftLength().getValues<int64_t>());
    if (fft_lengths.size() != 2) {
      return rewriter.notifyMatchFailure(
          fft_op, "TFLite RFFT2d requires 2D FFT Length.");
    }

    llvm::SmallVector<int32_t, 2> fft_lengths_i32{fft_lengths.begin(),
                                                  fft_lengths.end()};
    auto fft_len_f32_attr = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(2, rewriter.getIntegerType(32)),
        fft_lengths_i32);

    auto output_type = mlir::cast<ShapedType>(fft_op.getResult().getType());
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
  patterns.add<ConvertNDFftTo2DFftOp>(ctx);
}

}  // namespace mlir::odml
