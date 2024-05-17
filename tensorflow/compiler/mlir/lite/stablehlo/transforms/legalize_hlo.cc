/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for legalizing HLO to TensorFlow.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/BroadcastUtils.h"  // from @stablehlo
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/padding.h"

namespace mlir {
namespace odml {
namespace {

#define DEBUG_TYPE "tf-legalize-hlo"

#define GEN_PASS_DEF_LEGALIZEHLOTOTFPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

class LegalizeHloToTf : public impl::LegalizeHloToTfPassBase<LegalizeHloToTf> {
  /// Performs the legalization to the TF dialect.
  void runOnOperation() override;
};

using mhlo::DotDimensionNumbersAttr;

// Replaces `region`'s terminator to TF::Yield.
void ReplaceReturnOp(Region& region, PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  for (auto& block : region.getBlocks()) {
    Operation* terminator = block.getTerminator();
    auto return_op = llvm::dyn_cast_or_null<mhlo::ReturnOp>(terminator);
    if (return_op == nullptr) continue;

    rewriter.setInsertionPoint(return_op);
    rewriter.replaceOpWithNewOp<TF::YieldOp>(return_op,
                                             return_op->getOperands());
  }
}

// If `value` is a splat constant, returns a success and set `splat_value`
// to the splate constant value.
// `SplatValueType` can be `APInt` or `APFloat`.
template <typename SplatValueType>
LogicalResult GetConstantSplatValue(Value value, SplatValueType& splat_value) {
  DenseElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat()) {
    return failure();
  }

  splat_value = attr.getSplatValue<SplatValueType>();
  return success();
}

// Checks for attributes with no value and sets the default value according to
// the convolution op definition.
void SetDefaultConvAttributes(mhlo::ConvolutionOp& conv_op,
                              ConversionPatternRewriter& rewriter) {
  auto dnums = conv_op.getDimensionNumbers();
  int32_t input_spatial_dims =
      static_cast<int32_t>(dnums.getInputSpatialDimensions().size());
  if (!conv_op.getWindowStrides().has_value()) {
    conv_op.setWindowStridesAttr(
        rewriter.getI64TensorAttr(std::vector<int64_t>(input_spatial_dims, 1)));
  }
  if (!conv_op.getPadding().has_value()) {
    conv_op.setPaddingAttr(DenseIntElementsAttr::get(
        RankedTensorType::get({input_spatial_dims, 2}, rewriter.getI64Type()),
        SmallVector<int64_t>(input_spatial_dims * 2, 0)));
  }
  if (!conv_op.getLhsDilation().has_value()) {
    conv_op.setLhsDilationAttr(
        rewriter.getI64TensorAttr(std::vector<int64_t>(input_spatial_dims, 1)));
  }
  if (!conv_op.getRhsDilation().has_value()) {
    conv_op.setRhsDilationAttr(
        rewriter.getI64TensorAttr(std::vector<int64_t>(input_spatial_dims, 1)));
  }
}

template <int num_spatial_dims>
class ConvertNdConvOp : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SetDefaultConvAttributes(conv_op, rewriter);
    if (!IsSupportedConvOp(conv_op)) {
      return failure();
    }

    // tf Convolution doesn't support quantized type.
    if (mlir::isa<quant::QuantizedType>(
            conv_op.getRhs().getType().getElementType())) {
      return failure();
    }

    // Constructs strides array.
    // For example, [2, 3] -> [1, 2, 3, 1].
    SmallVector<int64_t, num_spatial_dims + 2> strides({1});
    for (const auto v :
         conv_op.getWindowStrides().value().getValues<int64_t>()) {
      strides.emplace_back(v);
    }
    strides.emplace_back(1);

    // Constructs dilation array.
    SmallVector<int64_t, num_spatial_dims + 2> dilation;
    if (auto rhs_dilation = conv_op.getRhsDilation()) {
      // For example, [2, 3] -> [1, 2, 3, 1].
      dilation.emplace_back(1);
      dilation.append(rhs_dilation.value().getValues<int64_t>().begin(),
                      rhs_dilation.value().getValues<int64_t>().end());
      dilation.emplace_back(1);
    }

    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();
    const int input_feature_dimension = dnums.getInputFeatureDimension();
    const int kernel_input_feature_dimension =
        dnums.getKernelInputFeatureDimension();
    const int input_channels =
        mlir::cast<ShapedType>(conv_op.getLhs().getType())
            .getDimSize(input_feature_dimension);
    const int kernel_input_channels =
        mlir::cast<ShapedType>(conv_op.getRhs().getType())
            .getDimSize(kernel_input_feature_dimension);
    int feature_group_count = conv_op.getFeatureGroupCount();

    // check if group count is valid
    if (feature_group_count != input_channels / kernel_input_channels ||
        input_channels % kernel_input_channels != 0) {
      return failure();
    }

    const bool is_depthwise_conv = input_channels == feature_group_count;
    std::string padding;
    SmallVector<int64_t, (num_spatial_dims * 2) + 4> explicit_padding;
    if (conv_op.getPadding().value().isSplat() &&
        conv_op.getPadding()->getSplatValue<int64_t>() == 0) {
      padding = "VALID";
    } else {
      SmallVector<int64_t, num_spatial_dims * 2> padding_array;
      for (const auto v : conv_op.getPadding().value().getValues<int64_t>()) {
        padding_array.emplace_back(v);
      }

      if (IsSamePadding(conv_op, strides, dilation, padding_array)) {
        // Check if padding is "SAME".
        padding = "SAME";
      } else {
        padding = "EXPLICIT";
        explicit_padding.push_back(0);
        explicit_padding.push_back(0);
        explicit_padding.append(padding_array);
        explicit_padding.push_back(0);
        explicit_padding.push_back(0);
      }
    }

    CreateConvOp(conv_op, strides, padding, explicit_padding, dilation,
                 is_depthwise_conv, input_channels, rewriter);

    return success();
  };

  static bool IsSupportedConvOp(mhlo::ConvolutionOp conv_op) {
    if (!mlir::cast<ShapedType>(conv_op.getRhs().getType()).hasStaticShape()) {
      return false;
    }
    if (!mlir::cast<ShapedType>(conv_op.getLhs().getType()).hasStaticShape() &&
        !mlir::cast<ShapedType>(conv_op.getType()).hasStaticShape()) {
      auto dnums = conv_op.getDimensionNumbers();
      auto lhs_type = mlir::cast<ShapedType>(conv_op.getLhs().getType());
      auto out_type = mlir::cast<ShapedType>(conv_op.getType());
      int64_t input_batch_dim = dnums.getInputBatchDimension();
      int64_t out_batch_dim = dnums.getOutputBatchDimension();
      for (size_t i = 0; i < lhs_type.getRank(); ++i) {
        // is this upcast of size_t or downcast
        if ((i != input_batch_dim && lhs_type.isDynamicDim(i)) ||
            (i != out_batch_dim && out_type.isDynamicDim(i))) {
          return false;
        }
      }
    }

    // All ones in "lhs_dilation" means this "mhlo.conv" op should be
    // converted to "tf.Conv2D" or "tf.DepthwiseConv2dNativeOp".
    auto lhs_dilation = conv_op.getLhsDilation().value();
    if (!lhs_dilation.isSplat() || lhs_dilation.getSplatValue<int64_t>() != 1)
      return false;

    if (mlir::cast<ShapedType>(conv_op.getWindowStrides().value().getType())
            .getRank() != 1)
      return false;

    auto spatial_dims =
        conv_op.getDimensionNumbers().getInputSpatialDimensions().size();
    if (spatial_dims != num_spatial_dims) return false;

    return true;
  }

 private:
  bool IsSamePadding(mhlo::ConvolutionOp conv_op, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> dilation,
                     ArrayRef<int64_t> padding_array) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();
    auto input_spatial_dim = dnums.getInputSpatialDimensions();
    auto kernel_spatial_dim = dnums.getKernelSpatialDimensions();
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      int dim = i + 1;
      int64_t output_size;
      int64_t pad_low_int64;
      int64_t pad_high_int64;
      tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerbose(
          mlir::cast<ShapedType>(conv_op.getLhs().getType())
              .getDimSize(input_spatial_dim[i]),
          mlir::cast<ShapedType>(conv_op.getRhs().getType())
              .getDimSize(kernel_spatial_dim[i]),
          dilation[dim], strides[dim], tensorflow::Padding::SAME, &output_size,
          &pad_low_int64, &pad_high_int64);
      if (!status.ok()) return false;
      if (padding_array[2 * i] != pad_low_int64 ||
          padding_array[2 * i + 1] != pad_high_int64)
        return false;
    }

    return true;
  }

  // Slices the input `value` if there are negative padding values in
  // `explicit_padding`.
  Value SliceNegativePadding(Value value, ArrayRef<int64_t> explicit_padding,
                             ConversionPatternRewriter& rewriter) const {
    // If no padding is negative return the input as is.
    if (llvm::all_of(explicit_padding, [](int64_t pad) { return pad >= 0; })) {
      return value;
    }

    auto input_type = mlir::cast<RankedTensorType>(value.getType());
    auto input_shape = input_type.getShape();

    llvm::SmallVector<int64_t, 4> start;
    llvm::SmallVector<int64_t, 4> size;
    start.reserve(explicit_padding.size() / 2);
    size.reserve(explicit_padding.size() / 2);
    for (int i = 0, e = explicit_padding.size() / 2; i < e; ++i) {
      int64_t pre_padding = explicit_padding[2 * i];
      int64_t post_padding = explicit_padding[2 * i + 1];
      int64_t pre_slice = pre_padding < 0 ? -pre_padding : 0;
      int64_t post_slice = post_padding < 0 ? -post_padding : 0;
      start.push_back(pre_slice);
      size.push_back(input_shape[i] - pre_slice - post_slice);
    }

    auto start_attr = rewriter.create<TF::ConstOp>(
        value.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(start.size())},
                                  rewriter.getI64Type()),
            start));
    auto size_attr = rewriter.create<TF::ConstOp>(
        value.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(size.size())},
                                  rewriter.getI64Type()),
            size));
    auto output_type = RankedTensorType::get(size, input_type.getElementType());

    return rewriter.create<TF::SliceOp>(value.getLoc(), output_type, value,
                                        start_attr, size_attr);
  }

  void CreateConvOp(mhlo::ConvolutionOp conv_op, ArrayRef<int64_t> strides,
                    StringRef padding, ArrayRef<int64_t> explicit_padding,
                    ArrayRef<int64_t> dilation, bool is_depthwise_conv,
                    int input_channels,
                    ConversionPatternRewriter& rewriter) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();
    // Transposes lhs and rhs if their formats are not NHWC.
    Value lhs = InsertTranspose(
        conv_op.getLhs(), dnums.getInputBatchDimension(),
        dnums.getInputFeatureDimension(), dnums.getInputSpatialDimensions(),
        /*default_batch_dim=*/0, /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/1, num_spatial_dims, rewriter);

    Value rhs = InsertTranspose(
        conv_op.getRhs(), dnums.getKernelInputFeatureDimension(),
        dnums.getKernelOutputFeatureDimension(),
        dnums.getKernelSpatialDimensions(),
        /*default_batch_dim=*/num_spatial_dims,
        /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/0, num_spatial_dims, rewriter);

    // Emulate negative padding with a slice and remove negative values from the
    // padding vector.
    Value sliced_lhs = SliceNegativePadding(lhs, explicit_padding, rewriter);
    auto new_padding =
        llvm::to_vector<(num_spatial_dims * 2) + 4>(llvm::map_range(
            explicit_padding, [](int64_t dim) { return dim > 0 ? dim : 0; }));

    // Add an TF.PadOp before the LHS if there is explicit_padding on a 3D
    // Convolution. This is needed because TF.Conv3DOp doesn't support EXPLICIT.
    if (padding == "EXPLICIT" && num_spatial_dims == 3) {
      auto lhs_type =
          mlir::dyn_cast<RankedTensorType>(conv_op.getLhs().getType());
      RankedTensorType padding_attr_type = mlir::RankedTensorType::get(
          {lhs_type.getRank(), 2}, rewriter.getIntegerType(64));
      auto padding_const = rewriter.create<TF::ConstOp>(
          conv_op->getLoc(),
          mlir::DenseElementsAttr::get(padding_attr_type,
                                       ArrayRef<int64_t>(new_padding)));
      // Add Pad op.
      auto pad_output_type = UnrankedTensorType::get(lhs_type.getElementType());
      sliced_lhs = rewriter.create<TF::PadOp>(
          conv_op->getLoc(), pad_output_type, sliced_lhs, padding_const);
      padding = "VALID";
    }

    auto conv_output_type = mlir::cast<RankedTensorType>(conv_op.getType());
    DenseIntElementsAttr permutation;
    const bool need_transpose_output = NeedsReformatTypeAndPermutation(
        dnums.getOutputBatchDimension(), dnums.getOutputFeatureDimension(),
        dnums.getOutputSpatialDimensions().front(),
        /*default_batch_dim=*/0, /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/1);
    if (need_transpose_output) {
      std::pair<RankedTensorType&, DenseIntElementsAttr&>(conv_output_type,
                                                          permutation) =
          GetReformatTypeAndPermutation(
              dnums.getOutputBatchDimension(),
              dnums.getOutputFeatureDimension(),
              dnums.getOutputSpatialDimensions().front(),
              /*default_batch_dim=*/0,
              /*default_feature_dim=*/num_spatial_dims + 1,
              /*default_spatial_dim_start=*/1, num_spatial_dims,
              conv_output_type, rewriter);
    }
    Value output;
    if (is_depthwise_conv && num_spatial_dims == 2) {
      // Reshapes filter format to [filter_height, filter_width, in_channels,
      // channel_multiplier] from HLO's [filter_height, filter_width, 1,
      // in_channels * channel_multiplier] format.
      auto filter_type = mlir::cast<ShapedType>(rhs.getType());
      llvm::ArrayRef<int64_t> hlo_filter_shape = filter_type.getShape();
      llvm::SmallVector<int64_t, 4> tf_filter_shape(hlo_filter_shape.begin(),
                                                    hlo_filter_shape.end());
      tf_filter_shape[2] = input_channels;
      tf_filter_shape[3] = hlo_filter_shape.back() / input_channels;
      auto reshaped_filter = rewriter.create<mhlo::ReshapeOp>(
          rhs.getLoc(),
          RankedTensorType::get(tf_filter_shape, filter_type.getElementType()),
          rhs);

      output = rewriter.create<TF::DepthwiseConv2dNativeOp>(
          conv_op.getLoc(), conv_output_type, sliced_lhs, reshaped_filter,
          rewriter.getI64ArrayAttr(strides),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr(new_padding),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    } else if (num_spatial_dims == 3) {
      output = rewriter.create<TF::Conv3DOp>(
          conv_op.getLoc(), conv_output_type, sliced_lhs, rhs,
          rewriter.getI64ArrayAttr(strides),
          /*padding=*/rewriter.getStringAttr(padding),
          /*data_format=*/rewriter.getStringAttr("NDHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    } else {
      output = rewriter.create<TF::Conv2DOp>(
          conv_op.getLoc(), conv_output_type, sliced_lhs, rhs,
          rewriter.getI64ArrayAttr(strides),
          /*use_cudnn_on_gpu=*/rewriter.getBoolAttr(true),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr(new_padding),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    }

    if (need_transpose_output) {
      // Converts from "NHWC" format back to the original output format.
      std::pair<RankedTensorType&, DenseIntElementsAttr&>(conv_output_type,
                                                          permutation) =
          GetReformatTypeAndPermutation(
              /*batch_dim=*/0, /*feature_dim=*/num_spatial_dims + 1,
              /*spatial_dim_start=*/1, dnums.getOutputBatchDimension(),
              dnums.getOutputFeatureDimension(),
              *dnums.getOutputSpatialDimensions().begin(), num_spatial_dims,
              conv_output_type, rewriter);
      output = rewriter.create<mhlo::TransposeOp>(
          conv_op.getLoc(), conv_op.getType(), output, permutation);
    }
    rewriter.replaceOp(conv_op, {output});
  }
};

// Convert a 1-D convolution into a 2-D convolution (which TF supports) so that
// it can be rewritten by the pattern `Convert2DConvOp`.
class Convert1DConvOp : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SetDefaultConvAttributes(conv_op, rewriter);
    // Check that input is a supported 1d convolution.
    if (!ConvertNdConvOp<1>::IsSupportedConvOp(conv_op) ||
        conv_op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(conv_op, "unsupported conv op.");

    const mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();

    // Group convolution is not supported yet.
    const int64_t input_feature_dimension = dnums.getInputFeatureDimension();
    const int64_t input_channels =
        mlir::cast<ShapedType>(conv_op.getLhs().getType())
            .getDimSize(input_feature_dimension);
    const int kernel_input_feature_dimension =
        dnums.getKernelInputFeatureDimension();
    const int kernel_input_channels =
        mlir::cast<ShapedType>(conv_op.getRhs().getType())
            .getDimSize(kernel_input_feature_dimension);
    const int64_t feature_group_count = conv_op.getFeatureGroupCount();
    if (feature_group_count != input_channels / kernel_input_channels ||
        input_channels % kernel_input_channels != 0)
      return failure();

    //
    // Transpose and reshape the input and kernel
    //

    // Reshape input image to add a new spatial dimension.
    auto image_type = mlir::cast<ShapedType>(conv_op.getLhs().getType());
    SmallVector<int64_t, 4> image_2d_shape(image_type.getShape().begin(),
                                           image_type.getShape().end());
    image_2d_shape.push_back(1);
    auto image_2d_type =
        RankedTensorType::get(image_2d_shape, image_type.getElementType());
    auto loc = conv_op.getLoc();
    auto image_2d_op = rewriter.create<mhlo::ReshapeOp>(
        conv_op.getLoc(), image_2d_type, conv_op.getLhs());

    // Transpose image to get it into NWHC form (where H is the added dim).
    SmallVector<int64_t, 4> image_permutation = {
        dnums.getInputBatchDimension(), dnums.getInputSpatialDimensions()[0],
        3,  // The trailing dim that we added.
        dnums.getInputFeatureDimension()};
    auto image_permutation_and_shape = GetPermutationAndTransposedShape(
        image_permutation, image_2d_type, rewriter);
    auto transposed_image_2d_op = rewriter.create<mhlo::TransposeOp>(
        loc, image_permutation_and_shape.shape, image_2d_op->getResult(0),
        image_permutation_and_shape.permutation);

    // Reshape kernel to add a new spatial dimension.
    auto kernel_type = mlir::cast<ShapedType>(conv_op.getRhs().getType());
    SmallVector<int64_t, 4> kernel_2d_shape;
    for (int64_t dim : kernel_type.getShape()) {
      kernel_2d_shape.push_back(dim);
    }
    kernel_2d_shape.push_back(1);
    auto kernel_2d_type =
        RankedTensorType::get(kernel_2d_shape, kernel_type.getElementType());
    auto kernel_2d_op =
        rewriter.create<mhlo::ReshapeOp>(loc, kernel_2d_type, conv_op.getRhs());

    // Transpose kernel to get it into WHIO form (where H is the added dim).
    SmallVector<int64_t, 4> kernel_permutation = {
        dnums.getKernelSpatialDimensions()[0],
        3,  // The trailing dim that we added.
        dnums.getKernelInputFeatureDimension(),
        dnums.getKernelOutputFeatureDimension()};
    auto kernel_permutation_and_shape = GetPermutationAndTransposedShape(
        kernel_permutation, kernel_2d_type, rewriter);
    auto transposed_kernel_2d_op = rewriter.create<mhlo::TransposeOp>(
        loc, kernel_permutation_and_shape.shape, kernel_2d_op->getResult(0),
        kernel_permutation_and_shape.permutation);

    //
    // Create 2d equivalents for 1d convolution attributes.
    //

    // Window Strides
    SmallVector<int64_t, 2> window_strides_2d_array;
    for (const auto v : conv_op.getWindowStrides()->getValues<int64_t>()) {
      window_strides_2d_array.emplace_back(v);
    }
    window_strides_2d_array.push_back(1);
    auto window_strides_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        window_strides_2d_array);

    // Padding
    SmallVector<int64_t, 4> padding_2d_array;
    for (const auto v : conv_op.getPadding().value().getValues<int64_t>()) {
      padding_2d_array.emplace_back(v);
    }
    // The newly added spatial dimension requires zero left and right padding.
    padding_2d_array.push_back(0);
    padding_2d_array.push_back(0);
    auto padding_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2, 2}, rewriter.getI64Type()), padding_2d_array);

    // LHS dilation
    SmallVector<int64_t, 4> lhs_dilation_array_2d;
    for (const auto v : conv_op.getLhsDilation().value().getValues<int64_t>()) {
      lhs_dilation_array_2d.emplace_back(v);
    }
    lhs_dilation_array_2d.push_back(1);
    auto lhs_dilation_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        lhs_dilation_array_2d);

    // RHS dilation
    SmallVector<int64_t, 4> rhs_dilation_array_2d;
    for (const auto v : conv_op.getRhsDilation().value().getValues<int64_t>()) {
      rhs_dilation_array_2d.emplace_back(v);
    }
    rhs_dilation_array_2d.push_back(1);
    auto rhs_dilation_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        rhs_dilation_array_2d);

    // Window reversal is unsupported.
    if (conv_op.getWindowReversal().has_value() &&
        conv_op.getWindowReversal()->getValues<bool>()[0] == true)
      return failure();
    auto window_reversal_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        SmallVector<int64_t>({0, 0}));

    // Dimension numbers reflect the form of the 2d conv op NWHC * WHIO -> NWHC
    auto dnums_2d =
        mhlo::ConvDimensionNumbersAttr::get(rewriter.getContext(),
                                            /*inputBatchDimension=*/0,
                                            /*inputFeatureDimension=*/3,
                                            /*inputSpatialDimensions=*/{1, 2},
                                            /*kernelInputDimension=*/2,
                                            /*kernelOutputDimension=*/3,
                                            /*kernelSpatialDimensions=*/{0, 1},
                                            /*outputBatchDimension=*/0,
                                            /*outputFeatureDimension=*/3,
                                            /*outputSpatialDimensions=*/{1, 2});
    //
    // Generate a 2-D convolution
    //

    // Determine the 2-D convolution output shape.
    auto output_type = mlir::cast<ShapedType>(conv_op->getResult(0).getType());
    SmallVector<int64_t, 4> output_2d_shape;
    for (int64_t dim : output_type.getShape()) {
      output_2d_shape.push_back(dim);
    }
    output_2d_shape.push_back(1);
    auto output_2d_type =
        RankedTensorType::get(output_2d_shape, output_type.getElementType());
    SmallVector<int64_t, 4> output_permutation = {
        dnums.getOutputBatchDimension(), dnums.getOutputSpatialDimensions()[0],
        3,  // The trailing dim that we added.
        dnums.getOutputFeatureDimension()};
    auto transposed_output_2d_shape =
        GetPermutationAndTransposedShape(output_permutation, output_2d_type,
                                         rewriter)
            .shape;

    auto conv2d_op = rewriter.create<mhlo::ConvolutionOp>(
        loc, transposed_output_2d_shape, transposed_image_2d_op.getResult(),
        transposed_kernel_2d_op.getResult(), window_strides_2d, padding_2d,
        lhs_dilation_2d, rhs_dilation_2d, window_reversal_2d, dnums_2d,
        conv_op.getFeatureGroupCount(), conv_op.getBatchGroupCount(),
        conv_op.getPrecisionConfigAttr());

    OpResult conv2d_output = conv2d_op->getResult(0);
    auto conv2d_output_type = mlir::cast<ShapedType>(conv2d_output.getType());

    //
    // Transpose and reshape the output
    //

    // Since output is in NWHC form we need to undo the permutation we have
    // affectively applied.
    auto output_permutation_and_shape = GetInversePermutationAndShape(
        output_permutation, conv2d_output_type, rewriter);
    auto transposed_output_2d_op = rewriter.create<mhlo::TransposeOp>(
        loc, output_permutation_and_shape.shape, conv2d_output,
        output_permutation_and_shape.permutation);

    // Drop the trailing spatial dimension from the output.
    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
        conv_op, output_type, transposed_output_2d_op.getResult());
    return success();
  }
};

using Convert2DConvOp = ConvertNdConvOp<2>;
using Convert3DConvOp = ConvertNdConvOp<3>;

// Utility function to check for supported non-trivial convolutions with
// lhs_dilation>1 and window_strides=1.
LogicalResult IsSupportedNonTrivialConvOp(mhlo::ConvolutionOp conv_op,
                                          ConversionPatternRewriter& rewriter) {
  if (!mlir::cast<ShapedType>(conv_op.getLhs().getType()).hasStaticShape() ||
      !mlir::cast<ShapedType>(conv_op.getRhs().getType()).hasStaticShape() ||
      !mlir::cast<ShapedType>(conv_op.getType()).hasStaticShape())
    return rewriter.notifyMatchFailure(conv_op, "requires static shape");
  mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();

  auto lhs_dilation = conv_op.getLhsDilation().value();
  if (lhs_dilation.isSplat() && lhs_dilation.getSplatValue<int64_t>() == 1)
    return rewriter.notifyMatchFailure(conv_op,
                                       "requires non-trivial lhs_dilation");

  if (mlir::cast<ShapedType>(conv_op.getWindowStrides().value().getType())
          .getRank() != 1)
    return rewriter.notifyMatchFailure(
        conv_op, "requires window_strides to equal to one");

  int num_spatial_dims = dnums.getInputSpatialDimensions().size();
  if (num_spatial_dims != 2)
    return rewriter.notifyMatchFailure(conv_op, "doesn't support more than 2D");

  if (llvm::any_of(conv_op.getPadding().value().getValues<int64_t>(),
                   [](int64_t v) { return v < 0; })) {
    return rewriter.notifyMatchFailure(conv_op,
                                       "doesn't support negative pads");
  }

  return success();
}

DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                        Builder* builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

class ConvertToResizeBilinearOpOrDepthwiseTransposedConvOp
    : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SetDefaultConvAttributes(conv_op, rewriter);

    // IsSupportedNonTrivialConvOp checks for supported lhs_dilation values.
    if (IsSupportedNonTrivialConvOp(conv_op, rewriter).failed()) {
      return rewriter.notifyMatchFailure(
          conv_op,
          "doesn't support conversion to ResizeBilinearOp or "
          "ConvBackpropInputOp");
    }

    // tf.ResizeBilinearOp is perferred than tf.Conv2DBackpropInputOp since
    // the former has better portability, especially in inference use cases.
    bool align_corners;
    llvm::SmallVector<int, 2> output_sizes;
    if (MatchToResizeBilinearOp(conv_op, align_corners, output_sizes, rewriter)
            .succeeded()) {
      CreateResizeBilinearOp(conv_op, output_sizes, align_corners, rewriter);
      return success();
    }

    // These checks narrow down the support to depthwise transpose conv2d.
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();
    const int input_feature_dimension = dnums.getInputFeatureDimension();
    const int input_channels =
        mlir::cast<ShapedType>(conv_op.getLhs().getType())
            .getDimSize(input_feature_dimension);
    int feature_group_count = conv_op.getFeatureGroupCount();
    const int kernel_input_feature_dimension =
        dnums.getKernelInputFeatureDimension();
    const int kernel_input_channels =
        mlir::cast<ShapedType>(conv_op.getRhs().getType())
            .getDimSize(kernel_input_feature_dimension);
    const int kernel_output_feature_dimension =
        dnums.getKernelOutputFeatureDimension();
    const int kernel_output_channels =
        mlir::cast<ShapedType>(conv_op.getRhs().getType())
            .getDimSize(kernel_output_feature_dimension);

    // To support a depthwise convolution, we need-
    // 1. feature_group_count != 1 (except when input_channels==1)
    // 2. feature_group_count == input_channels
    // 3. kernel_input_channels == 1
    // 4. kernel_output_channels % kernel_input_channels == 0
    if (feature_group_count == 1) {
      return rewriter.notifyMatchFailure(conv_op,
                                         "Not a depthwise convolution");
    }

    if (input_channels != feature_group_count) {
      return rewriter.notifyMatchFailure(
          conv_op, "Not a detphwise transposed convolution");
    }

    if ((kernel_output_channels % feature_group_count != 0) ||
        (kernel_input_channels != 1)) {
      return rewriter.notifyMatchFailure(
          conv_op, "Not a supported detphwise transposed convolution");
    }

    // This needs to be checked because the TFLite runtime generated incorrect
    // results for depthwise transpose convolutions with non-1 channel
    // multiplier.
    if ((kernel_output_channels / feature_group_count) != 1) {
      return rewriter.notifyMatchFailure(
          conv_op,
          "Unsupported detphwise transpose convolution with non-1 channel "
          "multiplier");
    }

    // Slicing with dynamic offsets (helper method advised)
    auto create_slice = [&](mlir::Value tensor, int depth_idx, int channel_idx,
                            bool is_kernel = false) -> mlir::Value {
      std::vector<int64_t> tensor_shape =
          mlir::cast<ShapedType>(tensor.getType()).getShape().vec();

      // Calculate offsets based on depth_idx, channel_idx and tensor_shape
      std::vector<int64_t> start_indices(tensor_shape.size(), 0);
      std::vector<int64_t> limit_indices = tensor_shape;
      const std::vector<int64_t> strides(tensor_shape.size(), 1);
      start_indices[channel_idx] = depth_idx;
      if (is_kernel) {
        // kernel can have a channel_multiplier that needs to be accounted for
        limit_indices[channel_idx] =
            depth_idx + (kernel_output_channels / feature_group_count);
      } else {
        limit_indices[channel_idx] = depth_idx + 1;
      }
      return rewriter.create<mhlo::SliceOp>(
          conv_op.getLoc(), tensor,
          GetI64ElementsAttr(start_indices, &rewriter),
          GetI64ElementsAttr(limit_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter));
    };

    // Storage for smaller convolution results
    std::vector<mlir::Value> conv_results;

    // Iterative Slicing and Convolutions
    for (int i = 0; i < feature_group_count; ++i) {
      auto sliced_input =
          create_slice(conv_op.getLhs(), i, input_feature_dimension);
      auto sliced_kernel = create_slice(conv_op.getRhs(), i,
                                        kernel_output_feature_dimension, true);

      // Calculate convolution output_type based on sliced_input and
      // sliced_kernel
      auto output_type =
          mlir::cast<ShapedType>(conv_op->getResult(0).getType());
      std::vector<int64_t> new_output_shape = output_type.getShape().vec();
      new_output_shape[dnums.getOutputFeatureDimension()] /=
          feature_group_count;
      auto new_output_type =
          RankedTensorType::get(new_output_shape, output_type.getElementType());

      // Create a Smaller Convolution (Ensure compatibility)
      auto conv_result = rewriter.create<mhlo::ConvolutionOp>(
          conv_op.getLoc(), new_output_type, sliced_input, sliced_kernel,
          conv_op.getWindowStridesAttr(), conv_op.getPaddingAttr(),
          conv_op.getLhsDilationAttr(), conv_op.getRhsDilationAttr(),
          conv_op.getWindowReversalAttr(), conv_op.getDimensionNumbers(), 1, 1,
          conv_op.getPrecisionConfigAttr());

      conv_results.push_back(conv_result);
    }

    auto final_output = rewriter.create<mhlo::ConcatenateOp>(
        conv_op.getLoc(), conv_results,
        rewriter.getI64IntegerAttr(dnums.getOutputFeatureDimension()));
    rewriter.replaceOp(conv_op, final_output.getResult());
    return mlir::success();
  }

 private:
  void CreateResizeBilinearOp(mhlo::ConvolutionOp conv_op,
                              llvm::ArrayRef<int32_t> output_sizes,
                              bool align_corners,
                              ConversionPatternRewriter& rewriter) const {
    Value output_sizes_attr = rewriter.create<TF::ConstOp>(
        conv_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(output_sizes.size())},
                                  rewriter.getI32Type()),
            output_sizes));
    // The value of half_pixel_centers couldn't be inferred from the IR and XLA
    // only support half_pixel_centers=True as in 01/11/2022. Here
    // half_pixel_centers=False is hardcoded.
    Value output = rewriter.create<TF::ResizeBilinearOp>(
        conv_op.getLoc(), conv_op.getType(), conv_op.getLhs(),
        output_sizes_attr,
        /*align_corners=*/rewriter.getBoolAttr(align_corners),
        /*half_pixel_centers=*/rewriter.getBoolAttr(false));
    rewriter.replaceOp(conv_op, {output});
  }

  LogicalResult MatchToResizeBilinearOp(
      mhlo::ConvolutionOp conv_op, bool& align_corners,
      llvm::SmallVector<int, 2>& output_sizes,
      ConversionPatternRewriter& rewriter) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();

    int feature_group_count = conv_op.getFeatureGroupCount();
    const int input_feature_dimension = dnums.getInputFeatureDimension();
    const int input_channels =
        mlir::cast<ShapedType>(conv_op.getLhs().getType())
            .getDimSize(input_feature_dimension);

    // Check for Group Convolution parameters
    if (feature_group_count != 1 && feature_group_count != input_channels) {
      // Group convolution is not supported yet.
      return rewriter.notifyMatchFailure(conv_op,
                                         "doesn't support group convolution");
    }

    auto input_spatial_dimensions = dnums.getInputSpatialDimensions();
    auto kernel_spatial_dimensions = dnums.getKernelSpatialDimensions();
    auto output_spatial_dimensions = dnums.getOutputSpatialDimensions();
    if (input_spatial_dimensions.size() != 2 ||
        output_spatial_dimensions.size() != 2 ||
        kernel_spatial_dimensions.size() != 2 ||
        input_spatial_dimensions[0] != output_spatial_dimensions[0] ||
        input_spatial_dimensions[1] != output_spatial_dimensions[1])
      return rewriter.notifyMatchFailure(
          conv_op, "can only be converted to 2D resize op");

    auto lhs_dilation = conv_op.getLhsDilation().value();
    auto rhs_dilation = conv_op.getRhsDilation().value();
    auto window_strides = conv_op.getWindowStrides().value();
    auto padding = conv_op.getPadding().value();
    if (lhs_dilation.getNumElements() != 2 || !rhs_dilation.isSplat() ||
        rhs_dilation.getSplatValue<int64_t>() != 1 ||
        window_strides.getNumElements() != 2 || padding.getNumElements() != 4)
      return rewriter.notifyMatchFailure(
          conv_op, "resize op requires [2] dilations and [2,2] padding");
    auto lhs_dilation_values = lhs_dilation.getValues<int64_t>();
    auto window_strides_values = window_strides.getValues<int64_t>();
    auto padding_values = padding.getValues<int64_t>();

    // Cast the dimension sizes to int.
    auto lhs_type = mlir::cast<ShapedType>(conv_op.getLhs().getType());
    llvm::SmallVector<int> input_sizes = {
        static_cast<int>(lhs_type.getDimSize(input_spatial_dimensions[0])),
        static_cast<int>(lhs_type.getDimSize(input_spatial_dimensions[1]))};
    output_sizes = {static_cast<int>(conv_op.getType().getDimSize(
                        output_spatial_dimensions[0])),
                    static_cast<int>(conv_op.getType().getDimSize(
                        output_spatial_dimensions[1]))};

    // This is based on method in compiler/tf2xla/kernels/image_resize_ops.cc
    auto can_convert_to_bilinear = [](bool align_corners, int64_t dilation,
                                      int64_t padding, int64_t stride,
                                      int64_t input_spatial,
                                      int64_t output_spatial) {
      int64_t input_spatial_size =
          align_corners ? input_spatial - 1 : input_spatial;
      int64_t output_spatial_size =
          align_corners ? output_spatial - 1 : output_spatial;

      int64_t gcd =
          tensorflow::MathUtil::GCD(static_cast<uint64_t>(input_spatial_size),
                                    static_cast<uint64_t>(output_spatial_size));
      if ((input_spatial_size % gcd != 0) ||
          (input_spatial_size / gcd != stride) || (dilation - 1 != padding)) {
        return false;
      }

      return true;
    };

    // Only of the lhs_dilation must be 1, then the non-1 dimension is the
    // resize dimension.
    if (lhs_dilation_values[0] != 1 && lhs_dilation_values[1] == 1) {
      if (can_convert_to_bilinear(
              /*align_corners=*/true, lhs_dilation_values[0], padding_values[0],
              window_strides_values[0], input_sizes[0], output_sizes[0])) {
        align_corners = true;
        return success();
      }
      if (can_convert_to_bilinear(
              /*align_corners=*/false, lhs_dilation_values[0],
              padding_values[0], window_strides_values[0], input_sizes[0],
              output_sizes[0])) {
        align_corners = false;
        return success();
      }
    }

    if (lhs_dilation_values[0] == 1 && lhs_dilation_values[1] != 1) {
      if (can_convert_to_bilinear(
              /*align_corners=*/true, lhs_dilation_values[1], padding_values[2],
              window_strides_values[1], input_sizes[1], output_sizes[1])) {
        align_corners = true;
        return success();
      }
      if (can_convert_to_bilinear(
              /*align_corners=*/false, lhs_dilation_values[1],
              padding_values[2], window_strides_values[1], input_sizes[1],
              output_sizes[1])) {
        align_corners = false;
        return success();
      }
    }

    return rewriter.notifyMatchFailure(conv_op,
                                       "can not be converted to resize op");
  }
};

class ConvertNonTrivialConvOp
    : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  // TODO(b/302150407): we should move to use direct legalization to TFlite
  // instead going through TF in the future
  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SetDefaultConvAttributes(conv_op, rewriter);
    if (IsSupportedNonTrivialConvOp(conv_op, rewriter).failed()) {
      return rewriter.notifyMatchFailure(
          conv_op, "doesn't support to convert to ConvBackpropInputOp");
    }

    int feature_group_count = conv_op.getFeatureGroupCount();
    // For depthwise and group convolutions, feature_group_count != 1
    if (feature_group_count != 1) {
      // Depthwise or Group convolution is not supported yet.
      return rewriter.notifyMatchFailure(
          conv_op, "group or depthwise convolution is not supported");
    }

    // Constructs strides array from lhs_dilation.
    // For example, [2, 3] -> [1, 2, 3, 1].
    SmallVector<int64_t, 4> strides({1});
    strides.append(
        conv_op.getLhsDilation().value().getValues<int64_t>().begin(),
        conv_op.getLhsDilation().value().getValues<int64_t>().end());
    strides.emplace_back(1);

    // Constructs dilation array.
    SmallVector<int64_t, 4> dilation;
    if (auto rhs_dilation = conv_op.getRhsDilation()) {
      // For example, [2, 3] -> [1, 2, 3, 1].
      dilation.emplace_back(1);
      dilation.append(rhs_dilation.value().getValues<int64_t>().begin(),
                      rhs_dilation.value().getValues<int64_t>().end());
      dilation.emplace_back(1);
    } else {
      // Default value
      dilation = {1, 1, 1, 1};
    }

    mhlo::ConvDimensionNumbersAttr dnums = conv_op.getDimensionNumbers();
    auto input_spatial_dims = dnums.getInputSpatialDimensions();
    int num_spatial_dims = input_spatial_dims.size();

    std::string padding;
    SmallVector<int64_t, 8> padding_array{0, 0};
    SmallVector<int64_t, 4> padding_attr_value(
        conv_op.getPadding().value().getValues<int64_t>().begin(),
        conv_op.getPadding().value().getValues<int64_t>().end());
    padding_array.append(padding_attr_value);
    padding_array.push_back(0);
    padding_array.push_back(0);

    if (IsValidPaddingForTransposedConv(conv_op, num_spatial_dims, strides,
                                        padding_array)) {
      padding = "VALID";
    } else {
      if (!IsSamePaddingForTransposedConv(conv_op, num_spatial_dims, strides)) {
        return rewriter.notifyMatchFailure(
            conv_op, "requires padding to be SAME or VALID");
      }
      padding = "SAME";
    }
    auto conv_input = InsertTranspose(
        conv_op.getLhs(), dnums.getInputBatchDimension(),
        dnums.getInputFeatureDimension(), dnums.getInputSpatialDimensions(),
        /*default_batch_dim=*/0,
        /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/1, num_spatial_dims, rewriter);

    // Mirror the filter in the spatial dimensions.
    mlir::Value reverse_filter_in = conv_op.getRhs();
    // If the kernel is with format anythoing other than HWOI, we
    // transpose it to [0,1,o,i] as the TF->TFL pass anticipates this and the
    // kernel format information will be lost once we legalize to TF
    if (!isKernelFormatHWOI(dnums)) {
      SmallVector<int64_t, 4> permutation;
      for (int64_t dim : dnums.getKernelSpatialDimensions()) {
        permutation.push_back(dim);
      }
      permutation.push_back(dnums.getKernelOutputFeatureDimension());
      permutation.push_back(dnums.getKernelInputFeatureDimension());

      auto filter_transposed = rewriter.create<mhlo::TransposeOp>(
          conv_op.getLoc(), conv_op.getRhs(),
          DenseIntElementsAttr::get(
              RankedTensorType::get({static_cast<int64_t>(permutation.size())},
                                    rewriter.getI64Type()),
              permutation));
      reverse_filter_in = filter_transposed;
    }

    // Lets hard-code the reverse indexes to be {0, 1} as the expectation is
    // that the kernel is always in HWOI format, with the above code.
    mhlo::ReverseOp filter = rewriter.create<mhlo::ReverseOp>(
        conv_op.getLoc(), reverse_filter_in, rewriter.getI64TensorAttr({0, 1}));

    // if output is not in [b, 0, 1, f] format, insert transpose to go back
    if (dnums.getOutputBatchDimension() != 0 ||
        dnums.getOutputFeatureDimension() != num_spatial_dims + 1) {
      std::vector<int64_t> transpose_order;
      transpose_order.resize(num_spatial_dims + 2);
      // the output always in b, 0, 1, f format
      transpose_order[dnums.getOutputBatchDimension()] = 0;
      transpose_order[dnums.getOutputFeatureDimension()] = num_spatial_dims + 1;
      for (size_t i = 0; i < num_spatial_dims; ++i) {
        transpose_order[dnums.getOutputSpatialDimensions().data()[i]] = i + 1;
      }
      auto output_shape =
          mlir::cast<RankedTensorType>(conv_op.getResult().getType())
              .getShape();
      SmallVector<int64_t, 4> transposed_output_shape = {
          output_shape[dnums.getOutputBatchDimension()],
          output_shape[dnums.getOutputSpatialDimensions().data()[0]],
          output_shape[dnums.getOutputSpatialDimensions().data()[1]],
          output_shape[dnums.getOutputFeatureDimension()]};
      // Converts int64_t to int32_t.
      SmallVector<int32_t, 4> transposed_output_shape_i32;
      for (int64_t dim : transposed_output_shape) {
        transposed_output_shape_i32.push_back(dim);
      }
      auto output_type = RankedTensorType::get(
          transposed_output_shape,
          mlir::cast<ShapedType>(conv_op.getRhs().getType()).getElementType());
      auto output_sizes = rewriter.create<TF::ConstOp>(
          conv_op.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {static_cast<int64_t>(transposed_output_shape_i32.size())},
                  rewriter.getI32Type()),
              transposed_output_shape_i32));
      auto new_conv = rewriter.create<TF::Conv2DBackpropInputOp>(
          conv_op.getLoc(), output_type, output_sizes, filter, conv_input,
          rewriter.getI64ArrayAttr(strides),
          /*use_cudnn_on_gpu=*/rewriter.getBoolAttr(true),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
      auto output_transpose = rewriter.create<mhlo::TransposeOp>(
          conv_op.getLoc(), new_conv.getResult(),
          rewriter.getI64TensorAttr(transpose_order));
      conv_op->replaceAllUsesWith(output_transpose);
      rewriter.eraseOp(conv_op);
    } else {
      SmallVector<int32_t, 4> output_shape_i32;
      for (int64_t dim :
           mlir::cast<RankedTensorType>(conv_op.getResult().getType())
               .getShape()) {
        output_shape_i32.push_back(dim);
      }
      auto output_sizes = rewriter.create<TF::ConstOp>(
          conv_op.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get(
                  {static_cast<int64_t>(output_shape_i32.size())},
                  rewriter.getI32Type()),
              output_shape_i32));
      rewriter.replaceOpWithNewOp<TF::Conv2DBackpropInputOp>(
          conv_op, conv_op.getType(), output_sizes, filter, conv_input,
          rewriter.getI64ArrayAttr(strides),
          /*use_cudnn_on_gpu=*/rewriter.getBoolAttr(true),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    }
    return success();
  };

 private:
  // Utility function to check if the padding on the mhlo.convolution op
  // equals to the padding needed on a transpose_conv for the same
  // inputs/outputs, to achieve the effect of VALID padding.
  bool IsValidPaddingForTransposedConv(mhlo::ConvolutionOp conv_op,
                                       size_t num_spatial_dims,
                                       ArrayRef<int64_t> strides,
                                       ArrayRef<int64_t> padding) const {
    auto dnums = conv_op.getDimensionNumbers();
    // The newly added spatial dimension requires zero left and right padding.
    ArrayRef<int64_t> input_spatial_dims = dnums.getInputSpatialDimensions();
    ArrayRef<int64_t> kernel_spatial_dims = dnums.getKernelSpatialDimensions();
    ArrayRef<int64_t> output_spatial_dims = dnums.getOutputSpatialDimensions();

    for (size_t i = 1; i <= num_spatial_dims; ++i) {
      int64_t stride = strides[i];
      int64_t input_size = mlir::cast<ShapedType>(conv_op.getLhs().getType())
                               .getDimSize(input_spatial_dims[i - 1]);
      int64_t kernel_size = mlir::cast<ShapedType>(conv_op.getRhs().getType())
                                .getDimSize(kernel_spatial_dims[i - 1]);
      int64_t output_size = mlir::cast<ShapedType>(conv_op.getType())
                                .getDimSize(output_spatial_dims[i - 1]);

      // stablehlo.convolution op needs explicit padding to be set to model any
      // Transposed-Convolution in JAX/PT. Checking to see if-
      // 1. Pre set padding matches to the desired padding
      // 2. Output size respects the `VALID` padding scenario
      if ((padding[2 * i] == padding[2 * i + 1]) &&
          (((kernel_size - 1) != padding[2 * i]) ||
           (output_size != (stride * (input_size - 1)) + kernel_size))) {
        // padding[2 * i] == padding[2 * i + 1] means equal padding is applied
        // on both sides of a spatial dimension.
        // This happens when kernel_dim >= stride
        return false;
      } else if ((padding[2 * i] != padding[2 * i + 1]) &&
                 (((kernel_size - 1) != padding[2 * i]) ||
                  ((stride - 1) != padding[2 * i + 1]) ||
                  (output_size != (stride * input_size)))) {
        return false;
      }
    }

    return true;
  }

  bool IsSamePaddingForTransposedConv(mhlo::ConvolutionOp conv_op,
                                      size_t num_spatial_dims,
                                      ArrayRef<int64_t> strides) const {
    auto dnums = conv_op.getDimensionNumbers();
    SmallVector<int64_t, 4> padding(
        conv_op.getPadding().value().getValues<int64_t>().begin(),
        conv_op.getPadding().value().getValues<int64_t>().end());
    // The newly added spatial dimension requires zero left and right padding.
    ArrayRef<int64_t> input_spatial_dims = dnums.getInputSpatialDimensions();
    ArrayRef<int64_t> output_spatial_dims = dnums.getOutputSpatialDimensions();
    for (size_t i = 0; i < num_spatial_dims; ++i) {
      // In some cases the total padding is odd, so we have 1 leftover, which is
      // why below we check pad_delta > 1.
      int64_t pad_delta = std::abs(padding[2 * i] - padding[2 * i + 1]);
      if (pad_delta > 1) {
        return false;
      }
      int64_t stride = strides[i + 1];
      int64_t input_size = mlir::cast<ShapedType>(conv_op.getLhs().getType())
                               .getDimSize(input_spatial_dims[i]);
      int64_t output_size = mlir::cast<ShapedType>(conv_op.getType())
                                .getDimSize(output_spatial_dims[i]);
      // The reason for the below check is as follows:
      // When computing the output, we have the following relation between
      // o - output dim size, i - input dim size, s - stride, P - total pads
      // o = (i-k+1) + (s-1)(i-1) + P
      // Where the first term is the kernel applications on the input,
      // the second term is the additional applications from the stride
      // and P is a term that captures the total padding. After expanding we get
      // o = si + k - s + 2 + P
      // Here JAX sets P to cancel k-s+2, leading to the expression below
      if (output_size != input_size * stride) {
        return false;
      }
    }
    return true;
  }

  bool isKernelFormatHWOI(mhlo::ConvDimensionNumbersAttr dnums) const {
    int64_t num_spatial_dims = dnums.getKernelSpatialDimensions().size();
    return dnums.getKernelInputFeatureDimension() == num_spatial_dims + 1 &&
           dnums.getKernelOutputFeatureDimension() == num_spatial_dims;
  }
};

class ConvertSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto begin = rewriter.create<TF::ConstOp>(slice_op.getLoc(),
                                              slice_op.getStartIndices());
    auto end = rewriter.create<TF::ConstOp>(slice_op.getLoc(),
                                            slice_op.getLimitIndices());
    auto strides =
        rewriter.create<TF::ConstOp>(slice_op.getLoc(), slice_op.getStrides());
    rewriter.replaceOpWithNewOp<TF::StridedSliceOp>(
        slice_op, slice_op.getType(), slice_op.getOperand(), begin, end,
        strides);
    return success();
  }
};

class ConvertDynamicSliceOp : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType input_type = mlir::cast<ShapedType>(op.getOperand().getType());
    if (!input_type.hasStaticShape()) return failure();
    Type start_indices_element_type =
        mlir::cast<ShapedType>(op.getStartIndices().front().getType())
            .getElementType();

    // The mhlo dynamic_slice's start_indices can be either signed/unsigned
    // int32/int64. However, TF only takes in either i32 or i64 types for begin,
    // so we will always put a cast.
    Type signed_start_indices_element_type;
    if (start_indices_element_type.isInteger(32)) {
      signed_start_indices_element_type = rewriter.getI32Type();
    } else {
      signed_start_indices_element_type = rewriter.getI64Type();
    }

    // Clamp indices to [0, input_size - output_size]
    llvm::SmallVector<Value, 4> start_indices_vector;
    start_indices_vector.reserve(op.getStartIndices().size());
    Value clamp_min = rewriter.create<TF::ConstOp>(
        op.getLoc(),
        rewriter.getIntegerAttr(signed_start_indices_element_type, 0));
    for (uint64_t i = 0, e = op.getStartIndices().size(); i < e; ++i) {
      // Always put a cast there.
      auto start = op.getStartIndices()[i];
      auto cast_type = mlir::cast<ShapedType>(start.getType())
                           .clone(signed_start_indices_element_type);
      auto cast_op = rewriter.create<TF::CastOp>(op.getLoc(), cast_type, start);
      Value clamp_max = rewriter.create<TF::ConstOp>(
          op.getLoc(), rewriter.getIntegerAttr(
                           signed_start_indices_element_type,
                           input_type.getShape()[i] -
                               op.getSliceSizes().getValues<int64_t>()[i]));
      Value clamped_index = rewriter.create<mhlo::ClampOp>(
          op.getLoc(), cast_type, clamp_min, cast_op, clamp_max);
      start_indices_vector.push_back(clamped_index);
    }

    // Pack individual start indices to start indices tensor.
    Type start_indices_type = RankedTensorType::get(
        {static_cast<int64_t>(start_indices_vector.size())},
        signed_start_indices_element_type);
    Value start_indices_op = rewriter.create<TF::PackOp>(
        op.getLoc(), start_indices_type, ValueRange(start_indices_vector));

    Value slice_sices_op =
        rewriter.create<TF::ConstOp>(op.getLoc(), op.getSliceSizes());
    rewriter.replaceOpWithNewOp<TF::SliceOp>(op, op.getType(), op.getOperand(),
                                             start_indices_op, slice_sices_op);
    return success();
  };
};

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range>
void Append(llvm::SmallVectorImpl<ValueT>& values, Range&& range) {
  values.insert(values.end(), range.begin(), range.end());
}

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range, typename... RangeTs>
void Append(llvm::SmallVectorImpl<ValueT>& values, Range&& range,
            RangeTs&&... ranges) {
  values.insert(values.end(), range.begin(), range.end());
  Append(values, ranges...);
}

// Returns the number of elements in `range`.
template <typename Range>
size_t Size(Range&& range) {
  return range.size();
}

// Returns the total number of elements in a variadic number of `ranges`.
template <typename Range, typename... RangeTs>
size_t Size(Range&& range, RangeTs&&... ranges) {
  return range.size() + Size(std::forward<RangeTs>(ranges)...);
}

// Concats all elements in `ranges` and returns a small vector as a result.
template <typename ValueT, typename... RangeTs>
llvm::SmallVector<ValueT, 4> Concat(RangeTs&&... ranges) {
  llvm::SmallVector<int64_t, 4> results;
  results.reserve(Size(std::forward<RangeTs>(ranges)...));
  Append(results, std::forward<RangeTs>(ranges)...);
  return results;
}

// A struct to hold axes and sizes for a set of dimensions.
struct DimensionVector {
  llvm::ArrayRef<int64_t> AxesArray() const { return axes; }
  llvm::ArrayRef<int64_t> SizesArray() const { return sizes; }

  llvm::SmallVector<int64_t, 4> axes;
  llvm::SmallVector<int64_t, 4> sizes;
};

// Create a tensor that is reshaped from input.
Value BuildReshapeOp(ImplicitLocOpBuilder& builder,
                     ConversionPatternRewriter& rewriter, Value input,
                     ArrayRef<int64_t> shape, Type idx_type,
                     Type element_type) {
  Value shape_cst = BuildIntArrayConstOp(builder, rewriter, shape, idx_type);
  Value reshaped_input = builder.create<TF::ReshapeOp>(
      RankedTensorType::get(shape, element_type), input, shape_cst);
  return reshaped_input;
}

// Create a tensor which is equal to input[begin: begin + size].
Value BuildSliceOp(ImplicitLocOpBuilder& builder,
                   ConversionPatternRewriter& rewriter, Value input,
                   Value begin, ArrayRef<int64_t> shape, Type idx_type,
                   Type element_type) {
  Value shape_cst = BuildIntArrayConstOp(builder, rewriter, shape, idx_type);
  Value slice_result = builder.create<TF::SliceOp>(
      RankedTensorType::get(shape, element_type), input, begin, shape_cst);
  return slice_result;
}

class ConvertDynamicUpdateSliceOp
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType operand_type = mlir::cast<ShapedType>(op.getOperand().getType());
    ShapedType update_type =
        mlir::dyn_cast_or_null<ShapedType>(op.getUpdate().getType());
    ShapedType start_indices_type = mlir::dyn_cast_or_null<ShapedType>(
        op.getStartIndices().front().getType());
    if (update_type == nullptr || start_indices_type == nullptr)
      return rewriter.notifyMatchFailure(
          op, "update and start_indices should have ShapedType");

    Type idx_type = start_indices_type.getElementType();
    int64_t shape_dim = operand_type.getRank();
    llvm::SmallVector<Value> start_indices_vector;
    Append(start_indices_vector, op.getStartIndices());
    auto shape_tensor_type = RankedTensorType::get({shape_dim}, idx_type);
    Value start_indices_tensor = rewriter.create<TF::PackOp>(
        op.getLoc(), shape_tensor_type, start_indices_vector);
    rewriter.replaceOpWithNewOp<TF::XlaDynamicUpdateSliceOp>(
        op, op.getType(), op.getOperand(), op.getUpdate(),
        start_indices_tensor);
    return success();
  };
};

template <typename ReturnOpType>
bool MatchTopKComparator(Region& comparator) {
  if (!comparator.hasOneBlock()) return false;
  Block& comparator_blk = comparator.front();
  using OpListType = llvm::iplist<Operation>;
  OpListType& operations = comparator_blk.getOperations();
  if (operations.size() != 2) return false;
  auto compare_op = dyn_cast_or_null<mhlo::CompareOp>(&operations.front());
  auto return_op = dyn_cast_or_null<ReturnOpType>(&operations.back());
  if (!compare_op || !return_op) return false;
  // TODO(xuanyuanluo): Support mhlo::ComparisonDirection::LT direction.
  if (compare_op.getComparisonDirection() != mhlo::ComparisonDirection::GT) {
    return false;
  }
  if (compare_op.getOperands()[0] != comparator_blk.getArgument(0) ||
      compare_op.getOperands()[1] != comparator_blk.getArgument(1)) {
    return false;
  }
  return return_op.getOperands().front() == compare_op.getResult();
}

// In general, we convert the following form of sort to tf.TopK:
//
// %result = "mhlo.sort" (%keys, %indices) ({
//  ^bb0(%key_0, %key_1, %index_0, %index_1):
//     %1 = "mhlo.compare"(%key_0, %key_1) {mhlo::ComparisonDirection::GT}
//     -> tensor<i1>
//  }),
//
// where the indices is obtained by an IotaOp (maybe folded).
class ConvertSortToTfTopk : public OpConversionPattern<mhlo::SortOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SortOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op->getOperands().size() != 2)
      return rewriter.notifyMatchFailure(
          op, "only match for the case where operands is of size 2");
    auto keys = op.getInputs()[0];
    auto indices = op.getInputs()[1];
    auto keys_ty = mlir::dyn_cast_or_null<ShapedType>(keys.getType());
    auto indices_ty = mlir::dyn_cast_or_null<ShapedType>(indices.getType());
    if (!keys_ty || !keys_ty.hasStaticShape() ||
        !keys_ty.getElementType().isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op,
          "only match for the case where the first operand has a static "
          "int/float shapeType");
    if (!indices_ty || !indices_ty.hasStaticShape() ||
        !indices_ty.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op,
          "only match for the case where the second operand an I32 shapeType");
    auto sort_dim = op.getDimension();
    auto k = indices_ty.getDimSize(sort_dim);
    auto rank = keys_ty.getRank();
    if (sort_dim != rank - 1 || k < 1)
      return rewriter.notifyMatchFailure(
          op, "only match for sort dim = rank - 1 and DimSize >= 1");

    // In the following, we'll check indices is obtained by a iota.
    auto sort_dim_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getI64Type()), {sort_dim});
    if (!MatchIota(sort_dim_attr, indices))
      return rewriter.notifyMatchFailure(
          op, "the second operand is supposed to be obtained from IOTA");
    if (!MatchTopKComparator<mhlo::ReturnOp>(op.getComparator()))
      return rewriter.notifyMatchFailure(op, "only match for GT comparator");
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    Value k_cst = BuildIntConstOp(builder, rewriter, k, rewriter.getI32Type());
    rewriter.replaceOpWithNewOp<TF::TopKV2Op>(op, keys.getType(),
                                              indices.getType(), keys, k_cst);
    return success();
  };
};

// A struct to hold information about dimensions of dot_general operands.
class DotDimensionsInfo {
 public:
  DotDimensionsInfo(ShapedType type, ArrayRef<int64_t> batch_dimensions,
                    ArrayRef<int64_t> contracting_dimensions) {
    const int64_t rank = type.getRank();
    for (const int64_t dim : batch_dimensions) {
      batch_dimensions_.axes.push_back(dim);
      batch_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (const int64_t dim : contracting_dimensions) {
      contracting_dimensions_.axes.push_back(dim);
      contracting_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (llvm::count(contracting_dimensions_.axes, dim) > 0 ||
          llvm::count(batch_dimensions_.axes, dim) > 0) {
        continue;
      }
      out_dimensions_.axes.push_back(dim);
      out_dimensions_.sizes.push_back(type.getDimSize(dim));
    }
  }

  const DimensionVector& batch_dimensions() const { return batch_dimensions_; }
  const DimensionVector& contracting_dimensions() const {
    return contracting_dimensions_;
  }
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  const DimensionVector& out_dimensions() const { return out_dimensions_; }

  // Returns the total dimension size after flattening all contracting
  // dimensions.
  int64_t FlattenedContractingDimensionSize() const {
    if (ShapedType::isDynamicShape(contracting_dimensions_.sizes)) {
      return ShapedType::kDynamic;
    }
    return std::accumulate(contracting_dimensions_.sizes.begin(),
                           contracting_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

  // Returns the total dimension size after flattening all out dimensions.
  int64_t FlattenedOutDimensionSize() const {
    if (ShapedType::isDynamicShape(out_dimensions_.sizes)) {
      return ShapedType::kDynamic;
    }
    return std::accumulate(out_dimensions_.sizes.begin(),
                           out_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

 private:
  DimensionVector batch_dimensions_;
  DimensionVector contracting_dimensions_;
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  DimensionVector out_dimensions_;
};

// Calculates the flattened shapes for dynamic shaped operands in
// mhlo.dot_general:
//   1. flattened_out_dim = UnsortedSegmentProdOp(operand_shape, out_axes)
//   2. flattened_contracting_dim = UnsortedSegmentProdOp(operand_shape,
//   contracting_axes)
//   3. batch_dimensions = Gather(operand_shape, batch_axes)
//   4. flattened_shape = Concat(batch_dimensions, flattened_out_dim,
//   flattened_contracting_dim)
// The flattened shape for LHS
// is like [batch_dimensions, flattened_out_dimension,
// flattened_contracting_dimension] and [batch_dimensions,
// flattened_contracting_dimension, flattened_out_dimension] for RHS.
Value BuildDotOperandFlattenedShapeOp(Value operand,
                                      DotDimensionsInfo dot_dimensions_info,
                                      ImplicitLocOpBuilder& builder,
                                      bool is_lhs) {
  auto operand_type = mlir::cast<ShapedType>(operand.getType());
  BoolAttr true_attr = builder.getBoolAttr(true);
  auto operand_shape = builder.create<TF::ShapeOp>(operand, true_attr);
  const int64_t operand_rank = operand_type.getRank();
  // Compute flattened out dimension and contracting dimension using
  // TF::UnsortedSegmentProdOp.
  llvm::SmallVector<int32_t, 4> flattened_out_segids =
      llvm::SmallVector<int32_t, 4>(operand_rank, static_cast<int32_t>(-1));
  for (int64_t i : dot_dimensions_info.out_dimensions().AxesArray()) {
    flattened_out_segids[i] = 0;
  }
  llvm::SmallVector<int32_t, 4> flattened_contracting_segids =
      llvm::SmallVector<int32_t, 4>(operand_rank, static_cast<int32_t>(-1));
  for (int64_t i : dot_dimensions_info.contracting_dimensions().AxesArray()) {
    flattened_contracting_segids[i] = 0;
  }
  auto seg_prod_result_type =
      RankedTensorType::get(static_cast<int32_t>(1), builder.getI32Type());
  auto out_segids_cst = builder.create<TF::ConstOp>(
      builder.getI32TensorAttr(flattened_out_segids));
  auto contracting_segids_cst = builder.create<TF::ConstOp>(
      builder.getI32TensorAttr(flattened_contracting_segids));
  auto num_segids_tensor =
      builder.create<TF::ConstOp>(builder.getI32IntegerAttr(1));
  auto flattened_out_dims = builder.create<TF::UnsortedSegmentProdOp>(
      seg_prod_result_type, operand_shape, out_segids_cst, num_segids_tensor);
  auto flattened_contracting_dims = builder.create<TF::UnsortedSegmentProdOp>(
      seg_prod_result_type, operand_shape, contracting_segids_cst,
      num_segids_tensor);
  llvm::SmallVector<Value, 3> flattend_shape_values;
  // Gather the batch dimensions.
  if (!dot_dimensions_info.batch_dimensions().AxesArray().empty()) {
    if (ShapedType::isDynamicShape(
            dot_dimensions_info.batch_dimensions().SizesArray())) {
      auto batch_axes_tensor =
          builder.create<TF::ConstOp>(builder.getI64TensorAttr(
              dot_dimensions_info.batch_dimensions().AxesArray()));
      auto batch_dims = builder.create<TF::GatherOp>(
          RankedTensorType::get(
              {static_cast<int>(
                  dot_dimensions_info.batch_dimensions().AxesArray().size())},
              builder.getIntegerType(32)),
          operand_shape, batch_axes_tensor, true_attr);
      flattend_shape_values.push_back(batch_dims);
    } else {
      llvm::SmallVector<int32_t> batch_i32_vec;
      for (int64_t element :
           dot_dimensions_info.batch_dimensions().SizesArray()) {
        batch_i32_vec.push_back(static_cast<int32_t>(element));
      }
      auto batch_dims =
          builder.create<TF::ConstOp>(builder.getI32TensorAttr(batch_i32_vec));
      flattend_shape_values.push_back(batch_dims);
    }
  }
  flattend_shape_values.push_back(
      (is_lhs ? flattened_out_dims : flattened_contracting_dims));
  flattend_shape_values.push_back(
      (is_lhs ? flattened_contracting_dims : flattened_out_dims));

  auto concat_result_type = RankedTensorType::get(
      {static_cast<int32_t>(
           dot_dimensions_info.batch_dimensions().AxesArray().size()) +
       2},
      builder.getIntegerType(32));
  // Concatenate the batch dimensions, flattened out dimension and flattened
  // contracting dimension.
  return builder.create<TF::ConcatOp>(
      concat_result_type,
      builder.create<TF::ConstOp>(builder.getI32IntegerAttr(0)),
      flattend_shape_values);
}

Value ConvertDot(PatternRewriter& rewriter, Value lhs, Value rhs,
                 DotDimensionNumbersAttr dot_dimension_numbers,
                 ShapedType result_type, mlir::Location loc) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());
  auto rhs_type = mlir::cast<ShapedType>(rhs.getType());
  const int lhs_rank = lhs_type.getRank();
  const int rhs_rank = rhs_type.getRank();
  ImplicitLocOpBuilder builder(loc, rewriter);

  // Collects lhs and rhs dimensions information.
  DotDimensionsInfo lhs_dot_dimensions_info(
      lhs_type, dot_dimension_numbers.getLhsBatchingDimensions(),
      dot_dimension_numbers.getLhsContractingDimensions());
  DotDimensionsInfo rhs_dot_dimensions_info(
      rhs_type, dot_dimension_numbers.getRhsBatchingDimensions(),
      dot_dimension_numbers.getRhsContractingDimensions());

  // Transposes lhs shape to be in the order of {batch_dimensions,
  // out_dimensions, contracting dimensions}.
  llvm::SmallVector<int64_t, 4> lhs_permutation = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().AxesArray(),
      lhs_dot_dimensions_info.out_dimensions().AxesArray(),
      lhs_dot_dimensions_info.contracting_dimensions().AxesArray());
  llvm::SmallVector<int64_t, 4> lhs_transposed_shape = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      lhs_dot_dimensions_info.out_dimensions().SizesArray(),
      lhs_dot_dimensions_info.contracting_dimensions().SizesArray());
  auto lhs_transposed = rewriter.create<mhlo::TransposeOp>(
      loc,
      RankedTensorType::get(lhs_transposed_shape, lhs_type.getElementType()),
      lhs,
      DenseIntElementsAttr::get(
          RankedTensorType::get({lhs_rank}, rewriter.getI64Type()),
          lhs_permutation));

  // Transposes rhs shape to be in the order of {batch_dimensions, contracting
  // dimensions, out_dimensions}.
  llvm::SmallVector<int64_t, 4> rhs_permutation = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().AxesArray(),
      rhs_dot_dimensions_info.contracting_dimensions().AxesArray(),
      rhs_dot_dimensions_info.out_dimensions().AxesArray());
  llvm::SmallVector<int64_t, 4> rhs_transposed_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      rhs_dot_dimensions_info.contracting_dimensions().SizesArray(),
      rhs_dot_dimensions_info.out_dimensions().SizesArray());
  auto rhs_transposed = rewriter.create<mhlo::TransposeOp>(
      loc,
      RankedTensorType::get(rhs_transposed_shape, rhs_type.getElementType()),
      rhs,
      DenseIntElementsAttr::get(
          RankedTensorType::get({rhs_rank}, rewriter.getI64Type()),
          rhs_permutation));

  // Reshapes lhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> lhs_flattened_shape = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
      llvm::ArrayRef<int64_t>{
          lhs_dot_dimensions_info.FlattenedContractingDimensionSize()});
  Value lhs_flattend;
  if (lhs_type.hasStaticShape()) {
    lhs_flattend = rewriter.create<mhlo::ReshapeOp>(
        loc,
        RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
        lhs_transposed.getResult());
  } else {
    auto lhs_flattend_shape_op = BuildDotOperandFlattenedShapeOp(
        lhs, lhs_dot_dimensions_info, builder, /*is_lhs=*/true);
    lhs_flattend = rewriter.create<mhlo::DynamicReshapeOp>(
        loc,
        RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
        lhs_transposed, lhs_flattend_shape_op);
  }

  // Reshapes rhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> rhs_flattened_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedContractingDimensionSize()},
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  Value rhs_flattend;
  if (rhs_type.hasStaticShape()) {
    rhs_flattend = rewriter.create<mhlo::ReshapeOp>(
        loc,
        RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
        rhs_transposed.getResult());
  } else {
    auto rhs_flattend_shape_op = BuildDotOperandFlattenedShapeOp(
        rhs, rhs_dot_dimensions_info, builder, /*is_lhs=*/false);
    rhs_flattend = rewriter.create<mhlo::DynamicReshapeOp>(
        loc,
        RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
        rhs_transposed, rhs_flattend_shape_op);
  }

  // Creates matmul op of `lhs_flattend` and `rhs_flattend`.
  llvm::SmallVector<int64_t, 4> matmul_shape =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
                      llvm::ArrayRef<int64_t>{
                          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
                      llvm::ArrayRef<int64_t>{
                          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  auto matmul = rewriter.create<TF::BatchMatMulV3Op>(
      loc, RankedTensorType::get(matmul_shape, result_type.getElementType()),
      lhs_flattend, rhs_flattend);

  if (result_type.hasStaticShape()) {
    auto reshaped =
        rewriter.create<mhlo::ReshapeOp>(loc, result_type, matmul.getResult());
    return reshaped.getResult();
  }

  // Reshape for dynamic shaped operands. The result shape is
  // [lhs_batch_dimensions, lhs_out_dimensions, rhs_out_dimensions].
  BoolAttr true_attr = rewriter.getBoolAttr(true);
  auto lhs_shape = rewriter.create<TF::ShapeOp>(loc, lhs, true_attr);
  auto rhs_shape = rewriter.create<TF::ShapeOp>(loc, rhs, true_attr);
  llvm::SmallVector<int64_t, 4> lhs_batch_and_out =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().AxesArray(),
                      lhs_dot_dimensions_info.out_dimensions().AxesArray());
  auto lhs_batch_and_out_cst = rewriter.create<TF::ConstOp>(
      loc, rewriter.getI64TensorAttr(lhs_batch_and_out));
  auto lhs_batch_and_out_dims = rewriter.create<TF::GatherOp>(
      loc,
      RankedTensorType::get({static_cast<int>(lhs_batch_and_out.size())},
                            rewriter.getIntegerType(32)),
      lhs_shape, lhs_batch_and_out_cst, true_attr);
  auto rhs_out_cst = rewriter.create<TF::ConstOp>(
      loc, rewriter.getI64TensorAttr(
               rhs_dot_dimensions_info.out_dimensions().AxesArray()));
  auto rhs_out_dims = rewriter.create<TF::GatherOp>(
      loc,
      RankedTensorType::get(
          {static_cast<int32_t>(
              rhs_dot_dimensions_info.out_dimensions().AxesArray().size())},
          rewriter.getIntegerType(32)),
      rhs_shape, rhs_out_cst, true_attr);
  auto result_shape_type = RankedTensorType::get(
      {static_cast<int32_t>(
          lhs_dot_dimensions_info.batch_dimensions().AxesArray().size() +
          lhs_dot_dimensions_info.out_dimensions().AxesArray().size() +
          rhs_dot_dimensions_info.out_dimensions().AxesArray().size())},
      rewriter.getIntegerType(32));
  auto result_shape = rewriter.create<TF::ConcatOp>(
      loc, result_shape_type,
      rewriter.create<TF::ConstOp>(loc, rewriter.getI32IntegerAttr(0)),
      ValueRange{lhs_batch_and_out_dims, rhs_out_dims});

  auto reshaped = rewriter.create<mhlo::DynamicReshapeOp>(
      loc, result_type, matmul.getResult(), result_shape);
  return reshaped.getResult();
}

// Converts mhlo.dot to tf.MatMul. Reshape ops will be inserted when
// necessary.
Value ConvertDotOp(PatternRewriter& rewriter, Operation* old_op) {
  auto dot_op = cast<mhlo::DotOp>(old_op);
  auto lhs_rank = mlir::cast<ShapedType>(dot_op.getLhs().getType()).getRank();
  auto dot_dimension_numbers =
      DotDimensionNumbersAttr::get(rewriter.getContext(),
                                   /*lhs_batching_dimensions=*/{},
                                   /*rhs_batching_dimensions=*/{},
                                   /*lhs_contracting_dimensions=*/
                                   {lhs_rank == 1 ? 0 : 1},
                                   /*rhs_contracting_dimensions=*/{0});
  return ConvertDot(
      rewriter, dot_op.getLhs(), dot_op.getRhs(), dot_dimension_numbers,
      mlir::cast<ShapedType>(dot_op.getResult().getType()), dot_op.getLoc());
}

// Converts mhlo.dot to tf.BatchMatMul. Reshape or Transpose ops will also be
// inserted to convert to well-formed matrix multiply.
Value ConvertDotGeneralOp(PatternRewriter& rewriter, Operation* old_op) {
  auto dot_general_op = cast<mhlo::DotGeneralOp>(old_op);
  return ConvertDot(
      rewriter, dot_general_op.getLhs(), dot_general_op.getRhs(),
      dot_general_op.getDotDimensionNumbers(),
      mlir::cast<ShapedType>(dot_general_op.getResult().getType()),
      dot_general_op.getLoc());
}

// Replace BinaryOp with a combination of TfBinaryOp and TfReduceOp if the
// init value doesn't match the expectation of TfReduceOp.
template <typename TfReduceOp, typename TfBinOp>
LogicalResult rewriteNonMatchInitValue(mhlo::ReduceOp reduce_op, Value input,
                                       TF::ConstOp reduction_indices,
                                       ConversionPatternRewriter& rewriter) {
  Value reduce_result = rewriter.create<TfReduceOp>(
      reduce_op.getLoc(), reduce_op.getType(0), input, reduction_indices,
      /*keep_dim=*/rewriter.getBoolAttr(false));
  rewriter.replaceOpWithNewOp<TfBinOp>(reduce_op, reduce_op.getType(0),
                                       reduce_result,
                                       reduce_op.getInitValues()[0]);
  return success();
}

// Cannot replace BinaryOp if the init value doesn't match the expectation of
// TfReduceOp and there is no corresponding TfBinaryOp.
template <>
LogicalResult rewriteNonMatchInitValue<TF::MaxOp, void>(
    mhlo::ReduceOp reduce_op, Value input, TF::ConstOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}

template <>
LogicalResult rewriteNonMatchInitValue<TF::MinOp, void>(
    mhlo::ReduceOp reduce_op, Value input, TF::ConstOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}

// Converts a mhlo.reduce op with a mlho binary operation into a tensorflow
// reduction operation. If the initial value can be ignored, then convert it
// into a single TfReduceOp. Otherwise, convert it into a TfReduceOp followed by
// a TfBinaryOp.
// For example:
//   1) A mhlo::ReduceOp on value `x` with a mhlo::AndOp and a constant initial
// value `true` is converted to a TF::Any on value `x`.
//   2) A mhlo::ReduceOp on value `x` with a mhlo::AndOp with a non-constant
// initial value `y` is converted to a TF::Any on value `x`, followed by a
// TF::And with initial value `y`.
template <typename BinaryOp, typename TfReduceOp, typename TfBinaryOp = void>
class ConvertReduceOpToTfOp : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(MatchReduceOpOperand(reduce_op))) return failure();

    if (failed(MatchBinaryReduceFunction<BinaryOp>(reduce_op.getBody())))
      return failure();

    auto operand = reduce_op.getInputs()[0];

    // Get reduction dimension.
    DenseIntElementsAttr dimension = reduce_op.getDimensions();
    SmallVector<int64_t, 4> reduce_dims;
    for (const int64_t& dim : dimension.getValues<int64_t>()) {
      reduce_dims.emplace_back(dim);
    }
    auto dim_type = RankedTensorType::get(
        {static_cast<int64_t>(reduce_dims.size())}, rewriter.getI64Type());
    auto reduction_indices = rewriter.create<TF::ConstOp>(
        reduce_op.getLoc(), dim_type, rewriter.getI64TensorAttr(reduce_dims));

    // In `MatchReduceOpOperand` function, we already match that the
    // "mhlo::ReduceOp" only has one operand, one init_value and one result.

    // If the init value matches with the init value expected for the target
    // TfReduceOp, then replace the BinaryOp with a TfReduceOp. Otherwise,
    // replace the BinaryOp with a TfBinaryOp and a TfReduceOp.
    if (succeeded(MatchInitValue(reduce_op.getInitValues()[0]))) {
      rewriter.replaceOpWithNewOp<TfReduceOp>(
          reduce_op, reduce_op.getType(0), operand, reduction_indices,
          /*keep_dim=*/rewriter.getBoolAttr(false));
      return success();
    }
    return rewriteNonMatchInitValue<TfReduceOp, TfBinaryOp>(
        reduce_op, operand, reduction_indices, rewriter);
  }

 private:
  // Checks that the init value matches with the init value expected for the
  // target TfReduceOp.
  virtual LogicalResult MatchInitValue(Value init_value) const = 0;

  // This function tries to match that the "mhlo::ReduceOp" only has one
  // operand, one init_value and one result.
  LogicalResult MatchReduceOpOperand(mhlo::ReduceOp reduce_op) const {
    if (reduce_op.getInputs().size() != 1 ||
        reduce_op.getInitValues().size() != 1 ||
        reduce_op.getResults().size() != 1)
      return failure();

    if (!mlir::isa<RankedTensorType>(reduce_op.getInputs()[0].getType()))
      return failure();
    if (!mlir::isa<RankedTensorType>(reduce_op.getType(0))) return failure();
    return success();
  }
};

class ConvertReduceOpToTfProd
    : public ConvertReduceOpToTfOp<mhlo::MulOp, TF::ProdOp, TF::MulOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      float const_value;
      if (failed(GetConstantSplatValue<float>(init_value, const_value)) ||
          const_value != 1.0)
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      int32_t const_value;
      if (failed(GetConstantSplatValue<int32_t>(init_value, const_value)) ||
          const_value != 1)
        return failure();
    } else {
      return failure();
    }

    return success();
  }
};

class ConvertReduceOpToTfSum
    : public ConvertReduceOpToTfOp<mhlo::AddOp, TF::SumOp, TF::AddOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else {
      return failure();
    }

    return success();
  }
};

class ConvertReduceOpToTfMax
    : public ConvertReduceOpToTfOp<mhlo::MaxOp, TF::MaxOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || !const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMinSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};

class ConvertReduceOpToTfMin
    : public ConvertReduceOpToTfOp<mhlo::MinOp, TF::MinOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();

    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMaxSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};

class ConvertReduceOpToTfAll
    : public ConvertReduceOpToTfOp<mhlo::AndOp, TF::AllOp, TF::LogicalAndOp> {
 public:
  using ConvertReduceOpToTfOp<mhlo::AndOp, TF::AllOp,
                              TF::LogicalAndOp>::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || !init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};

class ConvertReduceOpToTfAny
    : public ConvertReduceOpToTfOp<mhlo::OrOp, TF::AnyOp, TF::LogicalOrOp> {
 public:
  using ConvertReduceOpToTfOp<mhlo::OrOp, TF::AnyOp,
                              TF::LogicalOrOp>::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};

class ConvertReduceOpToTfArgmax
    : public ConvertReduceOpToArgMinMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp,
                                        true> {
 public:
  using ConvertReduceOpToArgMinMax::ConvertReduceOpToArgMinMax;

  bool IsValueInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (mlir::isa<FloatType>(element_type)) {
      auto value = *attr.value_begin<APFloat>();
      return value.isNegative() && value.isInfinity();
    } else if (element_type.isInteger(1)) {
      auto value = *attr.value_begin<APInt>();
      return value.isZero();
    } else {
      auto value = *attr.value_begin<APInt>();
      return element_type.isUnsignedInteger() ? value.isMinValue()
                                              : value.isMinSignedValue();
    }
  }
};

class ConvertReduceOpToTfArgmin
    : public ConvertReduceOpToArgMinMax<TF::MinOp, TF::ArgMinOp, TF::AllOp,
                                        false> {
 public:
  using ConvertReduceOpToArgMinMax::ConvertReduceOpToArgMinMax;

  bool IsValueInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (mlir::isa<FloatType>(element_type)) {
      auto value = *attr.value_begin<APFloat>();
      return !value.isNegative() && value.isInfinity();
    } else if (element_type.isInteger(1)) {
      auto value = *attr.value_begin<APInt>();
      return value.isZero();
    } else {
      auto value = *attr.value_begin<APInt>();
      return element_type.isUnsignedInteger() ? value.isMaxValue()
                                              : value.isMaxSignedValue();
    }
  }
};

class ConvertIotaOpToTfRange : public OpConversionPattern<mhlo::IotaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IotaOp iota_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    RankedTensorType type =
        mlir::dyn_cast_or_null<RankedTensorType>(iota_op.getType());
    // TF::RangeOp doesn't support UI16.
    if (!type || type.getElementType().isUnsignedInteger(16)) return failure();

    const uint64_t dimension = iota_op.getIotaDimension();
    Type element_type = type.getElementType();
    Attribute start, limit, delta;
    if (mlir::isa<FloatType>(element_type)) {
      start = rewriter.getFloatAttr(element_type, 0.0);
      limit = rewriter.getFloatAttr(element_type, type.getShape()[dimension]);
      delta = rewriter.getFloatAttr(element_type, 1.0);
    } else if (mlir::isa<IntegerType>(element_type)) {
      start = rewriter.getIntegerAttr(element_type, 0);
      limit = rewriter.getIntegerAttr(element_type, type.getShape()[dimension]);
      delta = rewriter.getIntegerAttr(element_type, 1);
    } else {
      return failure();
    }

    auto range_type =
        RankedTensorType::get({type.getShape()[dimension]}, element_type);
    Value start_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), start);
    Value limit_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), limit);
    Value delta_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), delta);
    Value result = rewriter.create<TF::RangeOp>(iota_op.getLoc(), range_type,
                                                start_op, limit_op, delta_op);

    if (type.getRank() > 1) {
      std::vector<int64_t> reshape_shape(type.getRank(), 1);
      reshape_shape[iota_op.getIotaDimension()] = type.getShape()[dimension];
      auto reshape_type = RankedTensorType::get(reshape_shape, element_type);
      Value reshape_shape_op = rewriter.create<TF::ConstOp>(
          iota_op.getLoc(), rewriter.getI64TensorAttr(reshape_shape));
      result = rewriter.create<TF::ReshapeOp>(iota_op.getLoc(), reshape_type,
                                              result, reshape_shape_op);

      Value broadcast_shape_op = rewriter.create<TF::ConstOp>(
          iota_op.getLoc(), rewriter.getI64TensorAttr(type.getShape()));
      result = rewriter.create<TF::BroadcastToOp>(iota_op.getLoc(), type,
                                                  result, broadcast_shape_op);
    }

    rewriter.replaceOp(iota_op, result);
    return success();
  }
};

// A helper function for ConvertMaxPoolOp and ConvertAvgPoolOp. Returns true
// if the given ReduceWindowOp is a spatial pooling without dilation. If returns
// true, also outputs the window strides and the TF padding mode ("VALID" or
// "SAME").
bool IsSpatialPoolingWithoutDilation(
    mhlo::ReduceWindowOp rw, llvm::SmallVectorImpl<int64_t>* window_strides,
    std::string* padding_mode, std::string* data_format) {
  // tf.max_pool or tf.avg_pool need at least 3 dimensions (batch, spatial,
  // channel).
  const uint64_t rank = rw.getWindowDimensions().size();
  if (rank <= 3 || rank > 5) return false;

  if (rw.getWindowStrides().has_value()) {
    window_strides->insert(window_strides->end(),
                           rw.getWindowStrides()->getValues<int64_t>().begin(),
                           rw.getWindowStrides()->getValues<int64_t>().end());
  } else {
    window_strides->resize(rank, 1);
  }

  llvm::SmallVector<int64_t, 10> padding;
  if (rw.getPadding().has_value()) {
    padding.insert(padding.begin(),
                   rw.getPadding()->getValues<int64_t>().begin(),
                   rw.getPadding()->getValues<int64_t>().end());
  } else {
    padding.resize(2 * rank, 0);
  }

  // Check that we don't do any reduction along the batch and channel
  // dimensions.
  auto verify_batch_channel_dims = [&rw, &window_strides, &padding](
                                       uint64_t batch_dim,
                                       uint64_t channel_dim) {
    return rw.getWindowDimensions().getValues<int64_t>()[batch_dim] == 1 &&
           rw.getWindowDimensions().getValues<int64_t>()[channel_dim] == 1 &&
           (*window_strides)[batch_dim] == 1 &&
           (*window_strides)[channel_dim] == 1 && padding[2 * batch_dim] == 0 &&
           padding[2 * batch_dim + 1] == 0 && padding[2 * channel_dim] == 0 &&
           padding[2 * channel_dim + 1] == 0;
  };

  bool is_pool2d = rank == 4;
  if (verify_batch_channel_dims(0, rank - 1)) {
    *data_format = is_pool2d ? "NHWC" : "NDHWC";
  } else if (verify_batch_channel_dims(0, 1)) {
    *data_format = is_pool2d ? "NCHW" : "NCDHW";
  } else {
    return false;
  }

  if (rw.getWindowDilations().has_value() &&
      !(rw.getWindowDilations()->isSplat() &&
        rw.getWindowDilations()->getSplatValue<APInt>() == 1))
    return false;

  if (rw.getBaseDilations().has_value() &&
      !(rw.getBaseDilations()->isSplat() &&
        rw.getBaseDilations()->getSplatValue<APInt>() == 1))
    return false;

  if (llvm::all_of(padding, [](int64_t i) { return i == 0; })) {
    *padding_mode = "VALID";
    return true;
  }

  // Check that the individual padding values are corresponding to SAME
  // padding from TensorFlow.
  auto operand_type =
      mlir::dyn_cast<RankedTensorType>(rw.getInputs()[0].getType());
  RankedTensorType output_type =
      mlir::dyn_cast<RankedTensorType>(rw.getResult(0).getType());
  if (!operand_type || !output_type) return false;

  for (uint64_t i = 1; i < rank - 1; ++i) {
    int64_t padding_size =
        (output_type.getShape()[i] - 1) * (*window_strides)[i] +
        rw.getWindowDimensions().getValues<int64_t>()[i] -
        operand_type.getShape()[i];
    if (padding[2 * i] != tensorflow::MathUtil::FloorOfRatio(
                              padding_size, static_cast<int64_t>(2)) ||
        padding[2 * i + 1] != tensorflow::MathUtil::CeilOfRatio(
                                  padding_size, static_cast<int64_t>(2)))
      return false;
  }

  *padding_mode = "SAME";
  return true;
}

// Convert a reduce_window operation into a cumulative operation where possible
// for a given binary operation.
template <class BinaryOp, class TfCumOp>
class ConvertLoweredCumOp : public OpConversionPattern<mhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  virtual bool IsInitValue(const DenseElementsAttr& attr) const = 0;

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp rw, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (rw.getNumResults() != 1 || rw.getInputs().size() != 1 ||
        rw.getInitValues().size() != 1)
      return failure();

    if (failed(MatchBinaryReduceFunction<BinaryOp>(rw.getBody())))
      return failure();

    // Ensure that initial_values are as expected.
    auto const_op = llvm::dyn_cast_or_null<mhlo::ConstantOp>(
        rw.getInitValues()[0].getDefiningOp());
    if (!const_op) return failure();
    auto const_op_dense_value =
        mlir::cast<DenseElementsAttr>(const_op.getValue());
    if (!const_op_dense_value || !IsInitValue(const_op_dense_value)) {
      return failure();
    }

    auto operand_type = mlir::cast<ShapedType>(rw.getInputs()[0].getType());

    // For a cumulative op, require a tensor of 1s for each dimension in
    // operand.
    auto is_splat_int64_ones =
        [&rewriter,
         &operand_type](const std::optional<DenseIntElementsAttr>& attr) {
          // According to the definition, the default value of these attributes
          // are all ones when unspecified.
          if (!attr.has_value()) return true;
          if (attr->getType().getShape()[0] != operand_type.getRank())
            return false;
          if (!attr->isSplat()) return false;
          if (attr->getElementType() != rewriter.getIntegerType(64))
            return false;
          if (attr->getSplatValue<APInt>().getSExtValue() != 1) return false;
          return true;
        };
    if (!is_splat_int64_ones(rw.getBaseDilations()) ||
        !is_splat_int64_ones(rw.getWindowDilations()) ||
        !is_splat_int64_ones(rw.getWindowStrides()))
      return failure();

    // Determine which axis is being used for the cumulative operation.
    //
    // For a cumulative op, window_dimensions should be of the form:
    //  dense<[1, 1, N, 1]>
    // where N is the same as the size of the corresponding input dimension
    // and there is a 1-entry for each input dimension not being operated
    // over.
    const auto& window_dimensions = rw.getWindowDimensions();
    if (window_dimensions.size() != operand_type.getRank()) return failure();
    int64_t cumulative_axis = -1;
    for (int64_t i = 0, e = window_dimensions.size(); i < e; ++i) {
      int64_t window_dimension = window_dimensions.getValues<int64_t>()[i];
      if (window_dimension == 1) continue;
      // Cumulative axis already set.
      if (cumulative_axis != -1) return failure();
      // Potential cumulative axis is not the right size.
      if (window_dimension != operand_type.getShape()[i]) return failure();
      cumulative_axis = i;
    }

    if (cumulative_axis == -1) {
      rw.emitOpError() << "no reduced dimension is found.";
      return failure();
    }

    // For a cumulative op, padding (expressed as a list of left-padding and
    // right-padding pairs) should be of the form:
    //  dense<[[0, 0], [0, 0], [N-1, 0], [0, 0]]>
    // where N is the size of the input dimension being operated over.
    if (!rw.getPadding()) return failure();
    const auto& padding = rw.getPadding()->getValues<int64_t>();
    if (padding.size() != operand_type.getRank() * 2) return failure();
    int64_t padding_value = operand_type.getShape()[cumulative_axis] - 1;
    for (int64_t dim = 0; dim < operand_type.getRank(); ++dim) {
      int64_t left_padding = padding[2 * dim];
      int64_t right_padding = padding[2 * dim + 1];
      if (dim == cumulative_axis) {
        if (left_padding != padding_value) return failure();
      } else {
        if (left_padding != 0) return failure();
      }
      if (right_padding != 0) return failure();
    }

    auto axis = rewriter.create<TF::ConstOp>(
        rw->getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), cumulative_axis));

    rewriter.replaceOpWithNewOp<TfCumOp>(rw, rw.getType(0), rw.getInputs()[0],
                                         axis, /* exclusive */ false,
                                         /* reverse */ false);
    return success();
  }
};

class ConvertLoweredCumSumOp
    : public ConvertLoweredCumOp<mhlo::AddOp, TF::CumsumOp> {
  using ConvertLoweredCumOp::ConvertLoweredCumOp;
  bool IsInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (mlir::isa<FloatType>(element_type)) {
      auto value = *attr.value_begin<APFloat>();
      return value.isZero();
    }
    auto value = *attr.value_begin<APInt>();
    return value.isZero();
  }
};

class ConvertLoweredCumProdOp
    : public ConvertLoweredCumOp<mhlo::MulOp, TF::CumprodOp> {
  using ConvertLoweredCumOp::ConvertLoweredCumOp;
  bool IsInitValue(const DenseElementsAttr& attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (mlir::isa<FloatType>(element_type)) {
      auto value = *attr.value_begin<APFloat>();
      return value.isExactlyValue(1.0);
    }
    auto value = *attr.value_begin<APInt>();
    return value.getSExtValue() == 1;
  }
};

// Maps the following representations of AvgPool in MHLO into a tf.AvgPool{3D}
// operation when they cleanly map to 2D or 3D average pool with VALID or SAME
// padding:
// * div(reduce_sum_window(x), constant(sizeof(window)))
// * div(reduce_sum_window(x), reduce_sum_window(constant(1)))
class ConvertAvgPoolOp : public OpConversionPattern<mhlo::DivOp> {
 public:
  explicit ConvertAvgPoolOp(MLIRContext* context)
      : OpConversionPattern(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(
      mhlo::DivOp div_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto rw =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(div_op.getLhs().getDefiningOp());
    if (!rw || rw->getNumResults() != 1) return failure();

    // Check that the reduce-window is a sum-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw.getBody())))
      return failure();

    // Check that this is a floating point reduce window with a rank of 4 or 5.
    const RankedTensorType rw_type =
        mlir::dyn_cast<RankedTensorType>(rw.getResult(0).getType());
    if (!rw_type || !mlir::isa<FloatType>(rw_type.getElementType()) ||
        rw_type.getRank() <= 3 || rw_type.getRank() > 5)
      return failure();

    // Check that the Div op doesn't do broadcasting on the output of the reduce
    // window.
    if (div_op.getType() != rw_type) return failure();

    // If the init value isn't zero then it can't be an average pool.
    if (!isFloatZero(rw.getInitValues()[0])) return failure();

    llvm::SmallVector<int64_t, 5> window_strides;
    std::string padding_mode, data_format;
    if (!IsSpatialPoolingWithoutDilation(rw, &window_strides, &padding_mode,
                                         &data_format)) {
      return rewriter.notifyMatchFailure(
          div_op, "not the root of spatial pooling without dilation");
    }

    DenseFPElementsAttr divisor;
    auto div_rhs = recursivelyWalkUp<mhlo::BroadcastInDimOp>(div_op.getRhs());
    if (matchPattern(div_rhs, m_Constant(&divisor))) {
      // If the divisor is a constant then check that it matches with the number
      // of elements inside the window what is required for a VALID AvgPool.
      if (!divisor.isSplat()) return failure();
      int64_t window_size = 1;
      for (int64_t w : rw.getWindowDimensions().getValues<int64_t>()) {
        window_size *= w;
      }
      if (!divisor.getSplatValue<APFloat>().isExactlyValue(window_size))
        return failure();

      if (padding_mode != "VALID") {
        return failure();
      }

      return replaceWithAvgPool(
          div_op, rw.getInputs()[0],
          llvm::to_vector<4>(rw.getWindowDimensions().getValues<int64_t>()),
          window_strides, "VALID", data_format, rewriter);
    }

    Value actual_divisor =
        recursivelyWalkUp<mhlo::BroadcastInDimOp, mhlo::ReshapeOp>(
            div_op.getRhs());
    auto rw_rhs =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(actual_divisor.getDefiningOp());
    if (rw_rhs && rw_rhs.getNumResults() == 1) {
      // Check that RHS is a sum-reduce-window.
      if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw_rhs.getBody())))
        return failure();

      // Check that the RHS is a reduce_window over a constant 1 operand with 0
      // as the init value.
      DenseFPElementsAttr rhs_operand;
      auto rw_rhs_operand =
          recursivelyWalkUp<mhlo::BroadcastInDimOp>(rw_rhs.getInputs()[0]);
      if (!isFloatZero(rw_rhs.getInitValues()[0]) ||
          !matchPattern(rw_rhs_operand, m_Constant(&rhs_operand)) ||
          !rhs_operand.isSplat() ||
          !rhs_operand.getSplatValue<APFloat>().isExactlyValue(1.0))
        return failure();

      // Check that the two reduce window have the same window configuration.
      if (rw.getWindowDimensions() != rw_rhs.getWindowDimensions() ||
          rw.getWindowStrides() != rw_rhs.getWindowStrides() ||
          rw.getWindowDilations() != rw_rhs.getWindowDilations() ||
          rw.getBaseDilations() != rw_rhs.getBaseDilations() ||
          rw.getPadding() != rw_rhs.getPadding())
        return failure();

      return replaceWithAvgPool(
          div_op, rw.getInputs()[0],
          llvm::to_vector<4>(rw.getWindowDimensions().getValues<int64_t>()),
          window_strides, padding_mode, data_format, rewriter);
    }

    return failure();
  }

 private:
  bool isFloatZero(Value value) const {
    DenseFPElementsAttr initial_value;
    return matchPattern(value, m_Constant(&initial_value)) &&
           initial_value.getNumElements() == 1 &&
           initial_value.getValues<APFloat>()[0].isZero();
  }

  LogicalResult replaceWithAvgPool(mhlo::DivOp op, Value input,
                                   llvm::ArrayRef<int64_t> ksizes,
                                   llvm::ArrayRef<int64_t> kstrides,
                                   llvm::StringRef padding,
                                   llvm::StringRef data_format,
                                   ConversionPatternRewriter& rewriter) const {
    if (ksizes.size() == 4) {
      rewriter.replaceOpWithNewOp<TF::AvgPoolOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr(data_format));
      return success();
    } else if (ksizes.size() == 5) {
      rewriter.replaceOpWithNewOp<TF::AvgPool3DOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr(data_format));
      return success();
    }
    return failure();
  }

  // Walks up the op and ignore all precedding ops of type Tys.
  // Returns the first producer op whose type is not in Tys.
  template <typename... Tys>
  Value recursivelyWalkUp(Value op) const {
    while (llvm::isa_and_nonnull<Tys...>(op.getDefiningOp())) {
      Operation* producer = op.getDefiningOp();
      op = producer->getOperand(/*idx=*/0);
    }

    return op;
  }
};

class ConvertMaxPoolOp : public OpConversionPattern<mhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp rw, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Check that the reduce-window is a max-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::MaxOp>(rw.getBody())))
      return failure();

    // Check that this is a floating point reduce window with a rank of 4 or 5.
    const RankedTensorType rw_type =
        mlir::dyn_cast<RankedTensorType>(rw.getResult(0).getType());
    if (!rw_type || !mlir::isa<FloatType>(rw_type.getElementType()) ||
        rw_type.getRank() <= 3 || rw_type.getRank() > 5)
      return failure();

    if (!isFloatMinusInfinity(rw.getInitValues()[0])) {
      return failure();
    }

    llvm::SmallVector<int64_t, 5> window_strides;
    std::string padding_mode, data_format;
    if (!IsSpatialPoolingWithoutDilation(rw, &window_strides, &padding_mode,
                                         &data_format)) {
      return rewriter.notifyMatchFailure(
          rw, "not the root of spatial pooling without dilation");
    }

    return replaceWithMaxPool(
        rw, rw.getInputs()[0],
        llvm::to_vector<4>(rw.getWindowDimensions().getValues<int64_t>()),
        window_strides, padding_mode, data_format, rewriter);
  }

 private:
  bool isFloatMinusInfinity(Value value) const {
    DenseFPElementsAttr float_value;
    if (!matchPattern(value, m_Constant(&float_value))) {
      return false;
    }

    if (float_value.getNumElements() != 1) {
      return false;
    }

    APFloat element = float_value.getValues<APFloat>()[0];
    if (!element.isInfinity()) {
      return false;
    }
    if (!element.isNegative()) {
      return false;
    }

    return true;
  }

  LogicalResult replaceWithMaxPool(mhlo::ReduceWindowOp op, Value input,
                                   llvm::ArrayRef<int64_t> ksizes,
                                   llvm::ArrayRef<int64_t> kstrides,
                                   llvm::StringRef padding,
                                   llvm::StringRef data_format,
                                   ConversionPatternRewriter& rewriter) const {
    if (ksizes.size() == 4) {
      rewriter.replaceOpWithNewOp<TF::MaxPoolOp>(
          op, op.getType(0), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          rewriter.getStringAttr(data_format));
      return success();
    } else if (ksizes.size() == 5) {
      rewriter.replaceOpWithNewOp<TF::MaxPool3DOp>(
          op, op.getType(0), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr(data_format));
      return success();
    }
    return failure();
  }
};

// Returns the shape of the given value in a Constant Op.
arith::ConstantOp ShapeToConst(PatternRewriter& rewriter, Value value) {
  ArrayRef<int64_t> shape = mlir::cast<ShapedType>(value.getType()).getShape();
  auto attr_type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                         rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, shape);
  return rewriter.create<arith::ConstantOp>(value.getLoc(), attr_type, attr);
}

bool IsSign(APInt a, APInt sign) {
  if (a.isZero()) return a == sign;
  if (a.isNegative()) return sign == -1;
  return sign == 1;
}

bool IsSign(APFloat a, APFloat sign) {
  if (a.isNaN() || a.isZero()) return a == sign;
  if (a.isNegative()) return sign.isExactlyValue(-1.0);
  return sign.isExactlyValue(1.0);
}

bool IsDenseSplatIntAttr(ElementsAttr float_or_int) {
  return mlir::isa<SplatElementsAttr>(float_or_int) &&
         mlir::isa<DenseIntElementsAttr>(float_or_int);
}

bool IsDenseSplatFloatAttr(ElementsAttr float_or_int) {
  return mlir::isa<SplatElementsAttr>(float_or_int) &&
         mlir::isa<DenseFPElementsAttr>(float_or_int);
}

bool ValueIsReciprocal(ElementsAttr float_or_int, ElementsAttr rhs) {
  if (IsDenseSplatFloatAttr(float_or_int) &&
      IsDenseSplatFloatAttr(float_or_int)) {
    return (mlir::cast<SplatElementsAttr>(float_or_int)
                .getSplatValue<APFloat>() *
            mlir::cast<SplatElementsAttr>(rhs).getSplatValue<APFloat>())
        .isExactlyValue(1.0);
  } else if (IsDenseSplatIntAttr(float_or_int) &&
             IsDenseSplatIntAttr(float_or_int)) {
    return (mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>() *
            mlir::cast<SplatElementsAttr>(rhs).getSplatValue<APInt>()) == 1;
  }
  return false;
}

bool ValueEquals(ElementsAttr float_or_int, double rhs) {
  if (IsDenseSplatFloatAttr(float_or_int)) {
    return mlir::cast<SplatElementsAttr>(float_or_int)
        .getSplatValue<APFloat>()
        .isExactlyValue(rhs);
  } else if (IsDenseSplatIntAttr(float_or_int)) {
    return mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>() ==
           static_cast<int>(rhs);
  }
  return false;
}

bool ValueGreaterThanZero(ElementsAttr float_or_int) {
  if (IsDenseSplatIntAttr(float_or_int)) {
    auto value =
        mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APInt>();
    return !value.isNegative() && !value.isZero();
  } else if (IsDenseSplatFloatAttr(float_or_int)) {
    auto value =
        mlir::cast<SplatElementsAttr>(float_or_int).getSplatValue<APFloat>();
    return !value.isNaN() && !value.isNegative() && !value.isZero();
  }
  return false;
}

// Returns whether the splat constant is the sign of the int or float Tensor.
bool TensorIsSign(PatternRewriter& rewriter, ElementsAttr float_or_int,
                  ElementsAttr sgn_cst) {
  auto sgn_splat = llvm::dyn_cast<SplatElementsAttr>(sgn_cst);
  if (!sgn_splat) return false;

  auto splat = dyn_cast<SplatElementsAttr>(float_or_int);
  if (auto float_spl = llvm::dyn_cast_if_present<FloatAttr>(splat),
      sgn_cst_spl = llvm::dyn_cast_if_present<FloatAttr>(sgn_splat);
      float_spl && sgn_cst_spl) {
    return IsSign(float_spl.getValue(), sgn_cst_spl.getValue());
  }
  if (auto int_spl = llvm::dyn_cast_if_present<IntegerAttr>(splat),
      sgn_cst_spl = llvm::dyn_cast_if_present<IntegerAttr>(sgn_splat);
      int_spl && sgn_cst_spl) {
    return IsSign(int_spl.getValue(), sgn_cst_spl.getValue());
  }
  if (mlir::isa<DenseFPElementsAttr>(float_or_int)) {
    auto sgn_splat_value = sgn_splat.getSplatValue<APFloat>();
    return llvm::all_of(float_or_int.getValues<APFloat>(), [&](APFloat value) {
      return IsSign(value, sgn_splat_value);
    });
  }
  if (mlir::isa<DenseIntElementsAttr>(float_or_int)) {
    auto sgn_splat_value = sgn_splat.getSplatValue<APInt>();
    return llvm::all_of(float_or_int.getValues<APInt>(), [&](APInt value) {
      return IsSign(value, sgn_splat_value);
    });
  }
  return false;
}

bool SameTypeOrDefaultCompare(mhlo::ComparisonTypeAttr comparison_type_attr,
                              ElementsAttr cst) {
  if (!comparison_type_attr) return true;
  auto comparison_type_attr_value = comparison_type_attr.getValue();
  if (comparison_type_attr_value == mhlo::ComparisonType::FLOAT &&
      IsDenseSplatFloatAttr(cst)) {
    return true;
  }
  if ((comparison_type_attr_value == mhlo::ComparisonType::SIGNED ||
       comparison_type_attr_value == mhlo::ComparisonType::UNSIGNED) &&
      IsDenseSplatIntAttr(cst)) {
    return true;
  }
  return false;
}

class ConvertGatherOp : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  // Helper params for representing the transpose params for the "canonicalized"
  // output to the real output.
  struct TransposeParams {
    std::vector<int64_t> permutation;
    // The following are the "canonicalized" output shape with offset dims.
    std::vector<int64_t> canonicalized_output_shape;
    std::vector<int64_t> canonicalized_offset_dims;
  };

  LogicalResult matchAndRewrite(
      mhlo::GatherOp gather_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (succeeded(ConvertGatherOpToSlice(gather_op, rewriter))) {
      return success();
    }

    Value operand = gather_op.getOperand();
    Value start_indices = gather_op.getStartIndices();

    // Can only convert with static shaped gather.
    ShapedType operand_type = mlir::cast<ShapedType>(operand.getType());
    ShapedType start_indices_type =
        mlir::cast<ShapedType>(start_indices.getType());
    ShapedType result_type =
        mlir::cast<ShapedType>(gather_op.getResult().getType());
    if (!operand_type.hasStaticShape()) {
      gather_op.emitOpError() << "Dynamic shaped operand is not supported.";
      return failure();
    }

    // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
    int64_t index_vector_dim =
        gather_op.getDimensionNumbers().getIndexVectorDim();
    if (failed(NormalizeIndexVector(gather_op, start_indices,
                                    start_indices_type, index_vector_dim,
                                    rewriter))) {
      return failure();
    }

    // Verify that start_index_map and collapsed_slice_dims contains the same
    // values.
    auto start_index_map = gather_op.getDimensionNumbers().getStartIndexMap();
    auto collapsed_slice_dims =
        gather_op.getDimensionNumbers().getCollapsedSliceDims();
    if (start_index_map.size() != collapsed_slice_dims.size()) {
      return rewriter.notifyMatchFailure(
          gather_op,
          "different size for start index map and collapsed slice dims");
    }
    for (auto c : collapsed_slice_dims) {
      if (llvm::count(start_index_map, c) == 0) {
        return rewriter.notifyMatchFailure(
            gather_op, "collapsed slice dim isn't present in start index map");
      }
    }

    // Verify that slice_sizes is 1 for the indexed dimensions and the full
    // shape for the rest of the dimensions.
    auto slice_sizes = gather_op.getSliceSizes();
    int64_t index = 0;
    for (int64_t s : slice_sizes.getValues<int64_t>()) {
      if (llvm::count(start_index_map, index)) {
        if (s != 1) {
          return rewriter.notifyMatchFailure(gather_op,
                                             "unsupported slice sizes");
        }
      } else {
        if (s != operand_type.getShape()[index]) {
          return rewriter.notifyMatchFailure(gather_op,
                                             "unsupported slice sizes");
        }
      }
      ++index;
    }

    // Verify that offset_dims are the tailing dimensions in the output tensor.
    auto offset_dims = gather_op.getDimensionNumbers().getOffsetDims();
    SmallVector<int64_t, 4> offset_dims_vector(offset_dims.begin(),
                                               offset_dims.end());
    const TransposeParams& transpose_params =
        CanonicalizeOffset(/*result_type=*/result_type,
                           /*original_offset_dims=*/offset_dims_vector);

    int64_t offset = start_indices_type.getRank() - 1;
    for (int64_t o : transpose_params.canonicalized_offset_dims) {
      if (o != offset) {
        return rewriter.notifyMatchFailure(gather_op,
                                           "unsupported offset dims");
      }
      ++offset;
    }

    // Transpose the operand to handle non-iota start index map.
    llvm::SmallVector<int64_t, 4> transpose_dimensions;
    llvm::SmallVector<int64_t, 4> transpose_shape;
    for (auto s : start_index_map) {
      transpose_dimensions.push_back(s);
      transpose_shape.push_back(operand_type.getShape()[s]);
    }
    for (int64_t i = 0, e = operand_type.getRank(); i < e; ++i) {
      if (llvm::count(start_index_map, i) == 0) {
        transpose_dimensions.push_back(i);
        transpose_shape.push_back(operand_type.getShape()[i]);
      }
    }
    operand_type =
        RankedTensorType::get(transpose_shape, operand_type.getElementType());
    operand = rewriter.create<mhlo::TransposeOp>(
        gather_op.getLoc(), operand_type, operand,
        rewriter.getI64TensorAttr(transpose_dimensions));

    // Check whether we need to append a transpose op after the gather nd.
    bool need_transpose_after = false;
    for (int i = 0; i < transpose_params.permutation.size(); ++i) {
      if (i != transpose_params.permutation[i]) {
        need_transpose_after = true;
        break;
      }
    }

    auto tf_gather_nd_result_type =
        RankedTensorType::get(transpose_params.canonicalized_output_shape,
                              result_type.getElementType());

    TF::CastOp cast_op = nullptr;
    if (start_indices_type.getElementType().isUnsignedInteger(32)) {
      cast_op = rewriter.create<TF::CastOp>(
          gather_op->getLoc(),
          RankedTensorType::get(start_indices_type.getShape(),
                                rewriter.getI64Type()),
          start_indices);
    }

    auto tf_gather_nd_op = rewriter.create<TF::GatherNdOp>(
        gather_op->getLoc(), tf_gather_nd_result_type, operand,
        cast_op ? cast_op.getResult() : start_indices);

    if (!need_transpose_after) {
      rewriter.replaceOp(gather_op, tf_gather_nd_op->getOpResults());
      return success();
    }

    // Insert the transpose op after the gather_nd.
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
        gather_op, result_type, tf_gather_nd_op,
        rewriter.getI64TensorAttr(transpose_params.permutation));

    return success();
  }

  // Convert gather op to tf.slice and tf.concat
  LogicalResult ConvertGatherOpToSlice(
      mhlo::GatherOp gather_op, ConversionPatternRewriter& rewriter) const {
    Value operand = gather_op.getOperand();
    Value start_indices = gather_op.getStartIndices();
    static const int rank_two = 2;
    // This converts a gather op to multiple slice ops, cap the number of slice
    // ops allowed.
    static const int max_batch_size = 50;

    // Can only convert with static shaped gather.
    ShapedType operand_type = mlir::cast<ShapedType>(operand.getType());
    ShapedType start_indices_type =
        mlir::cast<ShapedType>(start_indices.getType());
    ShapedType result_type =
        mlir::cast<ShapedType>(gather_op.getResult().getType());
    if (!operand_type.hasStaticShape() ||
        !start_indices_type.hasStaticShape() || !result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          gather_op,
          "Dynamic shaped inputs are not supported when legalizing mhlo.gather "
          "op to tf.slice.");
    }

    auto start_index_map = gather_op.getDimensionNumbers().getStartIndexMap();
    auto collapsed_slice_dims =
        gather_op.getDimensionNumbers().getCollapsedSliceDims();
    auto offset_dims = gather_op.getDimensionNumbers().getOffsetDims();
    auto slice_sizes = gather_op.getSliceSizes();
    llvm::SmallVector<int64_t, 2> slice_sizes_vector;
    slice_sizes_vector.reserve(slice_sizes.size());
    for (int64_t s : slice_sizes.getValues<int64_t>()) {
      slice_sizes_vector.push_back(s);
    }

    llvm::SmallVector<int64_t, 1> batch_dims;
    // Offset dims are guaranteed to be sorted.
    int offset_index = 0;
    for (int64_t i = 0; i < result_type.getRank(); ++i) {
      if (offset_index >= offset_dims.size() ||
          offset_dims[offset_index] != i) {
        batch_dims.push_back(i);
      } else {
        ++offset_index;
      }
    }
    // Here we only support gather with one batch dim and the batch dim is 0.
    if (batch_dims.size() != 1 || batch_dims[0] != 0) {
      return failure();
    }
    int64_t batch_dim = batch_dims[0];
    // Batch dim in operand and start indices should match.
    if (operand_type.getDimSize(batch_dim) > max_batch_size ||
        operand_type.getRank() != rank_two ||
        start_indices_type.getRank() != rank_two ||
        operand_type.getDimSize(batch_dim) !=
            start_indices_type.getDimSize(batch_dim) ||
        slice_sizes_vector[batch_dim] != 1) {
      return failure();
    }
    // Here we only support the case where [0, 1] in start_indices maps to
    // operand[0, 1]
    for (int64_t i = 0; i < start_index_map.size(); i++) {
      if (start_index_map[i] != i) {
        return failure();
      }
    }
    // Collapsed slice dims should contain the batch dim.
    if (collapsed_slice_dims.size() != start_index_map.size() - 1 ||
        collapsed_slice_dims.size() != 1 || collapsed_slice_dims[0] != 0) {
      return failure();
    }

    // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
    int64_t index_vector_dim =
        gather_op.getDimensionNumbers().getIndexVectorDim();
    if (failed(NormalizeIndexVector(gather_op, start_indices,
                                    start_indices_type, index_vector_dim,
                                    rewriter))) {
      return failure();
    }

    ImplicitLocOpBuilder builder(gather_op.getLoc(), rewriter);
    // Clamp the start indices to ensure it is in bounds.
    auto max_start_indices = BuildIntArrayConstOp(
        builder, rewriter,
        llvm::SmallVector<int64_t>(
            {operand_type.getDimSize(0) - slice_sizes_vector[0],
             operand_type.getDimSize(1) - slice_sizes_vector[1]}),
        start_indices_type.getElementType());
    auto min_start_indices = BuildIntArrayConstOp(
        builder, rewriter, llvm::SmallVector<int64_t>({0, 0}),
        start_indices_type.getElementType());
    auto start_indices_max_op = rewriter.create<TF::MaximumOp>(
        gather_op.getLoc(), start_indices, min_start_indices);
    auto clamped_start_indices_op = rewriter.create<TF::MinimumOp>(
        gather_op.getLoc(), start_indices_max_op, max_start_indices);

    int64_t batch_size = start_indices_type.getDimSize(batch_dim);
    auto slice_size = BuildIntArrayConstOp(
        builder, rewriter, slice_sizes_vector, rewriter.getI32Type());
    if (batch_size == 1) {
      auto squeeze_op = rewriter.create<TF::SqueezeOp>(
          gather_op.getLoc(),
          RankedTensorType::get({rank_two},
                                start_indices_type.getElementType()),
          clamped_start_indices_op,
          rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
      auto slice_op =
          rewriter.create<TF::SliceOp>(gather_op.getLoc(), gather_op.getType(),
                                       operand, squeeze_op, slice_size);
      rewriter.replaceOp(gather_op, slice_op);
      return mlir::success();
    }

    llvm::SmallVector<Value, 1> slices;
    slices.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      auto zero = BuildIntArrayConstOp(builder, rewriter,
                                       llvm::SmallVector<int64_t>({i, 0}),
                                       rewriter.getI32Type());
      auto two = BuildIntArrayConstOp(builder, rewriter,
                                      llvm::SmallVector<int64_t>({1, 2}),
                                      rewriter.getI32Type());
      auto begin = rewriter.create<TF::SliceOp>(
          gather_op.getLoc(),
          RankedTensorType::get({1, 2}, start_indices_type.getElementType()),
          clamped_start_indices_op, zero, two);
      auto squeeze_op = rewriter.create<TF::SqueezeOp>(
          gather_op.getLoc(),
          RankedTensorType::get({rank_two},
                                start_indices_type.getElementType()),
          begin,
          rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
      auto slice_op = rewriter.create<TF::SliceOp>(
          gather_op.getLoc(),
          RankedTensorType::get({1, slice_sizes_vector[1]},
                                operand_type.getElementType()),
          operand, squeeze_op, slice_size);
      slices.push_back(slice_op);
    }
    auto scalar_type = RankedTensorType::get({}, rewriter.getI32Type());
    auto zero_scalar = rewriter.create<TF::ConstOp>(
        gather_op.getLoc(),
        DenseIntElementsAttr::get(scalar_type, static_cast<int32_t>(0)));
    auto concat_op = rewriter.create<TF::ConcatV2Op>(
        gather_op.getLoc(), result_type, slices, zero_scalar);
    rewriter.replaceOp(gather_op, concat_op);
    return mlir::success();
  }

 private:
  // Canonicalize the offset dims to make sure the offset dims are the trailing
  // dimensions of the output tensor.
  // We will also return the permutation for (the transpose op).
  // However, it's not guaranteed the canonicalized offset dims can make it
  // always legalizable to tf.
  TransposeParams CanonicalizeOffset(
      ShapedType result_type, ArrayRef<int64_t> original_offset_dims) const {
    TransposeParams transpose_params;
    int output_rank = result_type.getRank();
    // The canonicalized offset should be the trailing of the output rank.
    for (int start = output_rank - original_offset_dims.size();
         start < output_rank; ++start) {
      transpose_params.canonicalized_offset_dims.push_back(start);
    }

    // For those dims NOT inside the original_offset_dims are considered "batch
    // dims".
    std::vector<int64_t> batch_dims;
    // Offset dims are guaranteed to be sorted.
    int offset_index = 0;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (offset_index >= original_offset_dims.size() ||
          original_offset_dims[offset_index] != i) {
        batch_dims.push_back(i);
      } else {
        ++offset_index;
      }
    }

    // Populate the trnaspose permutation params from a "canonicalized" output
    // to the real output.
    // The canonicalized layout would be batch_dims followed by sliced_dims.
    // The current layout is essentially a transpose after the canonicalized
    // layout.
    // Take the following as an example:
    // If we have the:
    // original_offset_dims like [1, 2, 4]
    // batch_dims like [0, 3]
    // It's like performing transpose on a "canonicalized"
    // [batch_dims, sliced_dims]: [B1, B2, O1, O2, O3]
    // into the current layout: [B1, O1, O2, B2, O3]
    // where the permutation is [0, 2, 3, 1, 4]
    int batch_idx = 0;
    int offset_idx = 0;
    int batch_dim_size = batch_dims.size();
    for (int i = 0; i < output_rank; ++i) {
      if (batch_idx >= batch_dims.size()) {
        transpose_params.permutation.push_back(batch_dim_size + offset_idx);
        ++offset_idx;
      } else if (offset_idx < original_offset_dims.size() &&
                 original_offset_dims[offset_idx] < batch_dims[batch_idx]) {
        transpose_params.permutation.push_back(batch_dim_size + offset_idx);
        ++offset_idx;
      } else {
        transpose_params.permutation.push_back(batch_idx++);
      }
    }

    // Finally, let's find out what are the "canonicalized" output shape looks
    // like.
    for (auto dim : batch_dims) {
      transpose_params.canonicalized_output_shape.push_back(
          result_type.getDimSize(dim));
    }
    for (auto dim : original_offset_dims) {
      transpose_params.canonicalized_output_shape.push_back(
          result_type.getDimSize(dim));
    }
    return transpose_params;
  }
};

class ConvertWhileOp : public OpConversionPattern<mhlo::WhileOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::WhileOp while_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // HLO WhileOp should have two regions: cond and body.
    if (while_op->getNumRegions() != 2) return failure();

    // This rule doesn't support mhlo::WhileOp with tuple inputs.
    for (auto type : while_op->getOperandTypes()) {
      if (mlir::isa<TupleType>(type)) return failure();
    }

    // Creates a TF::WhileRegionOp to replace the mhlo::WhileOp. HLO WhileOp
    // currently doesn't support stateless and shape invariant, so these
    // parameters are set to the default values.
    auto new_while = rewriter.create<TF::WhileRegionOp>(
        while_op.getLoc(), while_op->getResultTypes(), while_op->getOperands(),
        /*parallel_iterations=*/10,
        /*is_stateless=*/false, /*shape_invariant=*/false);
    new_while.getCond().takeBody(while_op.getCond());
    new_while.getBody().takeBody(while_op.getBody());
    ReplaceReturnOp(new_while.getCond(), rewriter);
    ReplaceReturnOp(new_while.getBody(), rewriter);
    rewriter.replaceOp(while_op, new_while.getResults());
    return success();
  }
};

class ConvertIfOp : public OpConversionPattern<mhlo::IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // HLO IfOp currently doesn't support stateless
    auto new_op = rewriter.create<TF::IfRegionOp>(
        op.getLoc(), op->getResultTypes(), op.getPred(),
        /*is_stateless=*/false, /*_then_func_name=*/nullptr,
        /*_else_func_name=*/nullptr);
    new_op.getThenBranch().takeBody(op.getTrueBranch());
    new_op.getElseBranch().takeBody(op.getFalseBranch());
    ReplaceReturnOp(new_op.getThenBranch(), rewriter);
    ReplaceReturnOp(new_op.getElseBranch(), rewriter);
    rewriter.replaceOp(op, new_op.getResults());
    return success();
  }
};

// Converts mhlo.pad to tf.PadV2
Value ConvertPadOp(PatternRewriter& rewriter, Operation* old_op) {
  auto pad_op = cast<mhlo::PadOp>(old_op);
  mlir::Location loc = pad_op.getLoc();

  // Calculates non-negative padding amount and slice begins/sizes.
  llvm::SmallVector<int64_t, 8> padding;
  llvm::SmallVector<int64_t, 8> pad_output_shape;

  bool has_negative_padding_amount = false;
  llvm::SmallVector<int64_t, 8> slice_begins;
  llvm::SmallVector<int64_t, 8> slice_sizes;

  for (auto p : llvm::zip(pad_op.getOperand().getType().getShape(),
                          pad_op.getEdgePaddingLow().getValues<APInt>(),
                          pad_op.getEdgePaddingHigh().getValues<APInt>())) {
    const int64_t input_dim_size = std::get<0>(p);
    int64_t pad_output_dim_size = input_dim_size;

    const int64_t pad_low = std::get<1>(p).getSExtValue();
    if (pad_low < 0) {
      has_negative_padding_amount = true;
      padding.push_back(0);
    } else {
      padding.push_back(pad_low);
      pad_output_dim_size += pad_low;
    }

    const int64_t pad_high = std::get<2>(p).getSExtValue();
    if (pad_high < 0) {
      has_negative_padding_amount = true;
      padding.push_back(0);
    } else {
      padding.push_back(pad_high);
      pad_output_dim_size += pad_high;
    }

    pad_output_shape.push_back(pad_output_dim_size);

    slice_begins.push_back(pad_low < 0 ? -pad_low : 0);
    slice_sizes.push_back(input_dim_size + pad_low + pad_high);
  }

  // Convert to PadV2.
  auto padding_attr_type = RankedTensorType::get(
      {pad_op.getEdgePaddingLow().size(), 2}, rewriter.getI64Type());
  auto padding_attr = DenseIntElementsAttr::get(padding_attr_type, padding);
  auto padding_amount_const_op =
      rewriter.create<arith::ConstantOp>(loc, padding_attr_type, padding_attr);
  auto new_pad_op = rewriter.create<TF::PadV2Op>(
      loc, pad_op.getType().clone(pad_output_shape), pad_op.getOperand(),
      padding_amount_const_op, pad_op.getPaddingValue());
  if (!has_negative_padding_amount) {
    return new_pad_op;
  }

  // Convert negative padding amount into slice.
  auto slice_attr_type = RankedTensorType::get(
      {pad_op.getEdgePaddingLow().size()}, rewriter.getI64Type());
  auto slice_begins_const_op = rewriter.create<arith::ConstantOp>(
      loc, slice_attr_type,
      DenseIntElementsAttr::get(slice_attr_type, slice_begins));
  auto slice_sizes_const_op = rewriter.create<arith::ConstantOp>(
      loc, slice_attr_type,
      DenseIntElementsAttr::get(slice_attr_type, slice_sizes));
  return rewriter.create<TF::SliceOp>(loc, pad_op.getType(), new_pad_op,
                                      slice_begins_const_op,
                                      slice_sizes_const_op);
}

class ConvertPopulationCountOp
    : public OpConversionPattern<mhlo::PopulationCountOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::PopulationCountOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto output_type = op.getType().clone(
        rewriter.getIntegerType(/*width=*/8, /*isSigned=*/false));
    auto pop_cnt = rewriter.create<TF::PopulationCountOp>(
        op.getLoc(), output_type, op.getOperand());
    auto cast_or_pop_cnt =
        rewriter.createOrFold<TF::CastOp>(op.getLoc(), op.getType(), pop_cnt);
    rewriter.replaceOp(op, {cast_or_pop_cnt});
    return success();
  }
};

class ConvertCustomCallWithApproxTopK
    : public mlir::OpConversionPattern<mhlo::CustomCallOp> {
 public:
  explicit ConvertCustomCallWithApproxTopK(MLIRContext* context,
                                           mlir::ModuleOp* module_op)
      : OpConversionPattern<mhlo::CustomCallOp>(context, /*benefit=*/0),
        module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mhlo::CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getCallTargetName() != "ApproxTopK") {
      return mlir::failure();
    }
    auto is_supported_attr_name = [](NamedAttribute attr) {
      auto name = attr.getName();
      return name == "call_target_name" || name == "backend_config" ||
             name == "api_version" || name == "called_computations";
    };
    for (const auto& attr : op->getAttrs()) {
      if (!is_supported_attr_name(attr)) {
        return op.emitOpError()
               << attr.getName().getValue()
               << " is not a supported attribute for ApproxTopK";
      }
    }
    auto backend_config =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(op.getBackendConfigAttr());
    if (!backend_config) {
      return op.emitOpError() << "Missing backend_config attribute";
    }

    for (const auto& attr : backend_config) {
      auto name = attr.getName();
      if (!(name == "top_k" || name == "reduction_dim" ||
            name == "recall_target" || name == "aggregate_to_topk" ||
            name == "reduction_input_size_override" || name == "is_fallback")) {
        return op.emitOpError()
               << name.getValue() << " is not a supported backend_config"
               << " attribute for ApproxTopK";
      }
    }

    auto check_i64_attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name)) {
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      }
      auto attr = backend_config.getAs<IntegerAttr>(attr_name);
      if (!attr || !attr.getType().isInteger(64)) {
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of i64 type";
      }
      return success();
    };
    auto check_f32_attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name)) {
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      }
      auto attr = backend_config.getAs<FloatAttr>(attr_name);
      if (!attr || !attr.getType().isF32()) {
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of f32 type";
      }
      return success();
    };
    auto check_bool_attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name)) {
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      }
      auto attr = backend_config.getAs<BoolAttr>(attr_name);
      if (!attr) {
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of bool type";
      }
      return success();
    };
    if (failed(check_i64_attr("top_k"))) return failure();
    if (failed(check_i64_attr("reduction_dim"))) return failure();
    if (failed(check_f32_attr("recall_target"))) return failure();
    if (failed(check_bool_attr("aggregate_to_topk"))) return failure();
    if (failed(check_i64_attr("reduction_input_size_override"))) {
      return failure();
    }
    bool has_is_fallback = backend_config.contains("is_fallback");
    if (has_is_fallback && !backend_config.getAs<BoolAttr>("is_fallback")) {
      return op.emitOpError()
             << "is_fallback attribute in backend_config must be of bool type";
    }

    auto top_k_attr = backend_config.getAs<IntegerAttr>("top_k");
    auto reduction_dim_attr =
        backend_config.getAs<IntegerAttr>("reduction_dim");
    auto recall_target_attr = backend_config.getAs<FloatAttr>("recall_target");
    auto aggregate_to_topk_attr =
        backend_config.getAs<BoolAttr>("aggregate_to_topk");
    auto reduction_input_size_override_attr =
        backend_config.getAs<IntegerAttr>("reduction_input_size_override");
    if (op.getInputs().size() % 2 != 0) {
      return op.emitOpError() << "ApproxTopK takes an even number of operands.";
    }

    auto called_computations = op.getCalledComputations();
    if (called_computations.size() != 1) {
      return op.emitOpError()
             << "ApproxTopK takes exactly 1 called_computation.";
    }
    mlir::func::FuncOp callee = module_op_->lookupSymbol<mlir::func::FuncOp>(
        mlir::cast<FlatSymbolRefAttr>(op.getCalledComputations()[0]));
    mlir::FunctionType callee_type = callee.getFunctionType();
    SmallVector<Type, 4> expected_callee_input_types;
    auto num_inputs = op.getInputs().size() / 2;
    for (unsigned i = 0; i < num_inputs; ++i) {
      auto input_type =
          mlir::dyn_cast<RankedTensorType>(op.getOperand(i).getType());
      auto scalar = RankedTensorType::get({}, input_type.getElementType());
      expected_callee_input_types.push_back(scalar);
      expected_callee_input_types.push_back(scalar);
    }
    FunctionType expected_callee_type = mlir::FunctionType::get(
        op->getContext(), expected_callee_input_types,
        RankedTensorType::get({}, IntegerType::get(op->getContext(), 1)));
    if (callee_type != expected_callee_type) {
      return op.emitOpError()
             << "called_computation type does not match the expected type. Got "
             << callee_type << " expected " << expected_callee_type;
    }
    if (!MatchTopKComparator<mlir::func::ReturnOp>(callee.getBody())) {
      return op.emitOpError() << "only match for GT comparator";
    }
    auto is_max_k = rewriter.getBoolAttr(true);

    auto approx_top_k = rewriter.create<TF::ApproxTopKOp>(
        op.getLoc(), op->getResultTypes(), op.getInputs()[0], top_k_attr,
        reduction_dim_attr, recall_target_attr, is_max_k,
        reduction_input_size_override_attr, aggregate_to_topk_attr);

    rewriter.replaceOp(op, approx_top_k.getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp* module_op_;
};

// Removes the `mhlo.custom_call @shape_assertion` custom call which represents
// an assertion that the first operand (`assert_what`) evaluates to `true`.
// This is a temporary workaround for unblocking dynamic model conversion
// because starting from version 7, in presence of shape polymorphism JAX will
// emit stablehlo.custom_call @shape_assertion to verify at compile time that
// the code is used with compatible actual shapes.
// TFLite runtime kernels support shape checking and shape inference to some
// extent, it is okay to remove the shape assertion in most scenarios. However
// this is not always the case, JAX may trace the program differently based on
// the shape polymorphism specification, for example, if the program contains
// a conditional on "x.shape[0] % 2 == 0" that conditional would evaluate to
// True with x specified as (2*b, ...) and False otherwise. We can revisit
// this when need arises. See b/295316438 for details.
class RemoveCustomCallWithShapeAssertion
    : public OpConversionPattern<mhlo::CustomCallOp> {
 public:
  explicit RemoveCustomCallWithShapeAssertion(MLIRContext* context)
      : OpConversionPattern<mhlo::CustomCallOp>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getCallTargetName() != "shape_assertion") {
      return mlir::failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Converts a MHLO::GetDimensionSizeOp to TF ops.
class ConvertGetDimensionSizeOp
    : public OpConversionPattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GetDimensionSizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    Value shape_op = rewriter.create<TF::ShapeOp>(op.getLoc(), op.getOperand(),
                                                  rewriter.getBoolAttr(true));
    Value size =
        BuildIntArrayConstOp(builder, rewriter, llvm::SmallVector<int64_t>({1}),
                             rewriter.getI32Type());
    Value begin = BuildIntArrayConstOp(
        builder, rewriter,
        llvm::SmallVector<int64_t>({static_cast<int64_t>(op.getDimension())}),
        rewriter.getI64Type());
    Value slice_op = rewriter.create<TF::SliceOp>(
        op.getLoc(),
        RankedTensorType::get({static_cast<int64_t>(1)},
                              op.getType().getElementType()),
        shape_op, begin, size);
    Value squeeze_op = rewriter.create<TF::SqueezeOp>(
        op.getLoc(), op.getType(), slice_op,
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({0})));
    rewriter.replaceOp(op, {squeeze_op});
    return success();
  }
};

class ConvertRealDynamicSliceOp
    : public OpConversionPattern<mhlo::RealDynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::RealDynamicSliceOp real_dynamic_slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto start_indices_type = mlir::cast<RankedTensorType>(
        real_dynamic_slice_op.getStartIndices().getType());
    auto end_indices_type = mlir::cast<RankedTensorType>(
        real_dynamic_slice_op.getLimitIndices().getType());

    if (start_indices_type.getNumDynamicDims() != 0 ||
        end_indices_type.getNumDynamicDims() != 0) {
      return rewriter.notifyMatchFailure(
          real_dynamic_slice_op,
          "Start indices and limit indices must not have dynamic dimensions.");
    }
    rewriter.replaceOpWithNewOp<TF::StridedSliceOp>(
        real_dynamic_slice_op, real_dynamic_slice_op.getType(),
        real_dynamic_slice_op.getOperand(),
        real_dynamic_slice_op.getStartIndices(),
        real_dynamic_slice_op.getLimitIndices(),
        real_dynamic_slice_op.getStrides());
    return success();
  };
};

class ConvertDynamicIotaOp : public OpConversionPattern<mhlo::DynamicIotaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicIotaOp dynamic_iota_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    RankedTensorType type =
        mlir::dyn_cast_or_null<RankedTensorType>(dynamic_iota_op.getType());
    if (!type || type.getElementType().isUnsignedInteger(64)) {
      return rewriter.notifyMatchFailure(dynamic_iota_op,
                                         "TF::RangeOp doesn't support UI64");
    }
    // Only support 1D for now.
    if (type.getRank() > 1 || dynamic_iota_op.getIotaDimension() != 0) {
      return rewriter.notifyMatchFailure(
          dynamic_iota_op, [&](::mlir::Diagnostic& diag) {
            diag << "Only 1D DynamicIotaOp with iota dimension 0 is supported";
          });
    }

    const uint64_t dimension = dynamic_iota_op.getIotaDimension();
    Type element_type = type.getElementType();
    Attribute start, delta;
    if (mlir::isa<FloatType>(element_type)) {
      start = rewriter.getFloatAttr(element_type, 0.0);
      delta = rewriter.getFloatAttr(element_type, 1.0);
    } else if (mlir::isa<IntegerType>(element_type)) {
      start = rewriter.getIntegerAttr(element_type, 0);
      delta = rewriter.getIntegerAttr(element_type, 1);
    } else {
      return failure();
    }
    auto output_shape = dynamic_iota_op.getOperand();
    if (mlir::isa<FloatType>(element_type)) {
      auto cast_type =
          mlir::cast<ShapedType>(output_shape.getType()).clone(element_type);
      output_shape = rewriter.create<TF::CastOp>(dynamic_iota_op.getLoc(),
                                                 cast_type, output_shape);
    }
    DenseIntElementsAttr scalar_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({0}, rewriter.getI32Type()),
        llvm::ArrayRef<int32_t>({}));
    auto scalar_shape =
        rewriter.create<TF::ConstOp>(dynamic_iota_op.getLoc(), scalar_attr);
    auto limit_scalar = rewriter.create<TF::ReshapeOp>(
        dynamic_iota_op.getLoc(), RankedTensorType::get({}, element_type),
        output_shape, scalar_shape);
    auto range_type =
        RankedTensorType::get({type.getShape()[dimension]}, element_type);
    Value start_op =
        rewriter.create<TF::ConstOp>(dynamic_iota_op.getLoc(), start);
    Value delta_op =
        rewriter.create<TF::ConstOp>(dynamic_iota_op.getLoc(), delta);
    Value range_op = rewriter.create<TF::RangeOp>(
        dynamic_iota_op.getLoc(), range_type, start_op, limit_scalar, delta_op);
    rewriter.replaceOp(dynamic_iota_op, range_op);
    return success();
  }
};
// Returns true if broadcast_dimensions obey Tensorflow convention, as in new
// dimensions are added as prefix.
bool IsTFStyleBroadcast(DenseIntElementsAttr broadcast_dimensions,
                        Value output) {
  // broadcast_dimensions is an increasing list by definition, thus it suffices
  // to check the first element.
  int64_t input_rank = broadcast_dimensions.getNumElements();
  int64_t output_rank = mlir::cast<ShapedType>(output.getType()).getRank();
  return input_rank == 0 ||
         (broadcast_dimensions.getValues<APInt>()[0].getSExtValue() ==
          output_rank - input_rank);
}

// Returns true if the operation producing the provided result (`op_result`)
// is within an op region of an operation of type `ParentType`.
template <typename ParentType>
bool IsWithinOpRegion(mlir::OpResult op_result) {
  mlir::Operation* parent_op = op_result.getDefiningOp()->getParentOp();

  if (llvm::dyn_cast<ParentType>(parent_op)) {
    return true;
  }
  return false;
}

// Returns the intermediate shape that input tensor should be reshaped to during
// legalization of BroadcastInDimOp.
arith::ConstantOp ExpandedShape(PatternRewriter& rewriter, Value input,
                                DenseIntElementsAttr broadcast_dimensions,
                                Value output) {
  // Initialize expanded shape with output rank and dimensions of 1.
  SmallVector<Attribute, 4> expanded_shape(
      mlir::cast<ShapedType>(output.getType()).getRank(),
      /*Value=*/rewriter.getI64IntegerAttr(1));

  // Set dimension sizes specified by broadcast_dimensions.
  ArrayRef<int64_t> input_shape =
      mlir::cast<ShapedType>(input.getType()).getShape();
  for (auto x : llvm::enumerate(broadcast_dimensions)) {
    expanded_shape[x.value().getSExtValue()] =
        rewriter.getI64IntegerAttr(input_shape[x.index()]);
  }

  // Create the expanded type wrapped in a arith::ConstantOp.
  auto attr_type =
      RankedTensorType::get({static_cast<int64_t>(expanded_shape.size())},
                            rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, expanded_shape);
  return rewriter.create<arith::ConstantOp>(output.getLoc(), attr_type, attr);
}

Value ExpandedDynamicShape(PatternRewriter& rewriter, Value input,
                           DenseIntElementsAttr broadcast_dimensions,
                           Value output) {
  assert(mlir::cast<ShapedType>(output.getType()) &&
         "output type must be of ShapedType");
  int64_t output_rank = mlir::cast<ShapedType>(output.getType()).getRank();
  llvm::SmallVector<int64_t, 4> expanded_dimensions;
  llvm::SmallSet<int64_t, 4> broadcast_dimensions_values;
  for (auto x : llvm::enumerate(broadcast_dimensions)) {
    broadcast_dimensions_values.insert(x.value().getSExtValue());
  }
  for (int64_t i = 0; i < output_rank; i++) {
    if (!broadcast_dimensions_values.contains(i)) {
      expanded_dimensions.push_back(i);
    }
  }
  Value expanded_input = input;
  for (int64_t i : expanded_dimensions) {
    auto index_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({}, rewriter.getI64Type()), {i});
    Value index = rewriter.create<TF::ConstOp>(output.getLoc(), index_attr);
    expanded_input = rewriter.create<TF::ExpandDimsOp>(output.getLoc(),
                                                       expanded_input, index);
  }
  return expanded_input;
}

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_legalize_hlo.inc"

/// Performs the lowering to TF dialect.
void LegalizeHloToTf::runOnOperation() {
  MLIRContext& context = getContext();
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(&getContext());
  // Conversion patterns for custom calls.
  patterns.add<RemoveCustomCallWithShapeAssertion>(&context);
  patterns.add<ConvertCustomCallWithApproxTopK>(&context, &module);
  PopulateLegalizeHloToTfPatterns(&patterns, &context);

  ConversionTarget target(context);
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();
  target.addLegalOp<mhlo::TupleOp>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("mhlo to TF legalization failed.");
    signalPassFailure();
  }
}

}  // end namespace

void PopulateLegalizeHloToTfPatterns(RewritePatternSet* patterns,
                                     MLIRContext* context) {
  patterns
      ->add<ConvertAvgPoolOp, Convert2DConvOp, Convert3DConvOp, Convert1DConvOp,
            ConvertToResizeBilinearOpOrDepthwiseTransposedConvOp,
            ConvertNonTrivialConvOp, ConvertDynamicSliceOp,
            ConvertDynamicUpdateSliceOp, ConvertGatherOp, ConvertIfOp,
            ConvertMaxPoolOp, ConvertPopulationCountOp, ConvertSliceOp,
            ConvertReduceOpToTfArgmax, ConvertReduceOpToTfArgmin,
            ConvertReduceOpToTfMax, ConvertReduceOpToTfMin,
            ConvertReduceOpToTfAll, ConvertReduceOpToTfProd,
            ConvertReduceOpToTfAny, ConvertReduceOpToTfSum, ConvertSortToTfTopk,
            ConvertIotaOpToTfRange, ConvertWhileOp, ConvertLoweredCumSumOp,
            ConvertLoweredCumProdOp, ConvertGetDimensionSizeOp,
            ConvertDynamicIotaOp, ConvertRealDynamicSliceOp>(context);
  populateWithGenerated(*patterns);
}

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfPass() {
  return std::make_unique<LegalizeHloToTf>();
}

static PassRegistration<LegalizeHloToTf> pass;

}  // end namespace odml
}  // end namespace mlir
