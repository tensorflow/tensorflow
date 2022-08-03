/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/utils/broadcast_utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace mlir {
namespace TF {
namespace {

using mhlo::DotDimensionNumbersAttr;

// Replaces `region`'s terminator to TF::Yield.
void ReplaceReturnOp(Region &region, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  for (auto &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
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
LogicalResult GetConstantSplatValue(Value value, SplatValueType &splat_value) {
  DenseElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat()) {
    return failure();
  }

  splat_value = attr.getSplatValue<SplatValueType>();
  return success();
}

struct PermutationAndShape {
  DenseIntElementsAttr permutation;
  ShapedType shape;
};

// Returns a DenseIntElementsAttr for a permutation and the shape after
// applying the permutation to a given shape through a transpose.
PermutationAndShape GetPermutationAndTransposedShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter &rewriter) {
  assert(permutation_array.size() == input_type.getRank());
  llvm::SmallVector<int64_t> transposed_shape(permutation_array.size());
  for (int64_t i = 0; i < permutation_array.size(); ++i) {
    transposed_shape[i] = input_type.getDimSize(permutation_array[i]);
  }
  auto transposed_type =
      RankedTensorType::get(transposed_shape, input_type.getElementType());
  DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
      RankedTensorType::get(permutation_array.size(), rewriter.getI64Type()),
      permutation_array);
  return {permutation, transposed_type};
}

// Returns the inverse permutation array for a permutation array.
llvm::SmallVector<int64_t> GetInversePermutationArray(
    llvm::ArrayRef<int64_t> permutation_array) {
  llvm::SmallVector<int64_t> inverse_permutation_array(
      permutation_array.size());
  const auto permutation_array_size = permutation_array.size();
  for (int64_t i = 0; i < permutation_array_size; ++i) {
    inverse_permutation_array[permutation_array[i]] = i;
  }
  return inverse_permutation_array;
}

// Returns the DenseIntElementsAttr for an inverse permutation given a
// permutation_array.
DenseIntElementsAttr GetInversePermutation(
    llvm::ArrayRef<int64_t> permutation_array,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t, 4> inverse_permutation_array =
      GetInversePermutationArray(permutation_array);
  return DenseIntElementsAttr::get(
      RankedTensorType::get(inverse_permutation_array.size(),
                            rewriter.getI64Type()),
      inverse_permutation_array);
}

// Returns a DenseIntElementsAttr for an inverse permutation and the shape after
// applying the inverse permutation to a given shape through a transpose.
PermutationAndShape GetInversePermutationAndShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t, 4> inverse_permutation_array =
      GetInversePermutationArray(permutation_array);
  return GetPermutationAndTransposedShape(inverse_permutation_array, input_type,
                                          rewriter);
}

// Common functionality for ConvertConvOp classes.
template <int SupportedSpatialDims>
struct ConvertNdConvOp {
  bool IsSupportedConvOp(mhlo::ConvolutionOp conv_op) const {
    if (!conv_op.lhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.rhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.getType().cast<ShapedType>().hasStaticShape())
      return false;

    // All ones in "lhs_dilation" means this "mhlo.conv" op should be
    // converted to "tf.Conv2D" or "tf.DepthwiseConv2dNativeOp".
    if (conv_op.lhs_dilation().has_value()) {
      auto lhs_dilation = conv_op.lhs_dilation().getValue();
      if (!lhs_dilation.isSplat() || lhs_dilation.getSplatValue<int64_t>() != 1)
        return false;
    }

    if (!conv_op.window_strides().has_value() || conv_op.window_strides()
                                                         .getValue()
                                                         .getType()
                                                         .cast<ShapedType>()
                                                         .getRank() != 1)
      return false;

    auto num_spatial_dims =
        conv_op.dimension_numbers().getInputSpatialDimensions().size();
    // TODO(b/158636600): Currently we don't support 3D Convolution.
    if (num_spatial_dims != SupportedSpatialDims) return false;

    return true;
  }
};

// Convert a 1-D convolution into a 2-D convolution (which TF supports) so that
// it can be rewritten by the pattern `Convert2DConvOp`.
class Convert1DConvOp : public OpConversionPattern<mhlo::ConvolutionOp>,
                        ConvertNdConvOp<1> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    //
    // Check that input is a supported 1d convolution.
    //

    if (!IsSupportedConvOp(conv_op) || conv_op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(conv_op, "unsupported conv op.");

    const mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();

    // Group convolution is not supported yet.
    const int64_t input_feature_dimension = dnums.getInputFeatureDimension();
    const int64_t input_channels =
        conv_op.lhs().getType().cast<ShapedType>().getDimSize(
            input_feature_dimension);
    const int64_t feature_group_count = conv_op.feature_group_count();
    if (feature_group_count != 1 && feature_group_count != input_channels)
      return rewriter.notifyMatchFailure(conv_op,
                                         "Group convolution is not supported,");

    //
    // Transpose and reshape the input and kernel
    //

    // Reshape input image to add a new spatial dimension.
    auto image_type = conv_op.lhs().getType().cast<ShapedType>();
    SmallVector<int64_t, 4> image_2d_shape(image_type.getShape().begin(),
                                           image_type.getShape().end());
    image_2d_shape.push_back(1);
    auto image_2d_type =
        RankedTensorType::get(image_2d_shape, image_type.getElementType());
    auto image_2d_op = rewriter.create<mhlo::ReshapeOp>(
        conv_op.getLoc(), image_2d_type, conv_op.lhs());

    // Transpose image to get it into NWHC form (where H is the added dim).
    SmallVector<int64_t, 4> image_permutation = {
        dnums.getInputBatchDimension(), dnums.getInputSpatialDimensions()[0],
        3,  // The trailing dim that we added.
        dnums.getInputFeatureDimension()};
    auto image_permutation_and_shape = GetPermutationAndTransposedShape(
        image_permutation, image_2d_type, rewriter);
    auto transposed_image_2d_op = rewriter.create<mhlo::TransposeOp>(
        conv_op.getLoc(), image_permutation_and_shape.shape,
        image_2d_op->getResult(0), image_permutation_and_shape.permutation);

    // Reshape kernel to add a new spatial dimension.
    auto kernel_type = conv_op.rhs().getType().cast<ShapedType>();
    SmallVector<int64_t, 4> kernel_2d_shape;
    for (int64_t dim : kernel_type.getShape()) {
      kernel_2d_shape.push_back(dim);
    }
    kernel_2d_shape.push_back(1);
    auto kernel_2d_type =
        RankedTensorType::get(kernel_2d_shape, kernel_type.getElementType());
    auto kernel_2d_op = rewriter.create<mhlo::ReshapeOp>(
        conv_op.getLoc(), kernel_2d_type, conv_op.rhs());

    // Transpose kernel to get it into WHIO form (where H is the added dim).
    SmallVector<int64_t, 4> kernel_permutation = {
        dnums.getKernelSpatialDimensions()[0],
        3,  // The trailing dim that we added.
        dnums.getKernelInputFeatureDimension(),
        dnums.getKernelOutputFeatureDimension()};
    auto kernel_permutation_and_shape = GetPermutationAndTransposedShape(
        kernel_permutation, kernel_2d_type, rewriter);
    auto transposed_kernel_2d_op = rewriter.create<mhlo::TransposeOp>(
        conv_op.getLoc(), kernel_permutation_and_shape.shape,
        kernel_2d_op->getResult(0), kernel_permutation_and_shape.permutation);

    //
    // Create 2d equivalents for 1d convolution attributes.
    //

    // Window Strides
    SmallVector<int64_t, 2> window_strides_2d_array;
    for (const auto v : conv_op.window_strides()->getValues<int64_t>()) {
      window_strides_2d_array.emplace_back(v);
    }
    window_strides_2d_array.push_back(1);
    auto window_strides_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        window_strides_2d_array);

    // Padding
    SmallVector<int64_t, 4> padding_2d_array;
    for (const auto v : conv_op.padding().getValue().getValues<int64_t>()) {
      padding_2d_array.emplace_back(v);
    }
    // The newly added spatial dimension requires zero left and right padding.
    padding_2d_array.push_back(0);
    padding_2d_array.push_back(0);
    auto padding_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2, 2}, rewriter.getI64Type()), padding_2d_array);

    // LHS dilation
    SmallVector<int64_t, 4> lhs_dilation_array_2d;
    for (const auto v :
         conv_op.lhs_dilation().getValue().getValues<int64_t>()) {
      lhs_dilation_array_2d.emplace_back(v);
    }
    lhs_dilation_array_2d.push_back(1);
    auto lhs_dilation_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        lhs_dilation_array_2d);

    // RHS dilation
    SmallVector<int64_t, 4> rhs_dilation_array_2d;
    for (const auto v :
         conv_op.rhs_dilation().getValue().getValues<int64_t>()) {
      rhs_dilation_array_2d.emplace_back(v);
    }
    rhs_dilation_array_2d.push_back(1);
    auto rhs_dilation_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        rhs_dilation_array_2d);

    // Window reversal is unsupported.
    if (conv_op.window_reversal().has_value() &&
        conv_op.window_reversal()->getValues<bool>()[0] == true)
      return failure();
    auto window_reversal_2d = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()),
        SmallVector<int64_t>({0, 0}));

    // Precision config
    if (!conv_op.precision_config().has_value()) return failure();

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
    auto output_type = conv_op->getResult(0).getType().cast<ShapedType>();
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
        conv_op.getLoc(), transposed_output_2d_shape,
        transposed_image_2d_op.getResult(), transposed_kernel_2d_op.getResult(),
        window_strides_2d, padding_2d, lhs_dilation_2d, rhs_dilation_2d,
        window_reversal_2d, dnums_2d, conv_op.feature_group_count(),
        conv_op.batch_group_count(), *conv_op.precision_config());

    OpResult conv2d_output = conv2d_op->getResult(0);
    auto conv2d_output_type = conv2d_output.getType().cast<ShapedType>();

    //
    // Transpose and reshape the output
    //

    // Since output is in NWHC form we need to undo the permutation we have
    // affectively applied.
    auto output_permutation_and_shape = GetInversePermutationAndShape(
        output_permutation, conv2d_output_type, rewriter);
    auto transposed_output_2d_op = rewriter.create<mhlo::TransposeOp>(
        conv_op.getLoc(), output_permutation_and_shape.shape, conv2d_output,
        output_permutation_and_shape.permutation);

    // Drop the trailing spatial dimension from the output.
    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
        conv_op, output_type, transposed_output_2d_op.getResult());
    return success();
  }
};

class Convert2DConvOp : public OpConversionPattern<mhlo::ConvolutionOp>,
                        ConvertNdConvOp<2> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!IsSupportedConvOp(conv_op)) {
      return failure();
    }

    // Constructs strides array.
    // For example, [2, 3] -> [1, 2, 3, 1].
    SmallVector<int64_t, 4> strides({1});
    for (const auto v :
         conv_op.window_strides().getValue().getValues<int64_t>()) {
      strides.emplace_back(v);
    }
    strides.emplace_back(1);

    // Constructs dilation array.
    SmallVector<int64_t, 4> dilation;
    if (auto rhs_dilation = conv_op.rhs_dilation()) {
      // For example, [2, 3] -> [1, 2, 3, 1].
      dilation.emplace_back(1);
      dilation.append(rhs_dilation.getValue().getValues<int64_t>().begin(),
                      rhs_dilation.getValue().getValues<int64_t>().end());
      dilation.emplace_back(1);
    } else {
      // Default value
      dilation = {1, 1, 1, 1};
    }

    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
    const int input_feature_dimension = dnums.getInputFeatureDimension();
    const int input_channels =
        conv_op.lhs().getType().cast<ShapedType>().getDimSize(
            input_feature_dimension);
    int feature_group_count = conv_op.feature_group_count();

    if (feature_group_count != 1 && feature_group_count != input_channels) {
      // Group convolution is not supported yet.
      return failure();
    }

    const int num_spatial_dims = dnums.getInputSpatialDimensions().size();
    const bool is_depthwise_conv = input_channels == feature_group_count;
    std::string padding;
    SmallVector<int64_t, 8> explicit_padding;
    if (!conv_op.padding().has_value() ||
        (conv_op.padding().getValue().isSplat() &&
         conv_op.padding()->getSplatValue<int64_t>() == 0)) {
      padding = "VALID";
    } else {
      SmallVector<int64_t, 4> padding_array;
      for (const auto v : conv_op.padding().getValue().getValues<int64_t>()) {
        padding_array.emplace_back(v);
      }

      if (IsSamePadding(conv_op, num_spatial_dims, strides, dilation,
                        padding_array)) {
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
                 is_depthwise_conv, input_channels, num_spatial_dims, rewriter);
    return success();
  };

 private:
  bool IsSamePadding(mhlo::ConvolutionOp conv_op, int num_spatial_dims,
                     ArrayRef<int64_t> strides, ArrayRef<int64_t> dilation,
                     ArrayRef<int64_t> padding_array) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
    auto input_spatial_dim = dnums.getInputSpatialDimensions();
    auto kernel_spatial_dim = dnums.getKernelSpatialDimensions();
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      int dim = i + 1;
      int64_t output_size;
      int64_t pad_low_int64;
      int64_t pad_high_int64;
      tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
          conv_op.lhs().getType().cast<ShapedType>().getDimSize(
              input_spatial_dim[i]),
          conv_op.rhs().getType().cast<ShapedType>().getDimSize(
              kernel_spatial_dim[i]),
          dilation[dim], strides[dim], tensorflow::Padding::SAME, &output_size,
          &pad_low_int64, &pad_high_int64);
      if (!status.ok()) return false;
      if (padding_array[2 * i] != pad_low_int64 ||
          padding_array[2 * i + 1] != pad_high_int64)
        return false;
    }

    return true;
  }

  // Returns true if the op needs reformat.
  bool NeedsReformatTypeAndPermutation(int batch_dim, int feature_dim,
                                       int spatial_dim_start,
                                       int default_batch_dim,
                                       int default_feature_dim,
                                       int default_spatial_dim_start) const {
    return batch_dim != default_batch_dim ||
           feature_dim != default_feature_dim ||
           spatial_dim_start != default_spatial_dim_start;
  }

  // Gets reformat type and permutation attribute. Call this function only if
  // NeedsReformatTypeAndPermutation returns true.
  std::pair<RankedTensorType, DenseIntElementsAttr>
  GetReformatTypeAndPermutation(int batch_dim, int feature_dim,
                                int spatial_dim_start, int default_batch_dim,
                                int default_feature_dim,
                                int default_spatial_dim_start,
                                int num_spatial_dims, RankedTensorType type,
                                ConversionPatternRewriter &rewriter) const {
    auto shape = type.getShape();
    llvm::SmallVector<int64_t, 4> permutation_array(num_spatial_dims + 2);
    permutation_array[default_batch_dim] = batch_dim;
    permutation_array[default_feature_dim] = feature_dim;
    llvm::SmallVector<int64_t, 4> transposed_shape(num_spatial_dims + 2);
    transposed_shape[default_batch_dim] = shape[batch_dim];
    transposed_shape[default_feature_dim] = shape[feature_dim];
    for (int i : llvm::seq<int>(0, num_spatial_dims)) {
      permutation_array[default_spatial_dim_start + i] = spatial_dim_start + i;
      transposed_shape[default_spatial_dim_start + i] =
          shape[spatial_dim_start + i];
    }
    auto new_type =
        RankedTensorType::get(transposed_shape, type.getElementType());
    auto permutation = DenseIntElementsAttr::get(
        RankedTensorType::get({type.getRank()}, rewriter.getI64Type()),
        permutation_array);
    return {new_type, permutation};
  }

  Value FormatToNHWC(Value value, int batch_dim, int feature_dim,
                     ArrayRef<int64_t> spatial_dimensions,
                     int default_batch_dim, int default_feature_dim,
                     int default_spatial_dim_start, int num_spatial_dims,
                     ConversionPatternRewriter &rewriter) const {
    auto type = value.getType().cast<RankedTensorType>();
    DenseIntElementsAttr permutation;
    const int spatial_dim_start = spatial_dimensions.front();
    if (!NeedsReformatTypeAndPermutation(
            batch_dim, feature_dim, spatial_dim_start, default_batch_dim,
            default_feature_dim, default_spatial_dim_start)) {
      // Transpose is not needed because the current format is "NHWC".
      return value;
    }
    std::pair<RankedTensorType &, DenseIntElementsAttr &>(type, permutation) =
        GetReformatTypeAndPermutation(batch_dim, feature_dim, spatial_dim_start,
                                      default_batch_dim, default_feature_dim,
                                      default_spatial_dim_start,
                                      num_spatial_dims, type, rewriter);
    return rewriter.create<mhlo::TransposeOp>(value.getLoc(), type, value,
                                              permutation);
  }

  // Slices the input `value` if there are negative padding values in
  // `explicit_padding`.
  Value SliceNegativePadding(Value value, ArrayRef<int64_t> explicit_padding,
                             ConversionPatternRewriter &rewriter) const {
    // If no padding is negative return the input as is.
    if (llvm::all_of(explicit_padding, [](int64_t pad) { return pad >= 0; })) {
      return value;
    }

    auto input_type = value.getType().cast<RankedTensorType>();
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

    auto start_attr = rewriter.create<ConstOp>(
        value.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(start.size())},
                                  rewriter.getI64Type()),
            start));
    auto size_attr = rewriter.create<ConstOp>(
        value.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(size.size())},
                                  rewriter.getI64Type()),
            size));
    auto output_type = RankedTensorType::get(size, input_type.getElementType());

    return rewriter.create<SliceOp>(value.getLoc(), output_type, value,
                                    start_attr, size_attr);
  }

  void CreateConvOp(mhlo::ConvolutionOp conv_op, ArrayRef<int64_t> strides,
                    StringRef padding, ArrayRef<int64_t> explicit_padding,
                    ArrayRef<int64_t> dilation, bool is_depthwise_conv,
                    int input_channels, int num_spatial_dims,
                    ConversionPatternRewriter &rewriter) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
    // Transposes lhs and rhs if their formats are not NHWC.
    Value lhs = FormatToNHWC(
        conv_op.lhs(), dnums.getInputBatchDimension(),
        dnums.getInputFeatureDimension(), dnums.getInputSpatialDimensions(),
        /*default_batch_dim=*/0, /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/1, num_spatial_dims, rewriter);
    Value rhs = FormatToNHWC(
        conv_op.rhs(), dnums.getKernelInputFeatureDimension(),
        dnums.getKernelOutputFeatureDimension(),
        dnums.getKernelSpatialDimensions(),
        /*default_batch_dim=*/num_spatial_dims,
        /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/0, num_spatial_dims, rewriter);

    // Emulate negative padding with a slice and remove negative values from the
    // padding vector.
    Value sliced_lhs = SliceNegativePadding(lhs, explicit_padding, rewriter);
    auto new_padding = llvm::to_vector<4>(llvm::map_range(
        explicit_padding, [](int64_t dim) { return dim > 0 ? dim : 0; }));

    auto conv_output_type = conv_op.getType().cast<RankedTensorType>();
    DenseIntElementsAttr permutation;
    const bool need_transpose_output = NeedsReformatTypeAndPermutation(
        dnums.getOutputBatchDimension(), dnums.getOutputFeatureDimension(),
        dnums.getOutputSpatialDimensions().front(),
        /*default_batch_dim=*/0, /*default_feature_dim=*/num_spatial_dims + 1,
        /*default_spatial_dim_start=*/1);
    if (need_transpose_output) {
      std::pair<RankedTensorType &, DenseIntElementsAttr &>(conv_output_type,
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
    if (is_depthwise_conv) {
      // Reshapes filter format to [filter_height, filter_width, in_channels,
      // channel_multiplier] from HLO's [filter_height, filter_width, 1,
      // in_channels * channel_multiplier] format.
      auto filter_type = rhs.getType().cast<ShapedType>();
      llvm::ArrayRef<int64_t> hlo_filter_shape = filter_type.getShape();
      llvm::SmallVector<int64_t, 4> tf_filter_shape(hlo_filter_shape.begin(),
                                                    hlo_filter_shape.end());
      tf_filter_shape[2] = input_channels;
      tf_filter_shape[3] = hlo_filter_shape.back() / input_channels;
      auto reshaped_filter = rewriter.create<mhlo::ReshapeOp>(
          rhs.getLoc(),
          RankedTensorType::get(tf_filter_shape, filter_type.getElementType()),
          rhs);

      output = rewriter.create<DepthwiseConv2dNativeOp>(
          conv_op.getLoc(), conv_output_type, sliced_lhs, reshaped_filter,
          rewriter.getI64ArrayAttr(strides),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr(new_padding),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    } else {
      output = rewriter.create<Conv2DOp>(
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
      std::pair<RankedTensorType &, DenseIntElementsAttr &>(conv_output_type,
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

class ConvertNonTrivialConvOp
    : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp conv_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (IsSupportedConvOp(conv_op, rewriter).failed()) {
      return rewriter.notifyMatchFailure(
          conv_op,
          "doesn't support to convert to ConvBackpropInputOp or "
          "ResizeBilinearOp");
    }

    // tf.ResizeBilinearOp is perferred than tf.Conv2DBackpropInputOp since
    // the former has better portability, especially in inference use cases.
    bool align_corners;
    llvm::SmallVector<int, 2> output_sizes;
    if (MatchResizeOp(conv_op, align_corners, output_sizes, rewriter)
            .succeeded()) {
      CreateResizeBilinearOp(conv_op, output_sizes, align_corners, rewriter);
      return success();
    }

    // Constructs strides array from lhs_dilation.
    // For example, [2, 3] -> [1, 2, 3, 1].
    SmallVector<int64_t, 4> strides({1});
    strides.append(
        conv_op.lhs_dilation().getValue().getValues<int64_t>().begin(),
        conv_op.lhs_dilation().getValue().getValues<int64_t>().end());
    strides.emplace_back(1);

    // Constructs dilation array.
    SmallVector<int64_t, 4> dilation;
    if (auto rhs_dilation = conv_op.rhs_dilation()) {
      // For example, [2, 3] -> [1, 2, 3, 1].
      dilation.emplace_back(1);
      dilation.append(rhs_dilation.getValue().getValues<int64_t>().begin(),
                      rhs_dilation.getValue().getValues<int64_t>().end());
      dilation.emplace_back(1);
    } else {
      // Default value
      dilation = {1, 1, 1, 1};
    }

    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
    std::string padding;
    if (!conv_op.padding().has_value() ||
        (conv_op.padding().getValue().isSplat() &&
         conv_op.padding()->getSplatValue<int64_t>() == 0)) {
      padding = "VALID";
    } else {
      auto spatial_dims = dnums.getInputSpatialDimensions();
      int num_spatial_dims =
          std::accumulate(spatial_dims.begin(), spatial_dims.end(), 1LL,
                          std::multiplies<int64_t>{});
      if (!IsSamePadding(conv_op, num_spatial_dims, strides)) {
        return rewriter.notifyMatchFailure(
            conv_op, "requires padding to be SAME or VALID");
      }
      padding = "SAME";
    }

    // Converts int64_t to int32_t.
    llvm::SmallVector<int, 4> input_shape;
    for (int64_t dim : conv_op.getType().cast<RankedTensorType>().getShape()) {
      input_shape.push_back(dim);
    }
    auto input_sizes = rewriter.create<ConstOp>(
        conv_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(input_shape.size())},
                                  rewriter.getI32Type()),
            input_shape));
    // Mirror the filter in the spatial dimensions.
    auto filter = rewriter.create<mhlo::ReverseOp>(
        conv_op.getLoc(), conv_op.rhs(),
        rewriter.getI64TensorAttr(dnums.getKernelSpatialDimensions()));
    rewriter.replaceOpWithNewOp<Conv2DBackpropInputOp>(
        conv_op, conv_op.getType(), input_sizes, filter, conv_op.lhs(),
        rewriter.getI64ArrayAttr(strides),
        /*use_cudnn_on_gpu=*/rewriter.getBoolAttr(true),
        /*padding=*/rewriter.getStringAttr(padding),
        /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
        /*data_format=*/rewriter.getStringAttr("NHWC"),
        /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    return success();
  };

 private:
  bool IsSamePadding(mhlo::ConvolutionOp conv_op, int num_spatial_dims,
                     ArrayRef<int64_t> strides) const {
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      int dim = i + 1;
      int stride = strides[dim];
      int input_size = conv_op.getType().cast<ShapedType>().getDimSize(dim);
      int output_size =
          conv_op.lhs().getType().cast<ShapedType>().getDimSize(dim);
      if (output_size != (input_size + stride - 1) / stride) {
        return false;
      }
    }

    return true;
  }

  LogicalResult IsSupportedConvOp(mhlo::ConvolutionOp conv_op,
                                  ConversionPatternRewriter &rewriter) const {
    if (!conv_op.lhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.rhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.getType().cast<ShapedType>().hasStaticShape())
      return rewriter.notifyMatchFailure(conv_op, "requires static shape");
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
    const int input_feature_dimension = dnums.getInputFeatureDimension();
    const int input_channels =
        conv_op.lhs().getType().cast<ShapedType>().getDimSize(
            input_feature_dimension);
    int feature_group_count = conv_op.feature_group_count();

    if (feature_group_count != 1 && feature_group_count != input_channels) {
      // Group convolution is not supported yet.
      return rewriter.notifyMatchFailure(conv_op,
                                         "doesn't support group convolution");
    }

    // Checks lhs_dilation is non-trivial.
    if (!conv_op.lhs_dilation().has_value()) {
      return rewriter.notifyMatchFailure(conv_op,
                                         "requires lhs_dilation attribute");
    }
    auto lhs_dilation = conv_op.lhs_dilation().getValue();
    if (lhs_dilation.isSplat() && lhs_dilation.getSplatValue<int64_t>() == 1)
      return rewriter.notifyMatchFailure(conv_op,
                                         "requires non-trivial lhs_dilation");

    if (!conv_op.window_strides().has_value() || conv_op.window_strides()
                                                         .getValue()
                                                         .getType()
                                                         .cast<ShapedType>()
                                                         .getRank() != 1)
      return rewriter.notifyMatchFailure(
          conv_op, "requires window_strides to equal to one");

    int num_spatial_dims = dnums.getInputSpatialDimensions().size();
    // TODO(chhe): Currently we don't support 3D Convolution.
    if (num_spatial_dims != 2)
      return rewriter.notifyMatchFailure(conv_op,
                                         "doesn't support more than 2D");

    // TODO(chhe): To support more data formats other than "NHWC".
    // Checks format [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f].
    if (dnums.getInputBatchDimension() != 0 ||
        dnums.getInputFeatureDimension() != num_spatial_dims + 1)
      return rewriter.notifyMatchFailure(conv_op,
                                         "requires input format [b, 0, 1, f]");
    auto input_spatial_dimensions = dnums.getInputSpatialDimensions();
    for (auto p : llvm::enumerate(input_spatial_dimensions)) {
      if (p.value() != p.index() + 1)
        return rewriter.notifyMatchFailure(
            conv_op, "requires input format [b, 0, 1, f]");
    }

    // Checks output dimensions.
    if (dnums.getOutputBatchDimension() != 0 ||
        conv_op.dimension_numbers().getOutputFeatureDimension() !=
            num_spatial_dims + 1)
      return rewriter.notifyMatchFailure(conv_op,
                                         "requires output format [b, 0, 1, f]");
    auto output_spatial_dimensions = dnums.getOutputSpatialDimensions();
    for (auto p : llvm::enumerate(output_spatial_dimensions)) {
      if (p.value() != p.index() + 1)
        return rewriter.notifyMatchFailure(
            conv_op, "requires output format [b, 0, 1, f]");
    }

    // Checks kernel dimensions.
    if (dnums.getKernelInputFeatureDimension() != num_spatial_dims + 1 ||
        dnums.getKernelOutputFeatureDimension() != num_spatial_dims)
      return rewriter.notifyMatchFailure(conv_op,
                                         "requires kernel format [b, 0, 1, f]");
    auto kernel_spatial_dimensions = dnums.getKernelSpatialDimensions();
    for (auto p : llvm::enumerate(kernel_spatial_dimensions)) {
      if (p.value() != p.index())
        return rewriter.notifyMatchFailure(
            conv_op, "requires kernel format [0, 1, o, i]");
    }

    return success();
  }

  void CreateResizeBilinearOp(mhlo::ConvolutionOp conv_op,
                              llvm::ArrayRef<int32_t> output_sizes,
                              bool align_corners,
                              ConversionPatternRewriter &rewriter) const {
    Value output_sizes_attr = rewriter.create<ConstOp>(
        conv_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(output_sizes.size())},
                                  rewriter.getI32Type()),
            output_sizes));
    // The value of half_pixel_centers couldn't be inferred from the IR and XLA
    // only support half_pixel_centers=True as in 01/11/2022. Here
    // half_pixel_centers=False is hardcoded.
    Value output = rewriter.create<ResizeBilinearOp>(
        conv_op.getLoc(), conv_op.getType(), conv_op.lhs(), output_sizes_attr,
        /*align_corners=*/rewriter.getBoolAttr(align_corners),
        /*half_pixel_centers=*/rewriter.getBoolAttr(false));
    rewriter.replaceOp(conv_op, {output});
  }

  LogicalResult MatchResizeOp(mhlo::ConvolutionOp conv_op, bool &align_corners,
                              llvm::SmallVector<int, 2> &output_sizes,
                              ConversionPatternRewriter &rewriter) const {
    mhlo::ConvDimensionNumbersAttr dnums = conv_op.dimension_numbers();
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

    // When "lhs_dilation" is 2D and contains at least "1", and "rhs_dilation"
    // are all "1"s, this "mhlo.conv" op can potentially be converted to
    // "tf.ResizeBilinearOp".
    if (!conv_op.rhs_dilation().has_value() || !conv_op.padding().has_value())
      return rewriter.notifyMatchFailure(
          conv_op, "resize op requires rhs_dilation and padding");

    auto lhs_dilation = conv_op.lhs_dilation().getValue();
    auto rhs_dilation = conv_op.rhs_dilation().getValue();
    auto window_strides = conv_op.window_strides().getValue();
    auto padding = conv_op.padding().getValue();
    if (lhs_dilation.getNumElements() != 2 || !rhs_dilation.isSplat() ||
        rhs_dilation.getSplatValue<int64_t>() != 1 ||
        window_strides.getNumElements() != 2 || padding.getNumElements() != 4)
      return rewriter.notifyMatchFailure(
          conv_op, "resize op requires [2] dilations and [2,2] padding");
    auto lhs_dilation_values = lhs_dilation.getValues<int64_t>();
    auto window_strides_values = window_strides.getValues<int64_t>();
    auto padding_values = padding.getValues<int64_t>();

    // Cast the dimension sizes to int.
    auto lhs_type = conv_op.lhs().getType().cast<ShapedType>();
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

class ConvertSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto begin =
        rewriter.create<ConstOp>(slice_op.getLoc(), slice_op.start_indices());
    auto end =
        rewriter.create<ConstOp>(slice_op.getLoc(), slice_op.limit_indices());
    auto strides =
        rewriter.create<ConstOp>(slice_op.getLoc(), slice_op.strides());
    rewriter.replaceOpWithNewOp<StridedSliceOp>(
        slice_op, slice_op.getType(), slice_op.operand(), begin, end, strides);
    return success();
  }
};

class ConvertDynamicSliceOp : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    ShapedType input_type = op.operand().getType().cast<ShapedType>();
    if (!input_type.hasStaticShape()) return failure();
    Type start_indices_element_type = op.start_indices()
                                          .front()
                                          .getType()
                                          .cast<ShapedType>()
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
    start_indices_vector.reserve(op.start_indices().size());
    Value clamp_min = rewriter.create<ConstOp>(
        op.getLoc(),
        rewriter.getIntegerAttr(signed_start_indices_element_type, 0));
    for (uint64_t i = 0, e = op.start_indices().size(); i < e; ++i) {
      // Always put a cast there.
      auto start = op.start_indices()[i];
      auto cast_type = start.getType().cast<ShapedType>().clone(
          signed_start_indices_element_type);
      auto cast_op = rewriter.create<CastOp>(op.getLoc(), cast_type, start);
      Value clamp_max = rewriter.create<ConstOp>(
          op.getLoc(), rewriter.getIntegerAttr(
                           signed_start_indices_element_type,
                           input_type.getShape()[i] -
                               op.slice_sizes().getValues<int64_t>()[i]));
      Value clamped_index = rewriter.create<mhlo::ClampOp>(
          op.getLoc(), cast_type, clamp_min, cast_op, clamp_max);
      start_indices_vector.push_back(clamped_index);
    }

    // Pack individual start indices to start indices tensor.
    Type start_indices_type = RankedTensorType::get(
        {static_cast<int64_t>(start_indices_vector.size())},
        signed_start_indices_element_type);
    Value start_indices_op = rewriter.create<PackOp>(
        op.getLoc(), start_indices_type, ValueRange(start_indices_vector));

    Value slice_sices_op =
        rewriter.create<ConstOp>(op.getLoc(), op.slice_sizes());
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getType(), op.operand(),
                                         start_indices_op, slice_sices_op);
    return success();
  };
};

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range>
void Append(llvm::SmallVectorImpl<ValueT> &values, Range &&range) {
  values.insert(values.end(), range.begin(), range.end());
}

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range, typename... RangeTs>
void Append(llvm::SmallVectorImpl<ValueT> &values, Range &&range,
            RangeTs &&...ranges) {
  values.insert(values.end(), range.begin(), range.end());
  Append(values, ranges...);
}

// Returns the number of elements in `range`.
template <typename Range>
size_t Size(Range &&range) {
  return range.size();
}

// Returns the total number of elements in a variadic number of `ranges`.
template <typename Range, typename... RangeTs>
size_t Size(Range &&range, RangeTs &&...ranges) {
  return range.size() + Size(std::forward<RangeTs>(ranges)...);
}

// Concats all elements in `ranges` and returns a small vector as a result.
template <typename ValueT, typename... RangeTs>
llvm::SmallVector<ValueT, 4> Concat(RangeTs &&...ranges) {
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

// Create a single const integer.
Value BuildIntConstOp(ImplicitLocOpBuilder &builder,
                      ConversionPatternRewriter &rewriter, int64_t const_value,
                      Type type) {
  Value result_const =
      builder.create<ConstOp>(rewriter.getIntegerAttr(type, const_value));
  return result_const;
}
// Create a const integer vector tensor (1-dim).
Value BuildIntArrayConstOp(ImplicitLocOpBuilder &builder,
                           ConversionPatternRewriter &rewriter,
                           ArrayRef<int64_t> const_value, Type type) {
  DenseIntElementsAttr const_value_raw;
  if (type == rewriter.getI64Type()) {
    const_value_raw = rewriter.getI64TensorAttr(const_value);
  } else {
    // Convert I64 const array to I32.
    llvm::SmallVector<int32_t> const_i32_vec;
    for (auto element : const_value) {
      const_i32_vec.push_back(static_cast<int32_t>(element));
    }
    const_value_raw = rewriter.getI32TensorAttr(const_i32_vec);
  }
  Value result_const = builder.create<ConstOp>(const_value_raw);
  return result_const;
}

// Create a tensor that is reshaped from input.
Value BuildReshapeOp(ImplicitLocOpBuilder &builder,
                     ConversionPatternRewriter &rewriter, Value input,
                     ArrayRef<int64_t> shape, Type idx_type,
                     Type element_type) {
  Value shape_cst = BuildIntArrayConstOp(builder, rewriter, shape, idx_type);
  Value reshaped_input = builder.create<ReshapeOp>(
      RankedTensorType::get(shape, element_type), input, shape_cst);
  return reshaped_input;
}

// Create a tensor which is equal to input[begin: begin + size].
Value BuildSliceOp(ImplicitLocOpBuilder &builder,
                   ConversionPatternRewriter &rewriter, Value input,
                   Value begin, ArrayRef<int64_t> shape, Type idx_type,
                   Type element_type) {
  Value shape_cst = BuildIntArrayConstOp(builder, rewriter, shape, idx_type);
  Value slice_result = builder.create<SliceOp>(
      RankedTensorType::get(shape, element_type), input, begin, shape_cst);
  return slice_result;
}

class ConvertDynamicUpdateSliceOp
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    ShapedType operand_type = op.operand().getType().cast<ShapedType>();
    ShapedType update_type =
        op.update().getType().dyn_cast_or_null<ShapedType>();
    ShapedType start_indices_type =
        op.start_indices().front().getType().dyn_cast_or_null<ShapedType>();
    if (update_type == nullptr || start_indices_type == nullptr)
      return rewriter.notifyMatchFailure(
          op, "update and start_indices should have ShapedType");
    if (!operand_type.hasStaticShape() || !update_type.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "shape of operand and update should be static");

    Type idx_type = start_indices_type.getElementType();
    int64_t shape_dim = operand_type.getRank();
    auto operand_shape = operand_type.getShape();
    auto update_shape = update_type.getShape();

    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    Value zero_cst = BuildIntConstOp(builder, rewriter, 0, idx_type);
    Value one_cst = BuildIntConstOp(builder, rewriter, 1, idx_type);
    // Clamp start indices in [0, operand_size - update_size].
    llvm::SmallVector<Value> start_indices_vector;
    Append(start_indices_vector, op.start_indices());
    auto shape_tensor_type = RankedTensorType::get({shape_dim}, idx_type);
    Value start_indices_tensor =
        builder.create<PackOp>(shape_tensor_type, start_indices_vector);
    Value operand_shape_cst =
        BuildIntArrayConstOp(builder, rewriter, operand_shape, idx_type);
    Value update_shape_cst =
        BuildIntArrayConstOp(builder, rewriter, update_shape, idx_type);
    Value max_start_indices =
        builder.create<SubOp>(operand_shape_cst, update_shape_cst);
    Value start_indices_clip_max =
        builder.create<MinimumOp>(start_indices_tensor, max_start_indices);
    Value clamped_start_indices =
        builder.create<MaximumOp>(start_indices_clip_max, zero_cst);

    // Do dynamic_upate_slice on flattened operand and update with the aid of
    // tf.TensorScatterUpdate op. It takes in 3 parameters: flat_operand,
    // indices and flat_update. The indices are computed as follows:
    // 1. Construct a range (0, n_operand). It arranges a id number to each
    //    element position in operand.
    // 2. Reshape the range to the shape of operand.
    // 3. Compute the id numbers of update positions by choose a slice form
    //    clamped_start_indices to clamped_start_indices + update_size.
    // 4. Flatten the update id numbers and the indices is obtained.
    int64_t n_operand = operand_type.getNumElements();
    Value n_operand_cst =
        BuildIntConstOp(builder, rewriter, n_operand, idx_type);
    Value range_flat =
        builder.create<RangeOp>(zero_cst, n_operand_cst, one_cst);
    Value range = BuildReshapeOp(builder, rewriter, range_flat, operand_shape,
                                 idx_type, idx_type);
    Value update_indices_raw =
        BuildSliceOp(builder, rewriter, range, clamped_start_indices,
                     update_shape, idx_type, idx_type);
    int64_t n_update = update_type.getNumElements();
    Type element_type = operand_type.getElementType();
    Value update_indices = BuildReshapeOp(builder, rewriter, update_indices_raw,
                                          {n_update, 1}, idx_type, idx_type);
    Value operand_flat = BuildReshapeOp(builder, rewriter, op.operand(),
                                        {n_operand}, idx_type, element_type);
    Value update_flat = BuildReshapeOp(builder, rewriter, op.update(),
                                       {n_update}, idx_type, element_type);
    Value flat_result = builder.create<TensorScatterUpdateOp>(
        operand_flat, update_indices, update_flat);

    // Reshape back before return.
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, operand_type, flat_result,
                                           operand_shape_cst);
    return success();
  };
};

// It returns "true" when Value $iota is obtained from the following mlir code:
//
// $iota = "mhlo.iota"(){iota_dimension = $dimensions[0]},
//
// where $dimensions must have size 1 and iota can have rank>=1.
// It usually used for matching rank 1 iota since the iotaOp will be folded to
// IotaOp + BroadCastInDimOp except for the case when result shape is rank 1.
bool MatchSingleIota(DenseIntElementsAttr dimensions, Value iota) {
  auto iota_op = dyn_cast_or_null<mhlo::IotaOp>(iota.getDefiningOp());
  if (!iota_op || dimensions.getNumElements() != 1) return false;
  auto dim = *dimensions.value_begin<APInt>();
  return dim == iota_op.iota_dimension();
}

// It matches %iota generated from the following mlir codes:
//
// %iota_r1 = "mhlo.iota"(){iota_dimension = 0} :() -> tensor<Lxi32>
// %iota = "mhlo.broadcast_in_dim(%iota_r1){
//    broadcast_dimensions = dense<[$dimensions[0]]>},
//
// where %dimensions is of size 1. It ususally comes from an IotaOp that is
// folded to IotaOp (rank1) + BroadCastInDimOp.
bool MatchIotaBroadCastInDim(DenseIntElementsAttr dimensions, Value iota) {
  auto iota_broadcast =
      dyn_cast_or_null<mhlo::BroadcastInDimOp>(iota.getDefiningOp());
  if (!iota_broadcast || iota_broadcast.broadcast_dimensions() != dimensions)
    return false;
  if (!isa_and_nonnull<mhlo::IotaOp>(iota_broadcast.operand().getDefiningOp()))
    return false;
  return true;
}

// Matches %iota generated from the following code (rank 3 example):
//
// %iota_r1 = "mhlo.iota"(){iota_dimension = 0 : i32} : () -> tensor<44xi32>
// %iota = "mhlo.reshape"(%iota_r1): (tensor<44xi32>) -> tensor<1x1x44xi32>
//
// Where $dimensions is of size 1 and $dimensions[0] = 2.
//
// In general matches a 1-D Iota with multiple dimensions of size 1 added
// through a reshape.
bool MatchReshapedIota(DenseIntElementsAttr dimensions, Value iota) {
  if (dimensions.getNumElements() != 1) return false;
  auto reshape_op = dyn_cast_or_null<mhlo::ReshapeOp>(iota.getDefiningOp());
  if (!reshape_op) return false;
  auto operand_type =
      reshape_op.operand().getType().dyn_cast<RankedTensorType>();
  if (!operand_type || !operand_type.hasStaticShape()) return false;
  auto reshape_type = reshape_op.getType().cast<RankedTensorType>();

  // Reshape can take a 1-D iota input and add extra dims of size one.
  if (operand_type.getRank() != 1) return false;
  if (!dyn_cast_or_null<mhlo::IotaOp>(reshape_op.operand().getDefiningOp()))
    return false;

  int64_t iota_dim = (*dimensions.value_begin<APInt>()).getSExtValue();
  for (int64_t i = 0, e = reshape_type.getRank(); i < e; ++i) {
    if (i == iota_dim) {
      if (reshape_type.getDimSize(i) != operand_type.getDimSize(0))
        return false;
    } else if (reshape_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

// It matches %iota generated from the following mlir codes:
//
// %iota_r1 = mhlo.constant dense<[0, 1, 2, ..., L]>
// %iota = "mhlo.broadcast_in_dim(%iota_r1){
//    broadcast_dimensions = dense<[$dimensions[0]]>},
//
// where $dimensions is of size 1. It ususally comes from an IotaOp that is
// folded to ConstOp (folded rank1 iota) + BroadCastInDimOp.
bool MatchConstIotaBroadCastInDim(DenseIntElementsAttr dimensions, Value iota) {
  if (dimensions.getNumElements() != 1) return false;
  auto iota_broadcast =
      dyn_cast_or_null<mhlo::BroadcastInDimOp>(iota.getDefiningOp());
  if (!iota_broadcast || iota_broadcast.broadcast_dimensions() != dimensions)
    return false;
  DenseElementsAttr range_const;
  if (!matchPattern(iota_broadcast.operand(), m_Constant(&range_const)))
    return false;
  int index = 0;
  for (auto value : range_const.getValues<APInt>()) {
    if (value != index++) return false;
  }
  return true;
}

// Facilitate access to 1-d backing data for a tensor so that values in a 1-d
// slice of the tensor can be accessed as if part of an ArrayView.
class StridedArrayViewBase {
 protected:
  StridedArrayViewBase(ArrayRef<int64_t> shape, ArrayRef<int64_t> index,
                       int64_t axis) {
    assert(shape.size() == index.size());
    assert(axis < shape.size());
    assert(axis >= 0);
    assert(index[axis] == 0);
    offset_ = IndexToOffset(shape, index);
    stride_ = StrideForAxis(shape, axis);
    size_ = shape[axis];
  }

  // Returns the size of the 1-d slice across the tensor.
  int64_t size() const { return size_; }

  // Calculates the next index in a tensor excluding a specified axis.
  //
  // Returns the next index where one exists.
  // If there is no valid next index, returns `std::nullopt`.
  //
  // `index` should have the same size as `shape`.
  // Each value `dim` in `index` should be in [0, shape[dim]).
  static llvm::Optional<SmallVector<int64_t>> NextTensorIndex(
      SmallVector<int64_t> index, ArrayRef<int64_t> shape, int64_t fixed_axis) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    assert(fixed_axis < shape.size());
    assert(fixed_axis >= 0);
    assert(index[fixed_axis] == 0);
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  // NDEBUG
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      if (dim == fixed_axis) continue;
      ++index[dim];
      if (index[dim] < shape[dim]) return std::move(index);
      index[dim] = 0;
    }
    return llvm::None;
  }

 protected:
  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access the value at a given index along a
  // StridedArrayView across a higher dimensional shape.
  //
  // The index `i` must be in [0, shape[axis])` where `shape` is the shape
  // of the tensor and `axis` is the axis along the tensor that the
  // StridedArrayView indexes along.
  int64_t OffsetForIndex(int64_t i) const { return offset_ + i * stride_; }

 private:
  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access the next value along a given axis.
  //
  // `axis` should be a valid dimension in `shape`.
  static int64_t StrideForAxis(ArrayRef<int64_t> shape, int64_t axis) {
    int64_t stride = 1;  // Start with the trailing dimension.
    for (int64_t dim = shape.size() - 1; dim > axis; --dim) {
      stride *= shape[dim];
    }
    return stride;
  }

  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access data at a specified index.
  //
  // `index` should have the same size as `shape`.
  // Each value `dim` in `index` should be in [0, shape[dim]).
  static int64_t IndexToOffset(ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> index) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  // NDEBUG
    int64_t offset = 0;
    int64_t stride = 1;
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      offset += index[dim] * stride;
      stride *= shape[dim];
    }
    return offset;
  }

  int64_t offset_;
  int64_t stride_;
  int64_t size_;
};

template <typename T>
class StridedArrayView;  // Class requires specialization.

// Wraps a DenseIntElementsAttr that holds backing data for a tensor so that
// int64_t values in a 1-d slice of the tensor can be accessed as if part of an
// ArrayView.
template <>
class StridedArrayView<DenseIntElementsAttr> : StridedArrayViewBase {
 public:
  StridedArrayView(const DenseIntElementsAttr &data, ArrayRef<int64_t> shape,
                   ArrayRef<int64_t> index, int64_t axis)
      : StridedArrayViewBase(shape, index, axis), data_(data) {
    int64_t element_count = 1;
    for (int64_t i = 0, e = shape.size(); i < e; ++i) {
      element_count *= shape[i];
    }
    assert(data.getNumElements() == element_count);
  }

  using StridedArrayViewBase::NextTensorIndex;
  using StridedArrayViewBase::size;

  int64_t operator[](int64_t i) const {
    return data_.getValues<APInt>()[OffsetForIndex(i)].getSExtValue();
  }

 private:
  const DenseIntElementsAttr &data_;
};

// Matches %iota generated from the following mlir codes (rank 2 example):
//
// %iota = mhlo.constant dense<[[0, 1, 2, ..., L],
//                              [0, 1, 2, ..., L]
//                              ...
//                              [0, 1, 2, ..., L]]>,
// where $dimensions is of size 1.
//
// StridedArrayViews are used to check the iota property across the constant
// data so that the iota dimension does not need to be the (inner) z-dimension.
bool MatchIotaConst(DenseIntElementsAttr dimensions, Value iota) {
  DenseIntElementsAttr iota_const_attr;
  if (!matchPattern(iota, m_Constant(&iota_const_attr))) return false;

  auto iota_type = iota_const_attr.getType();
  auto iota_shape = iota_type.getShape();
  auto reduce_dim = (*dimensions.value_begin<APInt>()).getSExtValue();
  if (reduce_dim < 0) reduce_dim += iota_type.getRank();

  auto index =
      llvm::Optional<SmallVector<int64_t>>(llvm::in_place, iota_type.getRank());
  while (index.has_value()) {
    StridedArrayView<DenseIntElementsAttr> array_view(
        iota_const_attr, iota_shape, *index, reduce_dim);
    for (int64_t i = 0; i < array_view.size(); ++i) {
      if (array_view[i] != i) return false;
    }
    index = StridedArrayView<DenseIntElementsAttr>::NextTensorIndex(
        std::move(*index), iota_shape, reduce_dim);
  }

  return true;
}

// The following 5 different forms of mhlo::iota will be matched:
// 1. IotaOp.
// 2. IotaOp + BroadCastInDim.
// 3. IotaOp + Reshape.
// 4. Constant (folded Iota) + BroadCastInDim.
// 5. Constant (folded result).
// Moreover, the dimensions has to match the iota_dimension.
bool MatchIota(DenseIntElementsAttr dimensions, Value iota) {
  return MatchSingleIota(dimensions, iota) ||
         MatchIotaBroadCastInDim(dimensions, iota) ||
         MatchReshapedIota(dimensions, iota) ||
         MatchConstIotaBroadCastInDim(dimensions, iota) ||
         MatchIotaConst(dimensions, iota);
}

bool MatchTopKComparator(Region &comparator) {
  if (!comparator.hasOneBlock()) return false;
  Block &comparator_blk = comparator.front();
  using OpListType = llvm::iplist<Operation>;
  OpListType &operations = comparator_blk.getOperations();
  if (operations.size() != 2) return false;
  auto compare_op = dyn_cast_or_null<mhlo::CompareOp>(&operations.front());
  auto return_op = dyn_cast_or_null<mhlo::ReturnOp>(&operations.back());
  if (!compare_op || !return_op) return false;
  // TODO(xuanyuanluo): Support mhlo::ComparisonDirection::LT direction.
  if (compare_op.comparison_direction() != mhlo::ComparisonDirection::GT)
    return false;
  if (compare_op.lhs() != comparator_blk.getArgument(0) ||
      compare_op.rhs() != comparator_blk.getArgument(1))
    return false;
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
      ConversionPatternRewriter &rewriter) const final {
    if (op->getOperands().size() != 2)
      return rewriter.notifyMatchFailure(
          op, "only match for the case where operands is of size 2");
    auto keys = op.operands()[0];
    auto indices = op.operands()[1];
    auto keys_ty = keys.getType().dyn_cast_or_null<ShapedType>();
    auto indices_ty = indices.getType().dyn_cast_or_null<ShapedType>();
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
    auto sort_dim = op.dimension();
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
    if (!MatchTopKComparator(op.comparator()))
      return rewriter.notifyMatchFailure(op, "only match for GT comparator");
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    Value k_cst = BuildIntConstOp(builder, rewriter, k, rewriter.getI32Type());
    rewriter.replaceOpWithNewOp<TopKV2Op>(op, keys.getType(), indices.getType(),
                                          keys, k_cst);
    return success();
  };
};

// A struct to hold information about dimensions of dot_general operands.
class DotDimensionsInfo {
 public:
  DotDimensionsInfo(ShapedType type, ArrayRef<int64_t> batch_dimensions,
                    ArrayRef<int64_t> contracting_dimensions) {
    const int rank = type.getRank();
    for (const int dim : batch_dimensions) {
      batch_dimensions_.axes.push_back(dim);
      batch_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (const int dim : contracting_dimensions) {
      contracting_dimensions_.axes.push_back(dim);
      contracting_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (int dim = 0; dim < rank; ++dim) {
      if (llvm::count(contracting_dimensions_.axes, dim) > 0 ||
          llvm::count(batch_dimensions_.axes, dim) > 0) {
        continue;
      }
      out_dimensions_.axes.push_back(dim);
      out_dimensions_.sizes.push_back(type.getDimSize(dim));
    }
  }

  const DimensionVector &batch_dimensions() const { return batch_dimensions_; }
  const DimensionVector &contracting_dimensions() const {
    return contracting_dimensions_;
  }
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  const DimensionVector &out_dimensions() const { return out_dimensions_; }

  // Returns the total dimension size after flattening all contracting
  // dimensions.
  int FlattenedContractingDimensionSize() const {
    return std::accumulate(contracting_dimensions_.sizes.begin(),
                           contracting_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

  // Returns the total dimension size after flattening all out dimensions.
  int FlattenedOutDimensionSize() const {
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

Value ConvertDot(PatternRewriter &rewriter, Value lhs, Value rhs,
                 DotDimensionNumbersAttr dot_dimension_numbers,
                 ShapedType result_type, mlir::Location loc) {
  auto lhs_type = lhs.getType().cast<ShapedType>();
  auto rhs_type = rhs.getType().cast<ShapedType>();
  const int lhs_rank = lhs_type.getRank();
  const int rhs_rank = rhs_type.getRank();

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
  auto lhs_flattend = rewriter.create<mhlo::ReshapeOp>(
      loc,
      RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
      lhs_transposed.getResult());

  // Reshapes rhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> rhs_flattened_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedContractingDimensionSize()},
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  auto rhs_flattend = rewriter.create<mhlo::ReshapeOp>(
      loc,
      RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
      rhs_transposed.getResult());

  // Creates matmul op of `lhs_flattend` and `rhs_flattend`.
  llvm::SmallVector<int64_t, 4> matmul_shape =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
                      llvm::ArrayRef<int64_t>{
                          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
                      llvm::ArrayRef<int64_t>{
                          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  auto matmul = rewriter.create<TF::BatchMatMulV3Op>(
      loc, RankedTensorType::get(matmul_shape, result_type.getElementType()),
      lhs_flattend.getResult(), rhs_flattend.getResult());
  auto reshaped =
      rewriter.create<mhlo::ReshapeOp>(loc, result_type, matmul.getResult());
  return reshaped.getResult();
}

// Converts mhlo.dot to tf.MatMul. Reshape ops will be inserted when
// necessary.
Value ConvertDotOp(PatternRewriter &rewriter, Operation *old_op) {
  auto dot_op = cast<mhlo::DotOp>(old_op);
  auto lhs_rank = dot_op.lhs().getType().cast<ShapedType>().getRank();
  auto dot_dimension_numbers =
      DotDimensionNumbersAttr::get(rewriter.getContext(),
                                   /*lhs_batching_dimensions=*/{},
                                   /*rhs_batching_dimensions=*/{},
                                   /*lhs_contracting_dimensions=*/
                                   {lhs_rank == 1 ? 0 : 1},
                                   /*rhs_contracting_dimensions=*/{0});
  return ConvertDot(rewriter, dot_op.lhs(), dot_op.rhs(), dot_dimension_numbers,
                    dot_op.getResult().getType().cast<ShapedType>(),
                    dot_op.getLoc());
}

// Converts mhlo.dot to tf.BatchMatMul. Reshape or Transpose ops will also be
// inserted to convert to well-formed matrix multiply.
Value ConvertDotGeneralOp(PatternRewriter &rewriter, Operation *old_op) {
  auto dot_general_op = cast<mhlo::DotGeneralOp>(old_op);
  return ConvertDot(rewriter, dot_general_op.lhs(), dot_general_op.rhs(),
                    dot_general_op.dot_dimension_numbers(),
                    dot_general_op.getResult().getType().cast<ShapedType>(),
                    dot_general_op.getLoc());
}

// Checks if the specified region is a binary reduction function that takes 2
// inputs, passes it to an instance of the specifiied reduction op and then
// returns the result.
template <typename ReductionOp>
LogicalResult MatchBinaryReduceFunction(mlir::Region &function) {
  Block &body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();

  ReductionOp reduce_op = dyn_cast_or_null<ReductionOp>(
      return_op.getOperands().front().getDefiningOp());
  if (!reduce_op) return failure();
  if (reduce_op.lhs() != body.getArgument(0) ||
      reduce_op.rhs() != body.getArgument(1))
    return failure();

  return success();
}

// Check if the specified region is a binary reduction function that takes 2
// inputs and returns the second input. Functions like this are used by update
// scatter like ops.
template <>
LogicalResult MatchBinaryReduceFunction<void>(mlir::Region &function) {
  Block &body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();
  if (return_op.getOperands().front() != body.getArgument(1)) return failure();
  return success();
}

// Replace BinaryOp with a combination of TfBinaryOp and TfReduceOp if the
// init value doesn't match the expection of TfReduceOp.
template <typename TfReduceOp, typename TfBinOp>
LogicalResult rewriteNonMatchInitValue(mhlo::ReduceOp reduce_op, Value input,
                                       ConstOp reduction_indices,
                                       ConversionPatternRewriter &rewriter) {
  Value reduce_result = rewriter.create<TfReduceOp>(
      reduce_op.getLoc(), reduce_op.getType(0), input, reduction_indices,
      /*keep_dim=*/rewriter.getBoolAttr(false));
  rewriter.replaceOpWithNewOp<TfBinOp>(reduce_op, reduce_op.getType(0),
                                       reduce_result,
                                       reduce_op.init_values()[0]);
  return success();
}

// Cannot replace BinaryOp if the init value doesn't match the expection of
// TfReduceOp and there is no corresponding TfBinaryOp.
template <>
LogicalResult rewriteNonMatchInitValue<TF::MaxOp, void>(
    mhlo::ReduceOp reduce_op, Value input, ConstOp reduction_indices,
    ConversionPatternRewriter &rewriter) {
  return failure();
}

template <>
LogicalResult rewriteNonMatchInitValue<TF::MinOp, void>(
    mhlo::ReduceOp reduce_op, Value input, ConstOp reduction_indices,
    ConversionPatternRewriter &rewriter) {
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
      ConversionPatternRewriter &rewriter) const final {
    if (failed(MatchReduceOpOperand(reduce_op))) return failure();

    if (failed(MatchBinaryReduceFunction<BinaryOp>(reduce_op.body())))
      return failure();

    auto operand = reduce_op.operands()[0];

    // Get reduction dimension.
    DenseIntElementsAttr dimension = reduce_op.dimensions();
    SmallVector<int64_t, 4> reduce_dims;
    for (const int64_t &dim : dimension.getValues<int64_t>()) {
      reduce_dims.emplace_back(dim);
    }
    auto dim_type = RankedTensorType::get(
        {static_cast<int64_t>(reduce_dims.size())}, rewriter.getI64Type());
    auto reduction_indices = rewriter.create<ConstOp>(
        reduce_op.getLoc(), dim_type, rewriter.getI64TensorAttr(reduce_dims));

    // In `MatchReduceOpOperand` function, we already match that the
    // "mhlo::ReduceOp" only has one operand, one init_value and one result.

    // If the init value matches with the init value expected for the target
    // TfReduceOp, then replace the BinaryOp with a TfReduceOp. Otherwise,
    // replace the BinaryOp with a TfBinaryOp and a TfReduceOp.
    if (succeeded(MatchInitValue(reduce_op.init_values()[0]))) {
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
    if (reduce_op.operands().size() != 1 ||
        reduce_op.init_values().size() != 1 ||
        reduce_op.getResults().size() != 1)
      return failure();

    if (!reduce_op.operands()[0].getType().isa<RankedTensorType>())
      return failure();
    if (!reduce_op.getType(0).isa<RankedTensorType>()) return failure();
    return success();
  }
};

class ConvertReduceOpToTfSum
    : public ConvertReduceOpToTfOp<mhlo::AddOp, TF::SumOp, TF::AddOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = init_value.getType().cast<ShapedType>().getElementType();
    if (type.isa<FloatType>()) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else if (type.isa<IntegerType>() && type.isSignlessInteger()) {
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
    auto type = init_value.getType().cast<ShapedType>().getElementType();
    if (type.isa<FloatType>()) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || !const_value.isNegative())
        return failure();
    } else if (type.isa<IntegerType>() && type.isSignlessInteger()) {
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
    auto type = init_value.getType().cast<ShapedType>().getElementType();

    if (type.isa<FloatType>()) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || const_value.isNegative())
        return failure();
    } else if (type.isa<IntegerType>() && type.isSignlessInteger()) {
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

template <typename TfReduce, typename TfArgReduce>
class ConvertReduceOpToTfArgMinMax
    : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (reduce_op.operands().size() != 2) return failure();
    if (reduce_op.dimensions().getNumElements() != 1) return failure();

    // Check that the operand init is the expected value.
    DenseElementsAttr operand_init;
    if (!matchPattern(reduce_op.init_values().front(),
                      m_Constant(&operand_init)))
      return failure();
    if (!IsValueInitValue(operand_init)) return failure();

    // Check that the iota init is zero.
    DenseElementsAttr iota_init;
    if (!matchPattern(reduce_op.init_values().back(), m_Constant(&iota_init)))
      return failure();
    if (iota_init.getValues<APInt>()[0] != 0) return failure();

    // Verify that the second argument is an Iota op along the same dimension
    // as the reduction.
    Value iota = reduce_op.operands().back();
    if (!MatchIota(reduce_op.dimensions(), iota)) return failure();

    // Match the reduction computation.
    const bool is_float = operand_init.getElementType().isa<FloatType>();
    if (failed(matchReduceComputation(reduce_op.body(), is_float)))
      return failure();

    Value operand = reduce_op.operands().front();
    int64_t axis = reduce_op.dimensions().getValues<int64_t>()[0];

    auto dim_type = RankedTensorType::get({1}, rewriter.getI64Type());
    auto reduction_indices = rewriter.create<ConstOp>(
        reduce_op.getLoc(), dim_type, rewriter.getI64TensorAttr({axis}));

    // Generate a Max and an ArgMax of as the mhlo op returns both while in TF
    // we have separate ops for them. If only one of them is used then the other
    // one will be garbage collected later.
    auto tf_reduce_op = rewriter.create<TfReduce>(
        reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
        reduction_indices,
        /*keep_dim=*/rewriter.getBoolAttr(false));
    auto tf_argreduce_op = rewriter.create<TfArgReduce>(
        reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
        reduction_indices);

    rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
    return success();
  }

  // Pattern matches the following reduction function for ArgMax/ArgMin:
  // %0 = compare{GT}(%lhs_value, %rhs_value)
  // %1 = compare{NE}(%lhs_value, %lhs_value)
  // %2 = or(%0, %1)
  // %3 = select(%2, %lhs_value, %rhs_value)
  // %4 = compare{EQ}(%lhs_value, %rhs_value)
  // %5 = compare{LT}(%lhs_index, %rhs_index)
  // %6 = and(%4, %5)
  // %7 = or(%2, %6)
  // %8 = select(%7, %lhs_index, %rhs_index)
  // return %3, %8
  // Also note that %1 may be folded if %lhs_value is of integer types.
  LogicalResult matchReduceComputation(Region &computation,
                                       bool is_float) const {
    Block &body = computation.front();
    if (body.getNumArguments() != 4) return failure();

    mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
    if (!return_op || return_op.getNumOperands() != 2) return failure();

    mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
        return_op.getOperand(0).getDefiningOp());
    if (!value_select || value_select.on_true() != body.getArgument(0) ||
        value_select.on_false() != body.getArgument(2))
      return failure();

    if (is_float) {
      mhlo::OrOp value_or = llvm::dyn_cast_or_null<mhlo::OrOp>(
          value_select.getOperand(0).getDefiningOp());
      if (!value_or) return failure();

      mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
          value_or.lhs().getDefiningOp());
      if (!value_gt || value_gt.comparison_direction() != CompareDirection() ||
          value_gt.lhs() != body.getArgument(0) ||
          value_gt.rhs() != body.getArgument(2))
        return failure();

      mhlo::CompareOp value_ne = llvm::dyn_cast_or_null<mhlo::CompareOp>(
          value_or.rhs().getDefiningOp());
      if (!value_ne ||
          value_ne.comparison_direction() != mhlo::ComparisonDirection::NE ||
          value_ne.lhs() != body.getArgument(0) ||
          value_ne.rhs() != body.getArgument(0))
        return failure();
    } else {
      mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
          value_select.getOperand(0).getDefiningOp());
      if (!value_gt || value_gt.comparison_direction() != CompareDirection() ||
          value_gt.lhs() != body.getArgument(0) ||
          value_gt.rhs() != body.getArgument(2))
        return failure();
    }

    mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
        return_op.getOperand(1).getDefiningOp());
    if (!index_select || index_select.on_true() != body.getArgument(1) ||
        index_select.on_false() != body.getArgument(3))
      return failure();

    mhlo::OrOp index_or =
        llvm::dyn_cast_or_null<mhlo::OrOp>(index_select.pred().getDefiningOp());

    if (!index_or || index_or.lhs() != value_select.pred()) return failure();

    mhlo::AndOp index_and =
        llvm::dyn_cast_or_null<mhlo::AndOp>(index_or.rhs().getDefiningOp());
    if (!index_and) return failure();

    mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        index_and.lhs().getDefiningOp());
    if (!value_eq ||
        value_eq.comparison_direction() != mhlo::ComparisonDirection::EQ ||
        value_eq.lhs() != body.getArgument(0) ||
        value_eq.rhs() != body.getArgument(2))
      return failure();

    mhlo::CompareOp index_lt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        index_and.rhs().getDefiningOp());
    if (!index_lt ||
        index_lt.comparison_direction() != mhlo::ComparisonDirection::LT ||
        index_lt.lhs() != body.getArgument(1) ||
        index_lt.rhs() != body.getArgument(3))
      return failure();

    return success();
  }

  virtual mhlo::ComparisonDirection CompareDirection() const = 0;

  virtual bool IsValueInitValue(const DenseElementsAttr &attr) const = 0;
};

class ConvertReduceOpToTfArgmax
    : public ConvertReduceOpToTfArgMinMax<TF::MaxOp, TF::ArgMaxOp> {
 public:
  using ConvertReduceOpToTfArgMinMax::ConvertReduceOpToTfArgMinMax;

  mhlo::ComparisonDirection CompareDirection() const override {
    return mhlo::ComparisonDirection::GT;
  }
  bool IsValueInitValue(const DenseElementsAttr &attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat() ||
        element_type.isInteger(1))
      return false;
    if (element_type.isa<FloatType>()) {
      auto value = *attr.value_begin<APFloat>();
      return value.isNegative() && value.isInfinity();
    } else {
      auto value = *attr.value_begin<APInt>();
      return element_type.isUnsignedInteger() ? value.isMinValue()
                                              : value.isMinSignedValue();
    }
  }
};

class ConvertReduceOpToTfArgmin
    : public ConvertReduceOpToTfArgMinMax<TF::MinOp, TF::ArgMinOp> {
 public:
  using ConvertReduceOpToTfArgMinMax::ConvertReduceOpToTfArgMinMax;

  mhlo::ComparisonDirection CompareDirection() const override {
    return mhlo::ComparisonDirection::LT;
  }
  bool IsValueInitValue(const DenseElementsAttr &attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat() ||
        element_type.isInteger(1))
      return false;
    if (element_type.isa<FloatType>()) {
      auto value = *attr.value_begin<APFloat>();
      return !value.isNegative() && value.isInfinity();
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
      ConversionPatternRewriter &rewriter) const final {
    RankedTensorType type =
        iota_op.getType().dyn_cast_or_null<RankedTensorType>();
    // TF::RangeOp doesn't support UI16.
    if (!type || type.getElementType().isUnsignedInteger(16)) return failure();

    const uint64_t dimension = iota_op.iota_dimension();
    Type element_type = type.getElementType();
    Attribute start, limit, delta;
    if (element_type.isa<FloatType>()) {
      start = rewriter.getFloatAttr(element_type, 0.0);
      limit = rewriter.getFloatAttr(element_type, type.getShape()[dimension]);
      delta = rewriter.getFloatAttr(element_type, 1.0);
    } else if (element_type.isa<IntegerType>()) {
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
      reshape_shape[iota_op.iota_dimension()] = type.getShape()[dimension];
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

// A helper function for ConvertMaxPoolOp and ConvertAvgMaxPoolOp. Returns true
// if the given ReduceWindowOp is a spatial pooling without dilation. If returns
// true, also outputs the window strides and the TF padding mode ("VALID" or
// "SAME").
bool IsSpatialPoolingWithoutDilation(
    mhlo::ReduceWindowOp rw, llvm::SmallVectorImpl<int64_t> *window_strides,
    std::string *padding_mode) {
  // tf.max_pool or tf.avg_pool need at least 3 dimensions (batch, spatial,
  // channel).
  const uint64_t rank = rw.window_dimensions().size();
  if (rank <= 2) return false;

  if (rw.window_strides().has_value()) {
    window_strides->insert(window_strides->end(),
                           rw.window_strides()->getValues<int64_t>().begin(),
                           rw.window_strides()->getValues<int64_t>().end());
  } else {
    window_strides->resize(rank, 1);
  }

  llvm::SmallVector<int64_t, 10> padding;
  if (rw.padding().has_value()) {
    padding.insert(padding.begin(), rw.padding()->getValues<int64_t>().begin(),
                   rw.padding()->getValues<int64_t>().end());
  } else {
    padding.resize(2 * rank, 0);
  }

  // Check that we don't do any reduction along the batch (first) and channel
  // (last) dimensions.
  const uint64_t batch_dim = 0;
  const uint64_t channel_dim = rank - 1;
  if (rw.window_dimensions().getValues<int64_t>()[batch_dim] != 1 ||
      rw.window_dimensions().getValues<int64_t>()[channel_dim] != 1 ||
      (*window_strides)[batch_dim] != 1 ||
      (*window_strides)[channel_dim] != 1 || padding[2 * batch_dim] != 0 ||
      padding[2 * batch_dim + 1] != 0 || padding[2 * channel_dim] != 0 ||
      padding[2 * channel_dim + 1] != 0)
    return false;

  if (rw.window_dilations().has_value() &&
      !(rw.window_dilations()->isSplat() &&
        rw.window_dilations()->getSplatValue<APInt>() == 1))
    return false;

  if (rw.base_dilations().has_value() &&
      !(rw.base_dilations()->isSplat() &&
        rw.base_dilations()->getSplatValue<APInt>() == 1))
    return false;

  if (llvm::all_of(padding, [](int64_t i) { return i == 0; })) {
    *padding_mode = "VALID";
    return true;
  }

  // Check that the individual padding values are corresponding to SAME
  // padding from TensorFlow.
  auto operand_type = rw.operands()[0].getType().dyn_cast<RankedTensorType>();
  RankedTensorType output_type =
      rw.getResult(0).getType().dyn_cast<RankedTensorType>();
  if (!operand_type || !output_type) return false;

  for (uint64_t i = 1; i < rank - 1; ++i) {
    int64_t padding_size =
        (output_type.getShape()[i] - 1) * (*window_strides)[i] +
        rw.window_dimensions().getValues<int64_t>()[i] -
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

  virtual bool IsInitValue(const DenseElementsAttr &attr) const = 0;

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp rw, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (rw.getNumResults() != 1 || rw.operands().size() != 1 ||
        rw.init_values().size() != 1)
      return failure();

    if (failed(MatchBinaryReduceFunction<BinaryOp>(rw.body())))
      return failure();

    // Ensure that initial_values are as expected.
    auto const_op = llvm::dyn_cast_or_null<mhlo::ConstantOp>(
        rw.init_values()[0].getDefiningOp());
    if (!const_op) return failure();
    auto const_op_dense_value = const_op.value().cast<DenseElementsAttr>();
    if (!const_op_dense_value || !IsInitValue(const_op_dense_value)) {
      return failure();
    }

    auto operand_type = rw.operands()[0].getType().cast<ShapedType>();

    // For a cumulative op, require a tensor of 1s for each dimension in
    // operand.
    auto is_splat_int64_ones =
        [&rewriter,
         &operand_type](const ::llvm::Optional<DenseIntElementsAttr> &attr) {
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
    if (!is_splat_int64_ones(rw.base_dilations()) ||
        !is_splat_int64_ones(rw.window_dilations()) ||
        !is_splat_int64_ones(rw.window_strides()))
      return failure();

    // Determine which axis is being used for the cumulative operation.
    //
    // For a cumulative op, window_dimensions should be of the form:
    //  dense<[1, 1, N, 1]>
    // where N is the same as the size of the corresponding input dimension
    // and there is a 1-entry for each input dimension not being operated
    // over.
    const auto &window_dimensions = rw.window_dimensions();
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
      return rewriter.notifyMatchFailure(rw, "no reduced dimension is found.");
    }

    // For a cumulative op, padding (expressed as a list of left-padding and
    // right-padding pairs) should be of the form:
    //  dense<[[0, 0], [0, 0], [N-1, 0], [0, 0]]>
    // where N is the size of the input dimension being operated over.
    if (!rw.padding()) return failure();
    const auto &padding = rw.padding()->getValues<int64_t>();
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

    rewriter.replaceOpWithNewOp<TfCumOp>(rw, rw.getType(0), rw.operands()[0],
                                         axis, /* exclusive */ false,
                                         /* reverse */ false);
    return success();
  }
};

class ConvertLoweredCumSumOp
    : public ConvertLoweredCumOp<mhlo::AddOp, TF::CumsumOp> {
  using ConvertLoweredCumOp::ConvertLoweredCumOp;
  bool IsInitValue(const DenseElementsAttr &attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (element_type.isa<FloatType>()) {
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
  bool IsInitValue(const DenseElementsAttr &attr) const override {
    auto element_type = attr.getType().getElementType();
    if (attr.getNumElements() != 1 || !element_type.isIntOrFloat())
      return false;
    if (element_type.isa<FloatType>()) {
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
  explicit ConvertAvgPoolOp(MLIRContext *context)
      : OpConversionPattern(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(
      mhlo::DivOp div_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto rw =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(div_op.lhs().getDefiningOp());
    if (!rw || rw->getNumResults() != 1) return failure();

    // Check that the reduce-window is a sum-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw.body())))
      return failure();

    // Check that this is a floating point reduce window with a rank of 4 or 5.
    const RankedTensorType rw_type =
        rw.getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!rw_type || !rw_type.getElementType().isa<FloatType>() ||
        rw_type.getRank() <= 3 || rw_type.getRank() > 5)
      return failure();

    // Check that the Div op doesn't do broadcasting on the output of the reduce
    // window.
    if (div_op.getType() != rw_type) return failure();

    // If the init value isn't zero then it can't be an average pool.
    if (!isFloatZero(rw.init_values()[0])) return failure();

    llvm::SmallVector<int64_t, 5> window_strides;
    std::string padding_mode;
    if (!IsSpatialPoolingWithoutDilation(rw, &window_strides, &padding_mode)) {
      return rewriter.notifyMatchFailure(
          div_op, "not the root of spatial pooling without dilation");
    }

    DenseFPElementsAttr divisor;
    if (matchPattern(div_op.rhs(), m_Constant(&divisor))) {
      // If the divisor is a constant then check that it matches with the number
      // of elements inside the window what is required for a VALID AvgPool.
      if (!divisor.isSplat()) return failure();
      int64_t window_size = 1;
      for (int64_t w : rw.window_dimensions().getValues<int64_t>()) {
        window_size *= w;
      }
      if (!divisor.getSplatValue<APFloat>().isExactlyValue(window_size))
        return failure();

      if (padding_mode != "VALID") {
        return failure();
      }

      return replaceWithAvgPool(
          div_op, rw.operands()[0],
          llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
          window_strides, "VALID", rewriter);
    }

    auto rw_rhs =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(div_op.rhs().getDefiningOp());
    if (rw_rhs && rw_rhs.getNumResults() == 1) {
      // Check that RHS is a sum-reduce-window.
      if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw_rhs.body())))
        return failure();

      // Check that the RHS is a reduce_window over a constant 1 operand with 0
      // as the init value.
      DenseFPElementsAttr rhs_operand;
      if (!isFloatZero(rw_rhs.init_values()[0]) ||
          !matchPattern(rw_rhs.operands()[0], m_Constant(&rhs_operand)) ||
          !rhs_operand.isSplat() ||
          !rhs_operand.getSplatValue<APFloat>().isExactlyValue(1.0))
        return failure();

      // Check that the two reduce window have the same window configuration.
      if (rw.window_dimensions() != rw_rhs.window_dimensions() ||
          rw.window_strides() != rw_rhs.window_strides() ||
          rw.window_dilations() != rw_rhs.window_dilations() ||
          rw.base_dilations() != rw_rhs.base_dilations() ||
          rw.padding() != rw_rhs.padding())
        return failure();

      return replaceWithAvgPool(
          div_op, rw.operands()[0],
          llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
          window_strides, padding_mode, rewriter);
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
                                   ConversionPatternRewriter &rewriter) const {
    if (ksizes.size() == 4) {
      rewriter.replaceOpWithNewOp<AvgPoolOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr("NHWC"));
      return success();
    } else if (ksizes.size() == 5) {
      rewriter.replaceOpWithNewOp<AvgPool3DOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr("NDHWC"));
      return success();
    }
    return failure();
  }
};

class ConvertMaxPoolOp : public OpConversionPattern<mhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp rw, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Check that the reduce-window is a max-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::MaxOp>(rw.body())))
      return failure();

    // Check that this is a floating point reduce window with a rank of 4 or 5.
    const RankedTensorType rw_type =
        rw.getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!rw_type || !rw_type.getElementType().isa<FloatType>() ||
        rw_type.getRank() <= 3 || rw_type.getRank() > 5)
      return failure();

    if (!isFloatMinusInfinity(rw.init_values()[0])) {
      return failure();
    }

    llvm::SmallVector<int64_t, 5> window_strides;
    std::string padding_mode;
    if (!IsSpatialPoolingWithoutDilation(rw, &window_strides, &padding_mode)) {
      return rewriter.notifyMatchFailure(
          rw, "not the root of spatial pooling without dilation");
    }

    return replaceWithMaxPool(
        rw, rw.operands()[0],
        llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
        window_strides, padding_mode, rewriter);
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
                                   ConversionPatternRewriter &rewriter) const {
    if (ksizes.size() == 4) {
      rewriter.replaceOpWithNewOp<MaxPoolOp>(
          op, op.getType(0), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          rewriter.getStringAttr("NHWC"));
      return success();
    } else if (ksizes.size() == 5) {
      rewriter.replaceOpWithNewOp<MaxPool3DOp>(
          op, op.getType(0), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr("NDHWC"));
      return success();
    }
    return failure();
  }
};

class LegalizeHloToTf : public TF::LegalizeHloToTfPassBase<LegalizeHloToTf> {
  /// Performs the legalization to the TF dialect.
  void runOnOperation() override;
};

// Returns the shape of the given value in a Constant Op.
arith::ConstantOp ShapeToConst(PatternRewriter &rewriter, Value value) {
  ArrayRef<int64_t> shape = value.getType().cast<ShapedType>().getShape();
  auto attr_type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                         rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, shape);
  return rewriter.create<arith::ConstantOp>(value.getLoc(), attr_type, attr);
}

bool IsSign(APFloat a, APFloat sign) {
  if (a.isNaN() || a.isZero()) return a == sign;
  if (a.isNegative()) return sign.isExactlyValue(-1.0);
  return sign.isExactlyValue(1.0);
}

// Returns whether the splat constant is the sign of the FloatTensor
bool FloatTensorIsSign(PatternRewriter &rewriter, ElementsAttr floatv,
                       ElementsAttr sgn_cst) {
  if (!sgn_cst.isa<SplatElementsAttr>()) return false;
  auto sgn_cst_spl = sgn_cst.cast<SplatElementsAttr>().getSplatValue<APFloat>();
  if (floatv.isa<SplatElementsAttr>()) {
    auto floatv_spl = floatv.cast<SplatElementsAttr>().getSplatValue<APFloat>();
    return IsSign(floatv_spl, sgn_cst_spl);
  } else if (floatv.isa<DenseElementsAttr>()) {
    auto floatv_dns = floatv.cast<DenseFPElementsAttr>();
    return llvm::all_of(floatv_dns.getValues<APFloat>(), [&](APFloat value) {
      return IsSign(value, sgn_cst_spl);
    });
  }
  return false;
}

// Check that `arr` is an R1 iota with integer element type starting from `0`
// with `size` number of values.
bool IsIotaAttr(ArrayRef<int64_t> arr, int64_t size) {
  if (arr.size() != size) return false;
  int64_t iota = 0;
  for (auto s : arr) {
    if (s != iota) return false;
    ++iota;
  }
  return true;
}

// Convert updates into canonical form as expected by tf.scatter ops.
//
// tf.scatter expects `update_window_dims` to be the trailing dimensions.
//
// To support scatter ops generated by numpy-like slice updates:
//   nd_array[:, [i,j]] = [i_values, j_values]
//
// `updates` must be transposed when the update_window_dims are the leading
// dimensions of `updates`.
//
// Other values of `update_window_dims` are left unsupported.
//
// Eg 1. An update in canonical form:
//  * indices shape(A,B,C)
//  * updates shape(A,B,D,E,F)
// Then:
//  * D,E,F are the update window dims [2,3,4]
//  * C is the index vector dimension
//  * A,B iterate over the updates and indices
//
// If `update_window_dims` are not the trailing dimensions then updates must be
// transposed.
//
// Eg 2. An update in non-canonical form:
//  * indices shape(a,b,c)
//  * updates shape(d,e,f,a,b)
// Then:
//  * d,e,f are the update window dims [0,1,2]
//  * c is the index vector dimension
//  * a,b iterate over the updates and indices
//
//  The update needs permuting to be in the form (a,b,d,e,f) so that the update
//  window dims are the trailing dimensions.
//
// To canonicalize the updates above, replace the updates with:
//   transpose(updates, permutation={3,4,0,1,2})
//
// Note: NormalizeIndexVector is assumed to have run on the indices already so
// that the index_vector_dim is the trailing dimension in `indices`.
LogicalResult CanonicalizeScatterUpdates(
    Operation *scatter_op, llvm::ArrayRef<int64_t> update_window_dims,
    const Value &indices, const ShapedType &indices_type, Value &updates,
    ShapedType &updates_type, ConversionPatternRewriter &rewriter) {
  auto canonical_update_window_dims = llvm::to_vector(
      llvm::seq<int64_t>(indices_type.getRank() - 1, updates_type.getRank()));

  if (canonical_update_window_dims == update_window_dims) return success();

  // Permute updates if `update_window_dims` are leading indices.
  // Other possibilities for `update_window_dims` are not supported yet.
  if (!IsIotaAttr(update_window_dims, update_window_dims.size()))
    return rewriter.notifyMatchFailure(
        scatter_op, "update_window_dims are not leading or trailing indices");

  SmallVector<int64_t, 4> permutation_array(updates_type.getRank());
  int64_t dim = 0;
  // Move leading indices to the back of the array.
  const auto permutation_array_size = permutation_array.size();
  for (int64_t i = update_window_dims.size(); i < permutation_array_size; ++i) {
    permutation_array[i] = dim;
    ++dim;
  }
  // Move trailing indices to the front of the array.
  for (int64_t i = 0; i < update_window_dims.size(); ++i) {
    permutation_array[i] = dim;
    ++dim;
  }

  auto permutation_and_shape = GetPermutationAndTransposedShape(
      permutation_array, updates_type, rewriter);

  auto transposed_updates = rewriter.create<mhlo::TransposeOp>(
      scatter_op->getLoc(), permutation_and_shape.shape, updates,
      permutation_and_shape.permutation);

  updates = transposed_updates;
  updates_type = permutation_and_shape.shape;
  return success();
}

// If index_vector_dim == indices.rank() then insert the implicit extra
// dimension into indices to normalize everything to index_vector_dim ==
// indices.rank() - 1.
LogicalResult NormalizeIndexVector(Operation *parent_op, Value &indices,
                                   ShapedType &indices_type,
                                   int64_t index_vector_dim,
                                   ConversionPatternRewriter &rewriter) {
  if (index_vector_dim == indices_type.getRank()) {
    llvm::SmallVector<int64_t, 4> new_start_indices_shape(
        indices_type.getShape().begin(), indices_type.getShape().end());
    new_start_indices_shape.push_back(1);
    indices_type = RankedTensorType::get(new_start_indices_shape,
                                         indices_type.getElementType());
    indices = rewriter.create<mhlo::ReshapeOp>(parent_op->getLoc(),
                                               indices_type, indices);
  } else if (index_vector_dim != indices_type.getRank() - 1) {
    // If index_vector_dim isn't the last dimension in indices then it isn't
    // supported yet.
    // TODO(tberghammer): Transpose indices to support this usecase.
    return rewriter.notifyMatchFailure(
        parent_op,
        "index vector dim isn't the last dimension in start indices");
  }
  return success();
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
      ConversionPatternRewriter &rewriter) const final {
    Value operand = gather_op.operand();
    Value start_indices = gather_op.start_indices();

    // Can only convert with static shaped gather.
    ShapedType operand_type = operand.getType().cast<ShapedType>();
    ShapedType start_indices_type = start_indices.getType().cast<ShapedType>();
    ShapedType result_type = gather_op.getResult().getType().cast<ShapedType>();
    if (!operand_type.hasStaticShape() ||
        !start_indices_type.hasStaticShape() || !result_type.hasStaticShape()) {
      return failure();
    }

    // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
    int64_t index_vector_dim =
        gather_op.dimension_numbers().getIndexVectorDim();
    if (failed(NormalizeIndexVector(gather_op, start_indices,
                                    start_indices_type, index_vector_dim,
                                    rewriter))) {
      return failure();
    }

    // Verify that start_index_map and collapsed_slice_dims contains the same
    // values.
    auto start_index_map = gather_op.dimension_numbers().getStartIndexMap();
    auto collapsed_slice_dims =
        gather_op.dimension_numbers().getCollapsedSliceDims();
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
    auto slice_sizes = gather_op.slice_sizes();
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
    auto offset_dims = gather_op.dimension_numbers().getOffsetDims();
    SmallVector<int64_t, 4> offset_dims_vector(offset_dims.begin(),
                                               offset_dims.end());
    const TransposeParams &transpose_params =
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
    auto tf_gather_nd_op = rewriter.create<TF::GatherNdOp>(
        gather_op->getLoc(), tf_gather_nd_result_type, operand, start_indices);
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
      ConversionPatternRewriter &rewriter) const final {
    // HLO WhileOp should have two regions: cond and body.
    if (while_op->getNumRegions() != 2) return failure();

    // This rule doesn't support mhlo::WhileOp with tuple inputs.
    for (auto type : while_op->getOperandTypes()) {
      if (type.isa<TupleType>()) return failure();
    }

    // Creates a TF::WhileRegionOp to replace the mhlo::WhileOp. HLO WhileOp
    // currently doesn't support stateless and shape invariant, so these
    // parameters are set to the default values.
    auto new_while = rewriter.create<TF::WhileRegionOp>(
        while_op.getLoc(), while_op->getResultTypes(), while_op->getOperands(),
        /*parallel_iterations=*/10,
        /*is_stateless=*/false, /*shape_invariant=*/false);
    new_while.cond().takeBody(while_op.cond());
    new_while.body().takeBody(while_op.body());
    ReplaceReturnOp(new_while.cond(), rewriter);
    ReplaceReturnOp(new_while.body(), rewriter);
    rewriter.replaceOp(while_op, new_while.getResults());
    return success();
  }
};

class ConvertIfOp : public OpConversionPattern<mhlo::IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // HLO IfOp currently doesn't support stateless
    auto new_op = rewriter.create<TF::IfRegionOp>(
        op.getLoc(), op->getResultTypes(), op.pred(),
        /*is_stateless=*/false, /*_then_func_name=*/nullptr,
        /*_else_func_name=*/nullptr);
    new_op.then_branch().takeBody(op.true_branch());
    new_op.else_branch().takeBody(op.false_branch());
    ReplaceReturnOp(new_op.then_branch(), rewriter);
    ReplaceReturnOp(new_op.else_branch(), rewriter);
    rewriter.replaceOp(op, new_op.getResults());
    return success();
  }
};

template <typename BinaryOp, typename TfOp>
class ConvertScatterOp : public OpConversionPattern<mhlo::ScatterOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp scatter_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    OperandRange operands = scatter_op.operands();
    Value indices = scatter_op.scatter_indices();
    OperandRange updates = scatter_op.updates();
    if (operands.size() != 1 || updates.size() != 1) return failure();

    ShapedType operand_type = operands[0].getType().cast<ShapedType>();
    ShapedType indices_type = indices.getType().cast<ShapedType>();
    ShapedType updates_type = updates[0].getType().cast<ShapedType>();

    Value new_updates = updates[0];

    // Can only convert with static shaped scatter.
    if (!operand_type.hasStaticShape() || !indices_type.hasStaticShape() ||
        !updates_type.hasStaticShape()) {
      return failure();
    }

    // Match the scatter computation against computations supported by TF.
    if (failed(MatchBinaryReduceFunction<BinaryOp>(
            scatter_op.update_computation()))) {
      return failure();
    }

    auto scatter_dimension_numbers = scatter_op.scatter_dimension_numbers();

    // Normalize indices so index_vector_dim == indices.rank() - 1.
    int64_t index_vector_dim = scatter_dimension_numbers.getIndexVectorDim();
    if (failed(NormalizeIndexVector(scatter_op, indices, indices_type,
                                    index_vector_dim, rewriter))) {
      return failure();
    }

    // Transform updates so that update window dims are the trailing dimensions
    // in the update tensor.
    auto update_window_dims = scatter_dimension_numbers.getUpdateWindowDims();
    if (failed(CanonicalizeScatterUpdates(scatter_op, update_window_dims,
                                          indices, indices_type, new_updates,
                                          updates_type, rewriter))) {
      return failure();
    }

    auto inserted_window_dims =
        scatter_dimension_numbers.getInsertedWindowDims();
    auto scatter_dims_to_operand_dims =
        scatter_dimension_numbers.getScatterDimsToOperandDims();

    if (IsIotaAttr(inserted_window_dims, indices_type.getShape().back()) &&
        IsIotaAttr(scatter_dims_to_operand_dims,
                   indices_type.getShape().back())) {
      rewriter.replaceOpWithNewOp<TfOp>(scatter_op,
                                        scatter_op.getResult(0).getType(),
                                        operands[0], indices, new_updates);
      return success();
    }
    // Insert tranposes to support scatter operations generated from
    // numpy-like slice operations:
    //   nd_array[:, [i,j]] = [i_values, j_values]
    //
    if (scatter_dims_to_operand_dims != inserted_window_dims) {
      // Support only dimension numbers generated by numpy-like slice
      // operations.
      return rewriter.notifyMatchFailure(
          scatter_op, "unsupported scatter_dims_to_operand_dims");
    }

    // Transpose the operand and so that the trailing dimensions of the
    // operand are being updated. Then apply a tf.scatter op and transpose
    // back the result to get the same shape as the original operand.

    SmallVector<int64_t, 4> permutation_array;
    for (int64_t i = 0; i < scatter_dims_to_operand_dims.size(); ++i) {
      permutation_array.push_back(scatter_dims_to_operand_dims[i]);
    }
    for (int64_t i = 0; i < operand_type.getRank(); ++i) {
      if (!llvm::is_contained(scatter_dims_to_operand_dims, i)) {
        permutation_array.push_back(i);
      }
    }
    auto permutation_and_shape = GetPermutationAndTransposedShape(
        permutation_array, operand_type, rewriter);

    Location loc = scatter_op.getLoc();
    auto transposed_operand = rewriter.create<mhlo::TransposeOp>(
        loc, permutation_and_shape.shape, operands[0],
        permutation_and_shape.permutation);

    // Apply TF scatter to update the trailing dimensions of the
    // transposed operand.
    auto tf_scatter_op =
        rewriter.create<TfOp>(loc, permutation_and_shape.shape,
                              transposed_operand, indices, new_updates);

    // Reverse the earlier transpose.
    auto inverse_permutation =
        GetInversePermutation(permutation_array, rewriter);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
        scatter_op, scatter_op.getResult(0).getType(), tf_scatter_op,
        inverse_permutation);

    return success();
  }
};
using ConvertScatterAddOp =
    ConvertScatterOp<mhlo::AddOp, TF::TensorScatterAddOp>;
using ConvertScatterMaxOp =
    ConvertScatterOp<mhlo::MaxOp, TF::TensorScatterMaxOp>;
using ConvertScatterMinOp =
    ConvertScatterOp<mhlo::MinOp, TF::TensorScatterMinOp>;
using ConvertScatterSubOp =
    ConvertScatterOp<mhlo::SubtractOp, TF::TensorScatterSubOp>;
using ConvertScatterUpdateOp =
    ConvertScatterOp<void, TF::TensorScatterUpdateOp>;

// Converts mhlo.pad to tf.PadV2
Value ConvertPadOp(PatternRewriter &rewriter, Operation *old_op) {
  auto pad_op = cast<mhlo::PadOp>(old_op);
  mlir::Location loc = pad_op.getLoc();

  llvm::SmallVector<APInt, 8> padding;
  for (auto p : llvm::zip(pad_op.edge_padding_low().getValues<APInt>(),
                          pad_op.edge_padding_high().getValues<APInt>())) {
    padding.push_back(std::get<0>(p));
    padding.push_back(std::get<1>(p));
  }
  auto attr_type = RankedTensorType::get({pad_op.edge_padding_low().size(), 2},
                                         rewriter.getI64Type());
  auto padding_attr = DenseIntElementsAttr::get(attr_type, padding);
  auto padding_op =
      rewriter.create<arith::ConstantOp>(loc, attr_type, padding_attr);
  return rewriter.create<PadV2Op>(loc, pad_op.getType(), pad_op.operand(),
                                  padding_op, pad_op.padding_value());
}

// Returns true if broadcast_dimensions obey Tensorflow convention, as in new
// dimensions are added as prefix.
bool IsTFStyleBroadcast(DenseIntElementsAttr broadcast_dimensions,
                        Value output) {
  // broadcast_dimensions is an increasing list by definition, thus it suffices
  // to check the first element.
  int64_t input_rank = broadcast_dimensions.getNumElements();
  int64_t output_rank = output.getType().cast<ShapedType>().getRank();
  return input_rank == 0 ||
         (broadcast_dimensions.getValues<APInt>()[0].getSExtValue() ==
          output_rank - input_rank);
}

// Returns the intermediate shape that input tensor should be reshaped to during
// legalization of BroadcastInDimOp.
arith::ConstantOp ExpandedShape(PatternRewriter &rewriter, Value input,
                                DenseIntElementsAttr broadcast_dimensions,
                                Value output) {
  // Initialize expanded shape with output rank and dimensions of 1.
  SmallVector<Attribute, 4> expanded_shape(
      output.getType().cast<ShapedType>().getRank(),
      /*Value=*/rewriter.getI64IntegerAttr(1));

  // Set dimension sizes specified by broadcast_dimensions.
  ArrayRef<int64_t> input_shape = input.getType().cast<ShapedType>().getShape();
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

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_legalize_hlo.inc"

/// Performs the lowering to XLA dialect.
void LegalizeHloToTf::runOnOperation() {
  MLIRContext &context = getContext();

  // Add legalization patterns to the list.
  RewritePatternSet patterns(&getContext());
  PopulateLegalizeHloToTfPatterns(&patterns, &context);

  ConversionTarget target(context);
  target.addLegalDialect<TensorFlowDialect>();
  target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();
  target.addLegalOp<mhlo::TupleOp>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("mhlo to TF legalization failed.");
    signalPassFailure();
  }
}

}  // end namespace

void PopulateLegalizeHloToTfPatterns(RewritePatternSet *patterns,
                                     MLIRContext *context) {
  patterns->add<
      ConvertAvgPoolOp, Convert2DConvOp, Convert1DConvOp,
      ConvertNonTrivialConvOp, ConvertDynamicSliceOp,
      ConvertDynamicUpdateSliceOp, ConvertGatherOp, ConvertIfOp,
      ConvertMaxPoolOp, ConvertScatterAddOp, ConvertScatterMaxOp,
      ConvertScatterMinOp, ConvertScatterSubOp, ConvertScatterUpdateOp,
      ConvertSliceOp, ConvertReduceOpToTfArgmax, ConvertReduceOpToTfArgmin,
      ConvertReduceOpToTfMax, ConvertReduceOpToTfMin, ConvertReduceOpToTfAll,
      ConvertReduceOpToTfAny, ConvertReduceOpToTfSum, ConvertSortToTfTopk,
      ConvertIotaOpToTfRange, ConvertWhileOp, ConvertLoweredCumSumOp,
      ConvertLoweredCumProdOp>(context);
  populateWithGenerated(*patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeHloToTfPass() {
  return std::make_unique<LegalizeHloToTf>();
}

}  // end namespace TF
}  // end namespace mlir
