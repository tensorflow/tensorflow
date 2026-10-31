/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/xla_ops_to_stablehlo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace TFL {
namespace {

// Reads an integer constant operand (i32 or i64 elements) into int64 values.
// Returns std::nullopt if `value` is not produced by a constant op.
std::optional<SmallVector<int64_t>> GetConstantI64Values(Value value) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return std::nullopt;
  SmallVector<int64_t> result;
  result.reserve(attr.getNumElements());
  for (const APInt& v : attr.getValues<APInt>()) {
    result.push_back(v.getSExtValue());
  }
  return result;
}

// xla::ConvolutionDimensionNumbers -> #stablehlo.conv. Mirrors
// xla::stablehlo::ConvertConvDimensionNumbers (hlo_to_mhlo/attribute_importer)
// without pulling in the HLO translate / MHLO libraries.
stablehlo::ConvDimensionNumbersAttr ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums, Builder& builder) {
  SmallVector<int64_t> input_spatial_dims(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  SmallVector<int64_t> kernel_spatial_dims(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  SmallVector<int64_t> output_spatial_dims(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  return stablehlo::ConvDimensionNumbersAttr::get(
      builder.getContext(), dnums.input_batch_dimension(),
      dnums.input_feature_dimension(), input_spatial_dims,
      dnums.kernel_input_feature_dimension(),
      dnums.kernel_output_feature_dimension(), kernel_spatial_dims,
      dnums.output_batch_dimension(), dnums.output_feature_dimension(),
      output_spatial_dims);
}

// xla::GatherDimensionNumbers -> #stablehlo.gather. Mirrors
// xla::stablehlo::ConvertGatherDimensionNumbers.
stablehlo::GatherDimensionNumbersAttr ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, Builder& builder) {
  SmallVector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  SmallVector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  SmallVector<int64_t> operand_batching_dims(
      dnums.operand_batching_dims().begin(),
      dnums.operand_batching_dims().end());
  SmallVector<int64_t> start_indices_batching_dims(
      dnums.start_indices_batching_dims().begin(),
      dnums.start_indices_batching_dims().end());
  SmallVector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  return stablehlo::GatherDimensionNumbersAttr::get(
      builder.getContext(), offset_dims, collapsed_slice_dims,
      operand_batching_dims, start_indices_batching_dims, start_index_map,
      dnums.index_vector_dim());
}

// xla::PrecisionConfig -> array of #stablehlo<precision ...>. Returns a null
// attribute if the config has no operand precisions, and failure if a
// precision value has no StableHLO equivalent (e.g. PACKED_NIBBLE).
FailureOr<ArrayAttr> ConvertPrecisionConfig(const xla::PrecisionConfig& config,
                                            Builder& builder) {
  if (config.operand_precision_size() == 0) return ArrayAttr();
  SmallVector<Attribute, 2> precision_attrs;
  for (int prec : config.operand_precision()) {
    std::optional<stablehlo::Precision> precision =
        stablehlo::symbolizePrecision(xla::PrecisionConfig_Precision_Name(
            static_cast<xla::PrecisionConfig::Precision>(prec)));
    if (!precision.has_value()) return failure();
    precision_attrs.push_back(
        stablehlo::PrecisionAttr::get(builder.getContext(), *precision));
  }
  return builder.getArrayAttr(precision_attrs);
}

// Converts tf.XlaConvV2 to stablehlo.convolution. Port of
// ConvertXlaConvV2Op in tf2xla/transforms/legalize_tf.cc.
class ConvertXlaConvV2Op : public OpRewritePattern<TF::XlaConvV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaConvV2Op op,
                                PatternRewriter& rewriter) const override {
    std::optional<SmallVector<int64_t>> window_strides =
        GetConstantI64Values(op.getWindowStrides());
    std::optional<SmallVector<int64_t>> padding =
        GetConstantI64Values(op.getPadding());
    std::optional<SmallVector<int64_t>> lhs_dilation =
        GetConstantI64Values(op.getLhsDilation());
    std::optional<SmallVector<int64_t>> rhs_dilation =
        GetConstantI64Values(op.getRhsDilation());
    std::optional<SmallVector<int64_t>> feature_group_count =
        GetConstantI64Values(op.getFeatureGroupCount());
    if (!window_strides || !padding || !lhs_dilation || !rhs_dilation ||
        !feature_group_count || feature_group_count->size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "window/padding/dilation/feature_group_count must be constant");
    }
    const int64_t num_spatial_dims =
        static_cast<int64_t>(window_strides->size());
    if (static_cast<int64_t>(padding->size()) != 2 * num_spatial_dims) {
      return rewriter.notifyMatchFailure(op, "unexpected padding size");
    }

    xla::ConvolutionDimensionNumbers dnums;
    if (!dnums.ParseFromString(op.getDimensionNumbers().str())) {
      return rewriter.notifyMatchFailure(
          op, "failed to parse ConvolutionDimensionNumbers");
    }
    xla::PrecisionConfig precision_config;
    if (!precision_config.ParseFromString(op.getPrecisionConfig().str())) {
      return rewriter.notifyMatchFailure(op, "failed to parse PrecisionConfig");
    }
    FailureOr<ArrayAttr> precision_config_attr =
        ConvertPrecisionConfig(precision_config, rewriter);
    if (failed(precision_config_attr)) {
      return rewriter.notifyMatchFailure(op, "unsupported precision config");
    }

    // padding is an [N, 2] i64 tensor attribute in StableHLO.
    auto padding_type = RankedTensorType::get({num_spatial_dims, 2},
                                              rewriter.getIntegerType(64));
    auto padding_attr =
        DenseIntElementsAttr::get(padding_type, ArrayRef<int64_t>(*padding));

    SmallVector<NamedAttribute, 8> attrs = {
        rewriter.getNamedAttr("window_strides",
                              rewriter.getDenseI64ArrayAttr(*window_strides)),
        rewriter.getNamedAttr("padding", padding_attr),
        rewriter.getNamedAttr("lhs_dilation",
                              rewriter.getDenseI64ArrayAttr(*lhs_dilation)),
        rewriter.getNamedAttr("rhs_dilation",
                              rewriter.getDenseI64ArrayAttr(*rhs_dilation)),
        rewriter.getNamedAttr(
            "feature_group_count",
            rewriter.getI64IntegerAttr(feature_group_count->front())),
        rewriter.getNamedAttr(
            "batch_group_count",
            rewriter.getI64IntegerAttr(op.getBatchGroupCount())),
        rewriter.getNamedAttr("dimension_numbers",
                              ConvertConvDimensionNumbers(dnums, rewriter)),
    };
    if (*precision_config_attr) {
      attrs.push_back(
          rewriter.getNamedAttr("precision_config", *precision_config_attr));
    }

    // XlaConvV2 allows mixed lhs/rhs element types; StableHLO (and the
    // downstream StableHLO->TF patterns) expect them to match.
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhs_type = mlir::cast<ShapedType>(lhs.getType());
    auto rhs_type = mlir::cast<ShapedType>(rhs.getType());
    if (lhs_type.getElementType() != rhs_type.getElementType()) {
      if (!rhs_type.hasRank() ||
          lhs_type.getElementType().getIntOrFloatBitWidth() <
              rhs_type.getElementType().getIntOrFloatBitWidth()) {
        return rewriter.notifyMatchFailure(op, "unsupported mixed types");
      }
      auto new_rhs_type =
          RankedTensorType::get(rhs_type.getShape(), lhs_type.getElementType());
      rhs = stablehlo::ConvertOp::create(rewriter, op.getLoc(), new_rhs_type,
                                         rhs);
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, op.getType(), ValueRange{lhs, rhs}, attrs);
    return success();
  }
};

// Converts tf.XlaGather to stablehlo.gather. Equivalent to the tf2xla kernel
// (tf2xla/kernels/gather_scatter_ops.cc) which emits xla::Gather.
class ConvertXlaGatherOp : public OpRewritePattern<TF::XlaGatherOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaGatherOp op,
                                PatternRewriter& rewriter) const override {
    std::optional<SmallVector<int64_t>> slice_sizes =
        GetConstantI64Values(op.getSliceSizes());
    if (!slice_sizes) {
      return rewriter.notifyMatchFailure(op, "slice_sizes must be constant");
    }

    xla::GatherDimensionNumbers dnums;
    // The tf2xla kernel uses ParsePartialFromString.
    if (!dnums.ParsePartialFromString(op.getDimensionNumbers().str())) {
      return rewriter.notifyMatchFailure(
          op, "failed to parse GatherDimensionNumbers");
    }

    rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
        op, op.getType(), op.getOperand(), op.getStartIndices(),
        ConvertGatherDimensionNumbers(dnums, rewriter),
        rewriter.getDenseI64ArrayAttr(*slice_sizes),
        rewriter.getBoolAttr(op.getIndicesAreSorted()));
    return success();
  }
};

// Converts stablehlo.gather to stablehlo.slice (+ stablehlo.reshape) when the
// gather has a single, constant index vector. StableHLO port of the
// mhlo::GatherOp "GatherSlice" canonicalization (mhlo/IR/hlo_ops.cc), which
// stablehlo::GatherOp does not provide as a canonicalizer.
class ConvertConstantIndexGatherToSlice
    : public OpRewritePattern<stablehlo::GatherOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::GatherOp gather,
                                PatternRewriter& rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index))) {
      return failure();
    }

    stablehlo::GatherDimensionNumbersAttr dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1) {
      return failure();
    }
    if (!dnums.getOperandBatchingDims().empty() ||
        !dnums.getStartIndicesBatchingDims().empty()) {
      return failure();
    }
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size())) {
      return failure();
    }

    auto operand_type =
        mlir::dyn_cast<RankedTensorType>(gather.getOperand().getType());
    if (!operand_type || !operand_type.hasStaticShape()) return failure();

    SmallVector<int64_t> slice_end(gather.getSliceSizes());
    SmallVector<int64_t> slice_start(slice_end.size(), 0);
    for (auto [map_index, idx] :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          std::clamp(idx.getSExtValue(), static_cast<int64_t>(0),
                     operand_type.getDimSize(map_index) - slice_end[map_index]);
      slice_start[map_index] += offset;
      slice_end[map_index] += offset;
    }

    SmallVector<int64_t> slice_stride(slice_end.size(), 1);
    SmallVector<int64_t> slice_shape(slice_end.size());
    for (size_t i = 0; i < slice_end.size(); ++i) {
      slice_shape[i] = slice_end[i] - slice_start[i];
    }
    Type element_type =
        mlir::cast<ShapedType>(gather.getType()).getElementType();
    auto slice_type = RankedTensorType::get(slice_shape, element_type);
    Value result = stablehlo::SliceOp::create(
        rewriter, gather.getLoc(), slice_type, gather.getOperand(),
        rewriter.getDenseI64ArrayAttr(slice_start),
        rewriter.getDenseI64ArrayAttr(slice_end),
        rewriter.getDenseI64ArrayAttr(slice_stride));

    ArrayRef<int64_t> collapsed_slice_dims = dnums.getCollapsedSliceDims();
    if (!collapsed_slice_dims.empty()) {
      SmallVector<int64_t> reshape_shape;
      for (size_t i = 0; i < slice_shape.size(); ++i) {
        if (!llvm::is_contained(collapsed_slice_dims,
                                static_cast<int64_t>(i))) {
          reshape_shape.push_back(slice_shape[i]);
        }
      }
      auto reshape_type = RankedTensorType::get(reshape_shape, element_type);
      result = stablehlo::ReshapeOp::create(rewriter, gather.getLoc(),
                                            reshape_type, result);
    }

    // Like the mhlo canonicalization, keep the (possibly less static) type of
    // the original gather so users are not affected. The op is expected to be
    // converted to TF right away (tf.StridedSlice / tf.Reshape).
    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

}  // namespace

void PopulateLegalizeTfXlaOpsToStablehloPatterns(MLIRContext* context,
                                                 RewritePatternSet& patterns) {
  patterns.add<ConvertXlaConvV2Op, ConvertXlaGatherOp>(context);
  // Prefer the slice rewrite over the generic stablehlo.gather -> TF lowering
  // (odml ConvertGatherOp), matching the previous mhlo canonicalize-first
  // behavior.
  patterns.add<ConvertConstantIndexGatherToSlice>(context, /*benefit=*/2);
}

}  // namespace TFL
}  // namespace mlir
