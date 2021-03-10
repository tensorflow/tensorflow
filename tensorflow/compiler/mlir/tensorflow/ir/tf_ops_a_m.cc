/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {

namespace {
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_helpers.inc"
#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<AddToAddV2>(context);
}

//===----------------------------------------------------------------------===//
// AddNOp
//===----------------------------------------------------------------------===//

OpFoldResult AddNOp::fold(ArrayRef<Attribute> operands) {
  if (operands.size() == 1) return *inputs().begin();
  return {};
}

//===----------------------------------------------------------------------===//
// AddV2Op
//===----------------------------------------------------------------------===//

void AddV2Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<AddV2OfNegLeft, AddV2OfNegRight>(context);
}

OpFoldResult AddV2Op::fold(ArrayRef<Attribute> operands) {
  return IdentityArithmeticOpFolder<AddV2Op>(*this, operands);
}

//===----------------------------------------------------------------------===//
// AllOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AllOp op) {
  return VerifyReductionInputAndDims(op.input(), op.reduction_indices(),
                                     op.getLoc());
}

//===----------------------------------------------------------------------===//
// AnyOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AnyOp op) {
  return VerifyReductionInputAndDims(op.input(), op.reduction_indices(),
                                     op.getLoc());
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

namespace {

// Removes Assert with constant true predicate.
struct AssertWithTrue : public OpRewritePattern<AssertOp> {
  using OpRewritePattern<AssertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssertOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr cst;
    if (matchPattern(op.condition(), m_Constant(&cst))) {
      if (cst.getValue<BoolAttr>({}).getValue()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};
}  // namespace

void AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<AssertWithTrue>(context);
}

//===----------------------------------------------------------------------===//
// BatchMatMulV2Op & BatchMatMulOp
//===----------------------------------------------------------------------===//

template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, BatchMatMulOp, BatchMatMulV2Op>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  if (!HasRankAtLeast(op.x(), 2)) {
    return op.emitOpError("requires lhs operand to have rank at least two");
  }
  if (!HasRankAtLeast(op.y(), 2)) {
    return op.emitOpError("requires rhs operand to have rank at least two");
  }

  RankedTensorType x_ty = GetRankedTensorTypeForOperand(op.x());
  RankedTensorType y_ty = GetRankedTensorTypeForOperand(op.y());

  if (!x_ty || !y_ty) return success();

  ArrayRef<int64_t> x_shape = x_ty.getShape();
  ArrayRef<int64_t> y_shape = y_ty.getShape();

  llvm::SmallVector<int64_t, 4> result_batch_shape;
  llvm::ArrayRef<int64_t> x_batches = x_shape.drop_back(2);
  llvm::ArrayRef<int64_t> y_batches = y_shape.drop_back(2);

  // Check compatibility of batch dimensions if both input shapes are known.
  // BatchMatMul should have exactly the same batch dimensions and
  // BatchMatMulV2 should have broadcastable batch dimensions.
  //
  // The last two dimensions are non-batch dimensions that don't need to
  // participate in batch dimension compatibility check.
  if (std::is_same<OpT, BatchMatMulOp>()) {
    for (const auto &dim_pairs : llvm::zip(x_batches, y_batches)) {
      int64_t x_dim = std::get<0>(dim_pairs);
      int64_t y_dim = std::get<1>(dim_pairs);
      if (!ShapedType::isDynamic(x_dim) && !ShapedType::isDynamic(y_dim) &&
          x_dim != y_dim) {
        return op.emitOpError()
               << "found mismatching batch dimensions for lhs shape " << x_ty
               << " and rhs shape " << y_ty;
      }
    }
  } else {
    if (!OpTrait::util::getBroadcastedShape(x_batches, y_batches,
                                            result_batch_shape))
      return op.emitOpError()
             << "found incompatible broadcast batch dimensions for lhs shape "
             << x_ty << " and rhs shape " << y_ty;
  }

  RankedTensorType output_ty = GetRankedTensorTypeForOperand(op.output());
  if (!output_ty) return success();

  int64_t expected_output_rank = std::max(x_ty.getRank(), y_ty.getRank());
  if (output_ty.getRank() != expected_output_rank)
    return op.emitOpError()
           << "found invalid output rank, expected " << expected_output_rank
           << " but got " << output_ty.getRank();

  // Check output batch dim with potential broadcasting.
  ArrayRef<int64_t> output_shape = output_ty.getShape();
  for (int i = 0; i < result_batch_shape.size(); ++i) {
    if (output_shape[i] != ShapedType::kDynamicSize &&
        output_shape[i] != result_batch_shape[i])
      return op.emitOpError()
             << "has mismatching input batch dimension "
             << result_batch_shape[i] << " and output batch dimension "
             << output_shape[i];
  }

  // Check output shape for non-batch dimension, following documentation below.
  // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-mat-mul
  int64_t x_row_dim = x_shape[x_shape.size() - 2];
  int64_t x_col_dim = x_shape[x_shape.size() - 1];
  int64_t y_row_dim = y_shape[y_shape.size() - 2];
  int64_t y_col_dim = y_shape[y_shape.size() - 1];
  int64_t out_row_dim = output_shape[output_shape.size() - 2];
  int64_t out_col_dim = output_shape[output_shape.size() - 1];

  int64_t expected_out_row_dim = op.adj_x() ? x_col_dim : x_row_dim;
  int64_t expected_out_col_dim = op.adj_y() ? y_row_dim : y_col_dim;

  if (expected_out_row_dim != ShapedType::kDynamicSize &&
      out_row_dim != ShapedType::kDynamicSize &&
      out_row_dim != expected_out_row_dim)
    return op.emitOpError()
           << "found invalid output dimension on row, expected "
           << expected_out_row_dim << " but got " << out_row_dim;
  if (expected_out_col_dim != ShapedType::kDynamicSize &&
      out_col_dim != ShapedType::kDynamicSize &&
      out_col_dim != expected_out_col_dim)
    return op.emitOpError()
           << "found invalid output dimension on col, expected "
           << expected_out_col_dim << " but got " << out_col_dim;

  return success();
}

void BatchMatMulOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<BatchMatMulToV2>(context);
}

void BatchMatMulV2Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<BatchMatMulV2ToMatMul>(context);
}

//===----------------------------------------------------------------------===//
// BatchToSpaceOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BatchToSpaceOp op) {
  // Op already has a constraint that block_size >= 2.
  int64_t block_size = op.block_size();

  llvm::SmallVector<int64_t, 4> input_shape(4, ShapedType::kDynamicSize);
  auto input_type = op.input().getType().cast<TensorType>();
  if (input_type.hasRank()) {
    if (input_type.getRank() != 4)
      return op.emitOpError()
             << "requires input to be a 4D tensor, but got " << input_type;

    int64_t input_batch = input_type.getDimSize(0);
    if (input_batch != ShapedType::kDynamicSize &&
        input_batch % (block_size * block_size) != 0) {
      return op.emitOpError()
             << "requires input batch (dimension 0) to be evenly divisible "
                "by (block_size * block_size), but got input batch "
             << input_batch << " and block_size " << block_size;
    }

    input_shape.assign(input_type.getShape().begin(),
                       input_type.getShape().end());
  }

  auto crops_type = op.crops().getType().cast<TensorType>();
  if (crops_type.hasRank()) {
    if (crops_type.getRank() != 2)
      return op.emitOpError()
             << "requires crops to be a 2D tensor, but got " << crops_type;

    auto dim_of_size = [&](int64_t dim, int64_t size) {
      if (crops_type.isDynamicDim(dim)) return true;
      return crops_type.getDimSize(dim) == size;
    };
    if (!dim_of_size(0, 2) || !dim_of_size(1, 2))
      return op.emitOpError()
             << "requires crops to be a tensor<2x2>, but got " << crops_type;
  }

  DenseIntElementsAttr crops_attr;
  // Crops are defined as [[crop_top, crop_bottom], [crop_left, crop_right]],
  // and flattened as [crop_top, crop_bottom, crop_left, crop_right]
  llvm::SmallVector<int64_t, 4> crops_values;
  if (matchPattern(op.crops(), m_Constant(&crops_attr))) {
    assert(crops_attr.getNumElements() == 4 &&
           "tf.BatchToSpace crops must have 4 elements");

    auto crops_range = crops_attr.getIntValues();
    for (const auto &crops_value : crops_range) {
      int64_t crops_value_int = crops_value.getSExtValue();
      if (crops_value_int < 0)
        return op.emitOpError()
               << "requires all crop values to be nonnegative, but got "
               << crops_attr;

      crops_values.push_back(crops_value_int);
    }
  }

  auto output_type = op.output().getType().cast<TensorType>();
  if (output_type.hasRank()) {
    if (output_type.getRank() != 4)
      return op.emitOpError()
             << "requires output to be a 4D tensor, but got " << output_type;

    auto static_dims = [](int64_t dim_a, int64_t dim_b) {
      return dim_a != ShapedType::kDynamicSize &&
             dim_b != ShapedType::kDynamicSize;
    };

    auto output_shape = output_type.getShape();

    // output batch = input batch / (block_size * block_size).
    int64_t input_batch = input_shape[0];
    int64_t output_batch = output_shape[0];
    if (static_dims(input_batch, output_batch) &&
        (output_batch * block_size * block_size) != input_batch)
      return op.emitOpError()
             << "requires output batch (dimension 0) to be equal to input "
                "batch (dimension 0) / (block_size * block_size), but got "
                "output batch "
             << output_batch << ", input batch " << input_batch
             << ", and block_size " << block_size;

    auto check_spatial_dim = [&](int64_t spatial_dim_index,
                                 llvm::StringRef dim_name,
                                 llvm::StringRef crop_a_name,
                                 llvm::StringRef crop_b_name) -> LogicalResult {
      int64_t input_dim = input_shape[spatial_dim_index];
      int64_t output_dim = output_shape[spatial_dim_index];
      if (!static_dims(input_dim, output_dim)) return success();

      int64_t input_dim_pad = input_dim * block_size;
      // If crops are unknown, the maximum output spatial dim size is input
      // spatial dim size * block_size, as crops can be minimum 0.
      if (crops_values.empty() && output_dim > input_dim * block_size)
        return op.emitOpError()
               << "requires output " << dim_name << " (dimension "
               << spatial_dim_index << ") to be less than or equal to input "
               << dim_name << " (dimension " << spatial_dim_index
               << ") * block_size, but got output " << dim_name << " "
               << output_dim << ", input " << dim_name << " " << input_dim
               << ", and block_size " << block_size;

      if (!crops_values.empty()) {
        // output spatial dim = input spatial dim * block_size - crops.
        int64_t crop_a = crops_values[2 * (spatial_dim_index - 1)];
        int64_t crop_b = crops_values[2 * (spatial_dim_index - 1) + 1];
        if (output_dim != input_dim_pad - crop_a - crop_b)
          return op.emitOpError()
                 << "requires output " << dim_name << " (dimension "
                 << spatial_dim_index << ") to be equal to input " << dim_name
                 << " (dimension " << spatial_dim_index << ") * block_size - "
                 << crop_a_name << " - " << crop_b_name << ", but got output "
                 << dim_name << " " << output_dim << ", input " << dim_name
                 << " " << input_dim << ", " << crop_a_name << " " << crop_a
                 << ", " << crop_b_name << " " << crop_b << ", and block_size "
                 << block_size;
      }

      return success();
    };

    if (failed(check_spatial_dim(1, "height", "crop_top", "crop_bottom")) ||
        failed(check_spatial_dim(2, "width", "crop_left", "crop_right")))
      return failure();

    int64_t input_depth = input_shape[3];
    int64_t output_depth = output_shape[3];
    if (static_dims(input_depth, output_depth) && output_depth != input_depth)
      return op.emitOpError()
             << "requires output depth (dimension 3) to be equal to input "
                "depth (dimension 3), but got output depth "
             << output_depth << " and input depth " << input_depth;
  }

  return success();
}

void BatchToSpaceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<BatchToSpaceToBatchToSpaceND>(context);
}

//===----------------------------------------------------------------------===//
// BatchToSpaceNDOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BatchToSpaceNDOp op) {
  auto block_shape_ty = op.block_shape().getType().cast<ShapedType>();
  auto crops_ty = op.crops().getType().cast<ShapedType>();

  if (block_shape_ty.hasStaticShape() && crops_ty.hasStaticShape()) {
    const int block_rank = block_shape_ty.getShape().front();
    if (crops_ty.getRank() != 2 || crops_ty.getShape().front() != block_rank ||
        crops_ty.getShape()[1] != 2) {
      op.emitOpError() << "crops should have shape [" << block_rank
                       << ", 2] instead of " << crops_ty.getShape();
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BiasAddOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * the value and bias operands have valid ranks or are unranked.
// * Channel dimension of the value operand and length of bias matches if they
//   are not unknown.
//
static LogicalResult Verify(BiasAddOp op) {
  absl::string_view data_format(op.data_format().data(),
                                op.data_format().size());
  tensorflow::TensorFormat format;
  bool is_valid = FormatFromString(data_format, &format);
  DCHECK(is_valid) << data_format;
  if (format == tensorflow::TensorFormat::FORMAT_NHWC) {
    if (!HasRankAtLeast(op.value(), 2))
      return op.emitOpError(
          "requires value operand to have rank at least two with `NHWC` data "
          "format");
  } else {
    // Op definition requires data_format to be either NHWC or NCHW.
    DCHECK_EQ(format, tensorflow::TensorFormat::FORMAT_NCHW);
    if (!HasRankAtLeast(op.value(), 3))
      return op.emitOpError(
          "requires value operand to have rank at least three with `NCHW` data "
          "format");
  }

  if (!IsOfRankOrUnranked(op.bias(), 1))
    return op.emitOpError("requires bias operand to have rank exactly one");

  RankedTensorType value_ty = op.value().getType().dyn_cast<RankedTensorType>();
  RankedTensorType bias_ty = op.bias().getType().dyn_cast<RankedTensorType>();
  if (!bias_ty || !value_ty) return success();

  int64_t feature_dim_idx =
      tensorflow::GetTensorFeatureDimIndex(value_ty.getRank(), format);
  int64_t feature_dim = value_ty.getDimSize(feature_dim_idx);
  int64_t bias_len = bias_ty.getDimSize(0);
  if (feature_dim != -1 && bias_len != -1 && feature_dim != bias_len) {
    return op.emitOpError()
           << "requires channel dimension and feature dimension to match; "
              "found "
           << feature_dim << " and " << bias_len << ", respectively";
  }
  return success();
}

LogicalResult BiasAddOp::UpdateDataFormat(StringRef data_format) {
  return ::mlir::TF::UpdateDataFormat(data_format, this);
}

StringRef BiasAddOp::GetOptimalLayout(const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // Prefer NHWC for GPU devices.
  return "NHWC";
}

//===----------------------------------------------------------------------===//
// BiasAddGradOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * the out_backprop operands have valid ranks or are unranked.
//
static LogicalResult Verify(BiasAddGradOp op) {
  absl::string_view data_format(op.data_format().data(),
                                op.data_format().size());
  tensorflow::TensorFormat format;
  bool is_valid = FormatFromString(data_format, &format);
  DCHECK(is_valid) << data_format;
  if (format == tensorflow::TensorFormat::FORMAT_NHWC) {
    if (!HasRankAtLeast(op.out_backprop(), 2))
      return op.emitOpError(
          "requires out_backprop operand to have rank at least two with `NHWC` "
          "data format");
  } else {
    // Op definition requires data_format to be either NHWC or NCHW.
    DCHECK_EQ(format, tensorflow::TensorFormat::FORMAT_NCHW);
    if (!HasRankAtLeast(op.out_backprop(), 3))
      return op.emitOpError(
          "requires out_backprop operand to have rank at least three with "
          "`NCHW` data format");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BiasAddV1Op
//===----------------------------------------------------------------------===//

void BiasAddV1Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<BiasAddV1ToBiasAdd>(context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

void BitcastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<BitcastSameType, BitcastNested>(context);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BroadcastToOp op) {
  // TODO(antiagainst): check that
  // * The 'shape' input is an 1-D int tensor.
  // * Each dimension pair of the source and target shapes are either equal
  //   or one of them is one.
  return success();
}

OpFoldResult BroadcastToOp::fold(ArrayRef<Attribute> operands) {
  Value input = this->input();

  // Fold broadcast if operand and result types are the same and all dimensions
  // are statically known (no-op broadcast).
  auto result_ty = getType().dyn_cast<ShapedType>();
  if (result_ty && result_ty.hasStaticShape() && result_ty == input.getType()) {
    return input;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BroadcastGradientArgsOp
//===----------------------------------------------------------------------===//

namespace {
// Returns `true` if both s0 & s1 are defined via constant op, and fills
// s0_shape & s1_shape.
bool ExtractInputConstShape(BroadcastGradientArgsOp op,
                            DenseIntElementsAttr &s0, DenseIntElementsAttr &s1,
                            SmallVectorImpl<int64_t> &s0_shape,
                            SmallVectorImpl<int64_t> &s1_shape) {
  if (!matchPattern(op.s0(), m_Constant(&s0))) return false;
  if (!matchPattern(op.s1(), m_Constant(&s1))) return false;

  for (auto s : s0.getIntValues()) s0_shape.push_back(s.getSExtValue());
  for (auto s : s1.getIntValues()) s1_shape.push_back(s.getSExtValue());

  return true;
}

// Calculates r0 & r1 output based on inputs and calculated broadcasted shape.
//
// For given bcasted_shape, s0_shape and s1_shape, the broadcasted dimension is
// calculated and push back to its corresponding result, r0 or r1. For example,
// for s0_shape [1,4] and s1_shape [4, 4], bcasted_shape is computed to be
// [4,4] - this leads to the result of r0 to be [0] as the first dimension of s0
// is broadcasted, and r1 to be <> as no broadcasting is happening for s1.
void GetOutputShapeForBroadcastGradientArgs(ArrayRef<int64_t> bcasted_shape,
                                            ArrayRef<int64_t> s0_shape,
                                            ArrayRef<int64_t> s1_shape,
                                            SmallVectorImpl<int64_t> &r0,
                                            SmallVectorImpl<int64_t> &r1) {
  r0.clear();
  r1.clear();

  // No broadcasting is required if both the shapes are equal.
  if (s0_shape == s1_shape) return;

  for (int i = bcasted_shape.size(); i > 0; --i) {
    int idx = bcasted_shape.size() - i;
    int s0_idx = i > s0_shape.size() ? -1 : s0_shape.size() - i;
    int s1_idx = i > s1_shape.size() ? -1 : s1_shape.size() - i;
    if (s0_idx == -1) {
      r0.push_back(idx);
      if (s1_shape[s1_idx] == 1) r1.push_back(idx);
    } else if (s1_idx == -1) {
      r1.push_back(idx);
      if (s0_shape[s0_idx] == 1) r0.push_back(idx);
    } else if (s0_shape[s0_idx] != s1_shape[s1_idx]) {
      if (s0_shape[s0_idx] != bcasted_shape[idx])
        r0.push_back(idx);
      else
        r1.push_back(idx);
    } else if (s0_shape[s0_idx] == 1) {
      // This op is used to compute the gradient dimensions requiring reduction
      // to match the input dimensions. In case both the dimensions are one,
      // reducing the dimension has no effect. We choose to reduce such
      // dimensions to match the TensorFlow kernel behavior. However, note that
      // the TF behavior in this case is inconsistent with the case with the
      // same shapes.
      r0.push_back(idx);
      r1.push_back(idx);
    }
  }
}
}  // namespace

// Verifies that,
// * Broadcast compatability for input shapes.
// * Output shape dimension matches the expected dimension size for input
// shapes.
static LogicalResult Verify(BroadcastGradientArgsOp op) {
  SmallVector<int64_t, 4> s0_shape, s1_shape;
  DenseIntElementsAttr s0, s1;
  if (!ExtractInputConstShape(op, s0, s1, s0_shape, s1_shape)) return success();

  // If both shape is known const, try to validate shape on them as well.
  SmallVector<int64_t, 4> bcasted_shape;
  if (!OpTrait::util::getBroadcastedShape(s0_shape, s1_shape, bcasted_shape))
    return op.emitOpError() << "requires broadcast compatible shape tensors "
                               "for 's0' and 's1', but got "
                            << s0 << " and " << s1;

  SmallVector<int64_t, 4> r0, r1;
  GetOutputShapeForBroadcastGradientArgs(bcasted_shape, s0_shape, s1_shape, r0,
                                         r1);

  // Verify that output types are of rank one and matches the computed result
  // shape.
  auto r0_ty = op.r0().getType().dyn_cast<RankedTensorType>();
  auto r1_ty = op.r1().getType().dyn_cast<RankedTensorType>();
  if (r0_ty && r0_ty.hasStaticShape() && r0_ty.getDimSize(0) != r0.size())
    return op.emitOpError() << "requires dimension 0 size of 'r0' to be "
                            << r0.size() << " but got " << r0_ty.getShape()[0];
  if (r1_ty && r1_ty.hasStaticShape() && r1_ty.getDimSize(0) != r1.size())
    return op.emitOpError() << "requires dimension 0 size of 'r1' to be "
                            << r1.size() << " but got " << r1_ty.getShape()[0];

  return success();
}

LogicalResult BroadcastGradientArgsOp::fold(
    ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
  SmallVector<int64_t, 4> s0_shape, s1_shape;
  DenseIntElementsAttr s0, s1;
  if (!ExtractInputConstShape(*this, s0, s1, s0_shape, s1_shape))
    return failure();

  // Fold BroadcastGradientArgs into two constants if both of the inputs have
  // known shape.
  SmallVector<int64_t, 4> bcasted_shape;
  // Verifier should already ensure the broadcast compatibility.
  bool bcast_compatible =
      OpTrait::util::getBroadcastedShape(s0_shape, s1_shape, bcasted_shape);
  assert(bcast_compatible);
  (void)bcast_compatible;

  SmallVector<int64_t, 4> r0, r1;
  GetOutputShapeForBroadcastGradientArgs(bcasted_shape, s0_shape, s1_shape, r0,
                                         r1);

  auto build_out_dense_element = [](SmallVectorImpl<int64_t> &shape,
                                    Type input_type) {
    Type element_type = input_type.cast<mlir::TensorType>().getElementType();
    RankedTensorType type = RankedTensorType::get(
        {static_cast<int64_t>(shape.size())}, element_type);
    // Input could only be i32 or i64. For i32, downcast to int32_t array.
    if (element_type.isInteger(32)) {
      SmallVector<int32_t, 4> i32_shape;
      for (auto s : shape) i32_shape.push_back(static_cast<int32_t>(s));
      return DenseIntElementsAttr::get(type, i32_shape);
    } else {
      assert(element_type.isInteger(64));
      return DenseIntElementsAttr::get(type, shape);
    }
  };

  results.push_back(build_out_dense_element(r0, this->s0().getType()));
  results.push_back(build_out_dense_element(r1, this->s1().getType()));

  return success();
}

//===----------------------------------------------------------------------===//
// CaseOp
//===----------------------------------------------------------------------===//

class FoldConstantCaseOp : public OpRewritePattern<TF::CaseOp> {
 public:
  explicit FoldConstantCaseOp(MLIRContext *context)
      : OpRewritePattern<TF::CaseOp>(context) {}
  LogicalResult matchAndRewrite(TF::CaseOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult FoldConstantCaseOp::matchAndRewrite(
    TF::CaseOp op, PatternRewriter &rewriter) const {
  // Extract the constant cond value.
  DenseIntElementsAttr branch;
  if (!matchPattern(op.branch_index(), m_Constant(&branch))) return failure();

  int index = *branch.getValues<int>().begin();
  if (index < 0 || index >= op.num_branches()) index = op.num_branches() - 1;

  auto func = op.branches()[index].cast<SymbolRefAttr>();
  auto empty = rewriter.getStringAttr("");
  auto call_op = rewriter.create<PartitionedCallOp>(
      op.getLoc(), op.getResultTypes(), op.getOperands().drop_front(), func,
      /*config=*/empty, /*config_proto=*/empty, /*executor_type=*/empty);
  CopyDeviceAndUnderscoredAttributes(op.getOperation(), call_op);
  rewriter.replaceOp(op, call_op.getResults());
  return success();
}

void CaseOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<FoldConstantCaseOp, DropAttributes<CaseOp>>(context);
}

static LogicalResult VerifyCaseOpBase(Operation *op, Value branch_index) {
  if (!IsOfRankOrUnranked(branch_index, 0))
    return op->emitOpError()
           << "expects 'branch_index' to be a scalar, but got "
           << branch_index.getType();
  return success();
}

static LogicalResult VerifyCaseOrIfOpBranchFunctions(
    SymbolTableCollection &symbol_table, Operation *op,
    ArrayRef<Attribute> branches,
    llvm::function_ref<std::string(unsigned branch_index)> branch_name) {
  SmallVector<FunctionType, 2> branch_types;
  branch_types.reserve(branches.size());

  // Functions have one less operand compared to op as first operand is elided
  // (`cond` of `tf.If` and `branch_index` of `tf.Case`).
  TypeRangeWithDesc input{op->getOperands().drop_front().getTypes(), "input"};
  TypeRangeWithDesc result{op->getResultTypes(), "result"};

  for (auto branch : llvm::enumerate(branches)) {
    auto branch_func = symbol_table.lookupNearestSymbolFrom<FuncOp>(
        op, branch.value().cast<SymbolRefAttr>());
    if (!branch_func)
      return op->emitOpError()
             << "expects " << branch_name(branch.index()) << " ("
             << branch.value() << ") to point to a defined function";

    FunctionType branch_type = branch_func.getType();
    std::string desc = branch_name(branch.index()) + " input";
    TypeRangeWithDesc branch_input{branch_type.getInputs(), desc};
    if (failed(VerifyTypeRangesAreCompatible(op, branch_input, input)))
      return failure();

    desc = branch_name(branch.index()) + " result";
    TypeRangeWithDesc branch_result{branch_type.getResults(), desc};
    if (failed(VerifyTypeRangesAreCompatible(op, branch_result, result)))
      return failure();

    branch_types.push_back(branch_type);
  }

  // If branches have incompatible input types that means that no tensor can
  // serve as input to all the functions. Hence, the op is invalid.
  int expected_num_inputs = op->getNumOperands() - 1;
  for (int i = 0; i < expected_num_inputs; ++i) {
    SmallVector<Type, 2> branch_input_i_types;
    branch_input_i_types.reserve(branches.size());
    llvm::transform(
        branch_types, std::back_inserter(branch_input_i_types),
        [i](FunctionType &branch_type) { return branch_type.getInput(i); });
    if (!AreCastCompatible(branch_input_i_types)) {
      std::string input_types_str;
      llvm::raw_string_ostream os(input_types_str);
      llvm::interleaveComma(branch_input_i_types, os);
      return op->emitOpError()
             << "expects all branch input type(s) (" << os.str()
             << ") at index " << i << " to be cast compatible";
    }
  }

  return success();
}

static LogicalResult Verify(CaseOp op) {
  return VerifyCaseOpBase(op, op.branch_index());
}

LogicalResult CaseOp::verifySymbolUses(SymbolTableCollection &symbol_table) {
  auto branch_name = [](unsigned index) {
    return llvm::formatv("branch #{0}", index).str();
  };
  // TODO(jpienaar): Remove.
  if (failed(CaseOpAdaptor(*this).verify(getLoc()))) return failure();
  return VerifyCaseOrIfOpBranchFunctions(symbol_table, *this,
                                         branches().getValue(), branch_name);
}

//===----------------------------------------------------------------------===//
// CaseRegionOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CaseRegionOp op) {
  if (op.branches().empty())
    return op.emitOpError() << "expects to have at least 1 region";

  if (failed(VerifyCaseOpBase(op, op.branch_index()))) return failure();

  TypeRangeWithDesc results{op.getResultTypes(), "result"};

  for (auto region_and_idx : llvm::enumerate(op.branches())) {
    std::string description =
        llvm::formatv("branch #{0} result", region_and_idx.index()).str();
    Operation *yield = region_and_idx.value().front().getTerminator();
    TypeRangeWithDesc branch_results{yield->getOperandTypes(), description};
    if (failed(VerifyTypeRangesAreCompatible(op, branch_results, results)))
      return failure();
  }

  return success();
}

namespace {
// Eliminate values that pass through the CaseRegionOp or IfRegionOp branches.
template <class CaseOrIfRegionOp>
class CaseOrIfRegionEliminatePassThrough
    : public OpRewritePattern<CaseOrIfRegionOp> {
  using OpRewritePattern<CaseOrIfRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CaseOrIfRegionOp op,
                                PatternRewriter &rewriter) const override {
    RegionRange branches = op.getRegions();
    SmallVector<Type, 4> new_result_types;
    // Maps pass through results to extern values.
    llvm::SmallDenseMap<Value, Value, 4> result_to_extern_value;

    for (auto result : op.getResults()) {
      unsigned index = result.getResultNumber();
      Region *first_branch = *branches.begin();
      Operation *first_terminator = first_branch->front().getTerminator();
      Value returned_val = first_terminator->getOperand(index);

      // Pass through values would be defined outside the branch region. Keep
      // the type of non pass through results to create a new op later, if
      // required.
      if (returned_val.getParentBlock() == &first_branch->front()) {
        new_result_types.push_back(result.getType());
        continue;
      }
      // Check if the same extern value is returned in each branch.
      for (Region *region : branches.drop_front()) {
        Operation *terminator = region->front().getTerminator();
        if (terminator->getOperand(index) != returned_val) return failure();
      }
      result_to_extern_value[result] = returned_val;
    }

    // If no pass through values are found, no change is required.
    if (result_to_extern_value.empty()) return failure();

    // Create new case/if region op.
    auto new_op = rewriter.create<CaseOrIfRegionOp>(
        op.getLoc(), new_result_types, op.getOperand(), op->getAttrs(),
        op.getNumRegions());

    int next_index = 0;
    for (auto result : op.getResults()) {
      if (!result_to_extern_value.count(result)) {
        result.replaceAllUsesWith(new_op.getResult(next_index++));
        continue;
      }
      result.replaceAllUsesWith(result_to_extern_value[result]);
      for (Region *branch : branches)
        branch->front().getTerminator()->eraseOperand(next_index);
    }

    // Move region bodies to the new op.
    for (auto region_index : llvm::seq<int>(0, branches.size()))
      new_op.getRegion(region_index).takeBody(op.getRegion(region_index));

    op.erase();
    return success();
  }
};
}  // namespace

void CaseRegionOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CaseOrIfRegionEliminatePassThrough<TF::CaseRegionOp>>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  // Cast with the same type is a no-op.
  Value operand = getOperand();
  if (getType() == operand.getType()) return operand;
  return {};
}

//===----------------------------------------------------------------------===//
// ConcatOp and ConcatV2Op
//===----------------------------------------------------------------------===//

template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, ConcatOp, ConcatV2Op>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  // TODO(hinsu): Convert variadic length attributes to derived attributes.
  Operation::operand_range values = op.values();

  int axis_idx = std::is_same<OpT, ConcatOp>() ? 0 : 1;
  Value axis = *op.getODSOperands(axis_idx).begin();
  if (!HasRankAtMost(axis, 1)) {
    return op.emitOpError(
        "requires axis to be of scalar type (or vector type for older "
        "versions)");
  }

  return VerifyTypesCompatibility(values,
                                  /*mask_one_dim=*/true, op.getOperation());
}

void ConcatOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<ConvertToConcatV2>(context);
}

namespace {

// Hoist coefficient-wise unary operation out of the Concat op:
//
//   %0 = "tf.Log1p"(%arg_0)
//   %1 = "tf.Log1p"(%arg_1)
//   ...
//   %n = "tf.Log1p"(%arg_n)
//   %m = "tf.ConcatV2"(%0, %1, ..., %n, %axis)
//
// Rewrite it to:
//
//   %0 = "tf.ConcatV2"(%arg_0, %arg_1, ..., %arg_n, %axis)
//   %1 = "tf.Log1p"(%0)
class HoistCwiseUnaryOutOfConcat : public OpRewritePattern<TF::ConcatV2Op> {
 public:
  explicit HoistCwiseUnaryOutOfConcat(MLIRContext *context)
      : OpRewritePattern<TF::ConcatV2Op>(context) {}
  LogicalResult matchAndRewrite(TF::ConcatV2Op op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult HoistCwiseUnaryOutOfConcat::matchAndRewrite(
    TF::ConcatV2Op op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // All concat operands must be defined by ops.
  Operation *first_arg_op = op.values().front().getDefiningOp();
  if (first_arg_op == nullptr) return failure();

  // All concat operands must be produced by the coeff-wise unary operation.
  if (!first_arg_op->hasTrait<OpTrait::TF::CwiseUnary>()) return failure();

  // All concat operands must be defined by the op of same kind.
  bool args_same_op = llvm::all_of(op.values(), [&](Value arg) -> bool {
    Operation *arg_op = arg.getDefiningOp();
    return arg_op && arg_op->getName() == first_arg_op->getName();
  });
  if (!args_same_op) return failure();

  // Collect unary operations operands.
  auto unary_operands = llvm::map_range(op.values(), [](Value arg) -> Value {
    return arg.getDefiningOp()->getOperand(0);
  });
  SmallVector<Value, 8> unary_ops_args(unary_operands);

  // Concatenate unary ops operands.
  auto concat_unary_operands =
      rewriter.create<ConcatV2Op>(loc, op.getType(), unary_ops_args, op.axis());

  // Replace original concat with an unary op.
  OperationState new_unary_op_state(loc, first_arg_op->getName().getStringRef(),
                                    concat_unary_operands.getResult(),
                                    op.getResult().getType(),
                                    ArrayRef<NamedAttribute>());
  Operation *new_unary_op = rewriter.createOperation(new_unary_op_state);

  rewriter.replaceOp(op, new_unary_op->getResults());

  return success();
}

// Hoist coefficient-wise binary operation out of the Concat op:
//
//   %0 = tf.Mul(%lhs_0, %rhs_0)
//   %1 = tf.Mul(%lhs_1, %rhs_1)
//   ...
//   %n = tf.Mul(%lhs_n, %rhs_n)
//   %m = tf.ConcatV2(%0, %1, ..., %n, %axis)
//
// Rewrite it to:
//
//   %0 = tf.ConcatV2(%lhs0, %lhs1, ..., %lhs_n, %lhs_concat_axis)
//   %1 = tf.ConcatV2(%rhs0, %rhs1, ..., %rhs_n, %rhs_concat_axis)
//   %2 = tf.Mul(%0, %1)
//
// If a minor fraction of the Concat inputs are not of the same binary op kind
// (tf.Mul in the above example), we will synthesize the binary ops for those
// inputs. e.g. if we instead have %1 = %lhs_1, then we would synthesize a
// tf.Mul op over it and a scalar const tensor 1.0. For now this only applies to
// float32 tensors.
// TODO(hongm): Implement this op synthesis optimization for other dtypes if
// needed.
//
// Because coefficient-wise binary operations support implicit broadcasting, we
// should be very careful with this optimization, and do not accidentally
// produce incorrect concat operations.
class HoistCwiseBinaryOutOfConcat : public OpRewritePattern<TF::ConcatV2Op> {
 public:
  explicit HoistCwiseBinaryOutOfConcat(MLIRContext *context)
      : OpRewritePattern<TF::ConcatV2Op>(context) {}
  LogicalResult matchAndRewrite(TF::ConcatV2Op op,
                                PatternRewriter &rewriter) const override;

 private:
  struct HoistParams {
    SmallVector<Value, 8> lhs_args;
    SmallVector<Value, 8> rhs_args;
    int64_t lhs_axis;
    int64_t rhs_axis;
    Type lhs_concat_type;
    Type rhs_concat_type;
    int scalar_operand_idx;  // can be 0 or 1 for the binary op's operands.
  };

  // Returns parameters of a binary op hoisting out of concatenation if all of
  // the operands are in one of the compatible configurations.
  // All inputs of `op` should be of the same binary op kind (e.g. tf.Mul),
  // except from the ones in `exceptions`. In that case, we can synthesize that
  // binary op kind for the values in `exceptions`.
  Optional<HoistParams> GetHoistParams(
      TF::ConcatV2Op op, int64_t axis,
      const llvm::SmallDenseMap<Value, unsigned, 4> &exceptions) const;
};

LogicalResult HoistCwiseBinaryOutOfConcat::matchAndRewrite(
    TF::ConcatV2Op op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Axis must be a constant scalar value.
  DenseIntElementsAttr axis_attr;
  if (!matchPattern(op.axis(), m_Constant(&axis_attr))) return failure();
  if (axis_attr.getNumElements() != 1) return failure();
  int64_t axis =
      axis_attr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // TODO(ezhulenev): Compute axis from rank. e.g. It might be common to concat
  // on the channels dim for NCHW layout as axis=-2.
  if (axis < 0) return failure();

  // All concat operands must be defined by ops of the same kind (e.g. tf.Mul),
  // or some other ops that we might convert to using the same op kind above
  // (e.g. converting op A to tf.Mul(A, 1.0))
  // TODO(hongm): generalize the code here to support cases where the first arg
  // has no defining op (e.g. might be a block arg).
  Operation *first_arg_op = op.values().front().getDefiningOp();
  if (first_arg_op == nullptr) return failure();

  // All concat operands must be produced by the coeff-wise binary operation.
  if (!first_arg_op->hasTrait<OpTrait::TF::CwiseBinary>()) return failure();

  // All concat operands must be defined by the op of same kind, except for a
  // minor portion which we track in `exceptions`.
  // Map from the operands to operand indices.
  llvm::SmallDenseMap<Value, unsigned, 4> exceptions;
  unsigned operand_idx = 0;
  for (Value arg : op.values()) {
    Operation *arg_op = arg.getDefiningOp();
    if (arg_op && arg_op->getName() == first_arg_op->getName()) {
      ++operand_idx;
      continue;
    }
    exceptions[arg] = operand_idx++;
  }
  // Recall those inputs to the concat op that are not produced by a binary op
  // of the `first_arg_op` kind (e.g. tf.Mul) are stored in `exceptions`. If
  // there are too many exceptions, it might not be cost effective to apply the
  // concat hoisting optimization here.
  // Setting the threshold to be 50% as a simple cost model heuristic. e.g. If 1
  // out of 2 concat inputs is an exception, we don't apply the hoist. If it's 1
  // out of 3, we do.
  const float exception_pct_threshold = 0.5;
  if (static_cast<float>(op.values().size()) * exception_pct_threshold <=
      exceptions.size())
    return failure();

  // Compute binary operands hoist parameters.
  auto hoist_params = GetHoistParams(op, axis, exceptions);
  if (!hoist_params.hasValue()) return failure();

  // Process `exceptions`: For each value there, synthesize a binary op of the
  // above kind, so that the concat hoisting optimization can still apply.
  if (!exceptions.empty()) {
    int identity_val;
    if (isa<AddOp>(first_arg_op) || isa<SubOp>(first_arg_op))
      identity_val = 0;
    else if (isa<MulOp>(first_arg_op) || isa<DivOp>(first_arg_op) ||
             isa<RealDivOp>(first_arg_op))
      identity_val = 1;
    else
      return failure();
    DenseElementsAttr const_attr;
    auto scalar_tensor_type =
        first_arg_op->getOperand(hoist_params->scalar_operand_idx)
            .getType()
            .dyn_cast<ShapedType>();
    Type scalar_dtype = scalar_tensor_type.getElementType();
    if (scalar_dtype.isa<FloatType>())
      const_attr = DenseElementsAttr::get(scalar_tensor_type,
                                          static_cast<float>(identity_val));
    else
      return failure();

    // All checks are passes, and we now prepare for rewrite.
    auto identity_const = rewriter.create<TF::ConstOp>(loc, const_attr);
    for (const auto &kv : exceptions) {
      assert(!hoist_params->lhs_args[kv.second]);
      assert(!hoist_params->rhs_args[kv.second]);

      if (hoist_params->scalar_operand_idx == 1) {
        hoist_params->lhs_args[kv.second] = kv.first;
        hoist_params->rhs_args[kv.second] = identity_const;
      } else {
        assert(hoist_params->scalar_operand_idx == 0);
        hoist_params->lhs_args[kv.second] = identity_const;
        hoist_params->rhs_args[kv.second] = kv.first;
      }
    }
  }

  // New lhs and rhs concatenation axis.
  auto axis_type = mlir::RankedTensorType::get({}, rewriter.getIntegerType(64));
  auto lhs_axis = rewriter.create<TF::ConstOp>(
      loc, DenseIntElementsAttr::get(axis_type, hoist_params->lhs_axis));
  auto rhs_axis = rewriter.create<TF::ConstOp>(
      loc, DenseIntElementsAttr::get(axis_type, hoist_params->rhs_axis));

  // Concatenate binary ops operands on the new axis.
  auto lhs_concat = rewriter.create<ConcatV2Op>(
      loc, hoist_params->lhs_concat_type, hoist_params->lhs_args, lhs_axis);
  auto rhs_concat = rewriter.create<ConcatV2Op>(
      loc, hoist_params->rhs_concat_type, hoist_params->rhs_args, rhs_axis);

  // Replace original concat with a binary op.
  OperationState new_binary_op_state(
      loc, first_arg_op->getName().getStringRef(),
      {lhs_concat.getResult(), rhs_concat.getResult()},
      op.getResult().getType(), ArrayRef<NamedAttribute>());
  Operation *new_binary_op = rewriter.createOperation(new_binary_op_state);

  rewriter.replaceOp(op, new_binary_op->getResults());

  return success();
}

Optional<HoistCwiseBinaryOutOfConcat::HoistParams>
HoistCwiseBinaryOutOfConcat::GetHoistParams(
    TF::ConcatV2Op op, int64_t axis,
    const llvm::SmallDenseMap<Value, unsigned, 4> &exceptions) const {
  assert(axis >= 0);
  // Collects lhs or rhs arguments of concat op operands.
  auto args = [&](int operand_idx) -> SmallVector<Value, 8> {
    auto range = llvm::map_range(op.values(), [&](Value arg) {
      if (exceptions.count(arg)) return Value();
      return arg.getDefiningOp()->getOperand(operand_idx);
    });
    return {range.begin(), range.end()};
  };

  // Returns true if all binary ops operands at `operand_idx` index are tensors
  // of `axis + 1` rank and axis dim has size `1`.
  auto is_all_tensors = [&](int operand_idx, int axis) -> bool {
    return llvm::all_of(op.values(), [&](Value arg) -> bool {
      if (exceptions.count(arg)) return true;
      auto operand = arg.getDefiningOp()->getOperand(operand_idx);
      auto ranked = operand.getType().dyn_cast<RankedTensorType>();
      return ranked && ranked.getRank() == (axis + 1) &&
             ranked.getShape()[axis] == 1;
    });
  };

  // Returns true if all binary ops operands at `operand_idx` index are scalars.
  auto is_all_scalars = [&](int operand_idx) -> bool {
    return llvm::all_of(op.values(), [&](Value arg) -> bool {
      if (exceptions.count(arg)) return true;
      auto operand = arg.getDefiningOp()->getOperand(operand_idx);
      auto ranked = operand.getType().dyn_cast<RankedTensorType>();
      return ranked && ranked.hasRank() && ranked.getRank() == 0;
    });
  };

  // Concat result type must be a ranked tensor.
  auto ranked = op.getType().dyn_cast<RankedTensorType>();
  if (!ranked) return None;

  // TODO(ezhulenev): Add support for more valid concat patterns.

  // Tensor + Scalar: [..., 1] + []  <- scalar
  //                        ^
  //                        \- axis is the innermost dimension.
  //
  // Concatenate tensor arguments on the same axis as the original operation,
  // and concatenate scalars into the vector.
  if (is_all_tensors(0, axis) && is_all_scalars(1)) {
    std::array<int64_t, 1> rhs_dims{static_cast<int64_t>(op.values().size())};
    auto rhs_type = RankedTensorType::get(rhs_dims, ranked.getElementType());
    return HoistParams{args(0),
                       args(1),
                       axis,
                       0,
                       op.getType(),
                       rhs_type,
                       /*scalar_operand_idx=*/1};
  } else if (is_all_tensors(1, axis) && is_all_scalars(0)) {
    std::array<int64_t, 1> lhs_dims{static_cast<int64_t>(op.values().size())};
    auto lhs_type = RankedTensorType::get(lhs_dims, ranked.getElementType());
    return HoistParams{args(0),
                       args(1),
                       0,
                       axis,
                       lhs_type,
                       op.getType(),
                       /*scalar_operand_idx=*/0};
  }
  return None;
}

}  // namespace

void ConcatV2Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<HoistCwiseBinaryOutOfConcat, HoistCwiseUnaryOutOfConcat>(
      context);
}

//===----------------------------------------------------------------------===//
// CumsumOp and CumprodOp
//===----------------------------------------------------------------------===//

template <typename OpT, typename std::enable_if<llvm::is_one_of<
                            OpT, CumsumOp, CumprodOp>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  if (!IsOfRankOrUnranked(op.axis(), 0))
    return op.emitOpError("requires scalar axis operand");

  DenseIntElementsAttr axis_attr;
  if (matchPattern(op.axis(), m_Constant(&axis_attr))) {
    auto input_ty = op.x().getType().template dyn_cast<RankedTensorType>();
    if (input_ty) {
      int64_t rank = input_ty.getRank();
      assert(axis_attr.getNumElements() == 1 &&
             "scalar attribute should have exactly one element");
      int64_t axis = (*axis_attr.begin()).getSExtValue();
      if (axis < -rank || axis >= rank) {
        return op.emitError()
               << "axis operand should be within range [" << -rank << ", "
               << rank << "); actual value: " << axis;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOffsetOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ConcatOffsetOp op) {
  if (op.N() < 2)
    return op.emitOpError() << "requires N to be at least 2, got " << op.N();

  if (op.shape().size() != op.offset().size())
    return op.emitOpError()
           << "requires sizes of shapes and offsets to be the same, got sizes "
           << op.shape().size() << " and " << op.offset().size();

  auto ranked_dim = op.concat_dim().getType().dyn_cast<RankedTensorType>();
  if (ranked_dim && ranked_dim.getRank() != 0)
    return op.emitOpError()
           << "requires concat_dim to be a scalar, got tensor of rank "
           << ranked_dim.getRank();

  int64_t num_dims = -1;
  for (auto shape_offset_idx :
       llvm::enumerate(llvm::zip(op.shape(), op.offset()))) {
    Value shape = std::get<0>(shape_offset_idx.value());
    Value offset = std::get<1>(shape_offset_idx.value());
    const size_t idx = shape_offset_idx.index();

    if (failed(verifyCompatibleShape(shape.getType(), offset.getType())))
      return op.emitOpError() << "requires operand and result " << idx
                              << " to have compatible shapes";

    auto ranked_shape = shape.getType().dyn_cast<RankedTensorType>();
    if (!ranked_shape) continue;

    if (ranked_shape.getRank() != 1)
      return op.emitOpError() << "requires shape tensor operand " << idx
                              << " to be of rank 1, got tensor of rank "
                              << ranked_shape.getRank();

    if (!ranked_shape.hasStaticShape()) continue;

    int64_t ranked_shape_dim = ranked_shape.getDimSize(0);
    if (num_dims == -1)
      num_dims = ranked_shape_dim;
    else if (ranked_shape_dim != num_dims)
      return op.emitOpError()
             << "requires shape tensor (rank 1) operand " << idx
             << " to be of length " << num_dims
             << ", got tensor (rank 1) of length " << ranked_shape_dim;
  }

  return success();
}

LogicalResult ConcatOffsetOp::fold(ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
  // ConcatOffset must have its first operand be concat_dim and at least two
  // shape tensors in variadic shapes operand.
  if (operands.size() < 3) return failure();

  // Check concat_dim is a scalar.
  auto concat_dim_attr = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!concat_dim_attr || concat_dim_attr.getType().getRank() != 0)
    return failure();

  llvm::SmallVector<DenseIntElementsAttr, 4> shapes;
  shapes.reserve(operands.size() - 1);
  for (Attribute shape : llvm::drop_begin(operands, 1))
    if (auto shape_attr = shape.dyn_cast_or_null<DenseIntElementsAttr>())
      shapes.push_back(shape_attr);
    else
      return failure();

  // Check all shapes are vectors of the same length.
  if (shapes.front().getType().getRank() != 1) return success();
  const int64_t num_dims = shapes.front().getNumElements();
  for (DenseIntElementsAttr shape : llvm::drop_begin(shapes, 1))
    if (shape.getType().getRank() != 1 || shape.getNumElements() != num_dims)
      return failure();

  // Check concat_dim is within [-num_dims, num_dims).
  int32_t concat_dim = (*concat_dim_attr.getValues<int32_t>().begin());
  if (concat_dim < 0) concat_dim += num_dims;
  if (concat_dim >= num_dims || concat_dim < 0) return failure();

  // Check all elements besides at concat_dim match across all shape tensors.
  SmallVector<int32_t, 4> shape0;
  shape0.reserve(num_dims);
  for (int32_t dim : shapes.front().getValues<int32_t>()) shape0.push_back(dim);

  for (DenseIntElementsAttr shape : llvm::drop_begin(shapes, 1)) {
    for (auto dims_and_idx : llvm::enumerate(llvm::zip(shape0, shape))) {
      if (dims_and_idx.index() == concat_dim) continue;

      if (std::get<0>(dims_and_idx.value()) !=
          std::get<1>(dims_and_idx.value()).getSExtValue())
        return failure();
    }
  }

  // Compute an exclusive cumulative sum of elements at concat_dim.
  results.reserve(shapes.size());
  SmallVector<int32_t, 4> cumulative_sum(num_dims, 0);
  RankedTensorType offset_type =
      RankedTensorType::get({num_dims}, IntegerType::get(getContext(), 32));
  for (DenseIntElementsAttr shape : shapes) {
    results.push_back(DenseIntElementsAttr::get(offset_type, cumulative_sum));
    cumulative_sum[concat_dim] += shape.getValue<int32_t>(concat_dim);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`. The result
// op's type is deduced from `value`; if `value` is of scalar type,
// wraps it up with a tensor type of empty shape.
// TODO(jpienaar): This one differs from the autogenerated one as it takes an
// attribute but always creates an ElementsAttr internally.
void ConstOp::build(OpBuilder &builder, OperationState &result,
                    Attribute value) {
  ShapedType type;
  if (auto elem_attr = value.dyn_cast<ElementsAttr>()) {
    return ConstOp::build(builder, result, elem_attr);
  } else if (value.isa<BoolAttr, FloatAttr, IntegerAttr>()) {
    // All TensorFlow types must be tensor types. In the build() method,
    // we want to provide more flexibility by allowing attributes of scalar
    // types. But we need to wrap it up with ElementsAttr to construct
    // valid TensorFlow constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    return ConstOp::build(builder, result, DenseElementsAttr::get(type, value));
  }
  // TODO(jpienaar): support other TensorFlow specific types.
  llvm_unreachable("unsupported attribute type for building tf.Const");
}

void ConstOp::build(OpBuilder &builder, OperationState &result, Type type,
                    Attribute value) {
  // Handle the case where the type and value are already tensors.
  if (type.isa<TensorType>() && value.isa<ElementsAttr>()) {
    result.addTypes(type);
    result.addAttribute("value", value);
    return;
  }

  // Otherwise, default to the attribute builder.
  ConstOp::build(builder, result, value);
  assert(type == result.types[0] && "type mismatch in construction");
}

LogicalResult ConstOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto value = attributes.get("value");
  if (!value) return emitOptionalError(location, "missing attribute 'value'");
  if (auto elem_attr = value.dyn_cast<ElementsAttr>()) {
    inferredReturnTypes.assign({elem_attr.getType()});
    return success();
  }
  return emitOptionalError(location,
                           "attribute 'value' failed to satisfy constraint: "
                           "constant vector/tensor");
}

//===----------------------------------------------------------------------===//
// Conv2DOp and Conv3DOp
//===----------------------------------------------------------------------===//

static LogicalResult VerifyConvOpAttributes(
    int num_dims, ArrayRef<Attribute> strides, ArrayRef<Attribute> dilations,
    llvm::Optional<mlir::Location> location) {
  int64_t strides_size = strides.size();
  if (strides_size != num_dims)
    return emitOptionalError(
        location, "requires strides attribute length to be ", num_dims);
  auto is_not_positive = [](Attribute val) {
    return val.cast<IntegerAttr>().getValue().getSExtValue() <= 0;
  };
  if (llvm::any_of(strides, is_not_positive))
    return emitOptionalError(location, "requires positive strides");

  int64_t dilations_size = dilations.size();
  if (dilations_size != num_dims)
    return emitOptionalError(
        location, "requires dilations attribute length to be ", num_dims);
  if (llvm::any_of(dilations, is_not_positive))
    return emitOptionalError(location, "requires positive dilations");

  return success();
}

// Verifies that,
// * Number of input channels is divisible by the number of filter input
//   channels
template <typename OpT, typename std::enable_if<llvm::is_one_of<
                            OpT, Conv2DOp, Conv3DOp>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  int num_spatial_dims = std::is_same<OpT, Conv2DOp>() ? 2 : 3;
  int num_dims = 2 + num_spatial_dims;

  int64_t input_channels = -1;
  if (auto ty = op.input().getType().template dyn_cast<RankedTensorType>()) {
    absl::string_view data_format(op.data_format().data(),
                                  op.data_format().size());
    tensorflow::TensorFormat format;
    auto is_valid = FormatFromString(data_format, &format);
    DCHECK(is_valid) << data_format;
    int idx = tensorflow::GetTensorFeatureDimIndex(num_dims, format);
    input_channels = ty.getDimSize(idx);
  }

  int64_t filter_channels = -1;
  if (auto ty = op.filter().getType().template dyn_cast<RankedTensorType>()) {
    int idx = tensorflow::GetFilterTensorInputChannelsDimIndex(
        num_dims, tensorflow::FORMAT_HWIO);
    filter_channels = ty.getDimSize(idx);
  }

  if (input_channels != -1 && filter_channels != -1 &&
      input_channels % filter_channels != 0)
    return op.emitOpError()
           << "requires the number of input channels to be divisible by the "
              "number of filter input channels; found "
           << input_channels << " and " << filter_channels << ", respectively";

  return success();
}

LogicalResult Conv2DOp::UpdateDataFormat(StringRef data_format) {
  auto perm = GetDataFormatPermutation(this->data_format(), data_format);
  if (perm.empty()) return failure();

  // Update data_format attribute and result types.
  if (failed(::mlir::TF::UpdateDataFormat(data_format, this))) return failure();

  // Update convolution attributes.
  (*this)->setAttr("dilations", ShuffleArrayAttr(dilations(), perm));
  (*this)->setAttr("strides", ShuffleArrayAttr(strides(), perm));
  (*this)->setAttr("explicit_paddings",
                   ShuffleArrayAttr(explicit_paddings(), perm, 2));

  return success();
}

// Verifies the inferred return type of the given operation.
template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, Conv2DOpAdaptor, Conv3DOpAdaptor>::value>::type * = nullptr>
static LogicalResult inferConvReturnTypes(
    OpT op, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes,
    llvm::Optional<mlir::Location> location,
    ArrayRef<Attribute> explicit_padding) {
  const int64_t num_spatial_dims = std::is_same<OpT, Conv2DOpAdaptor>() ? 2 : 3;
  const int64_t num_dims = 2 + num_spatial_dims;
  const Value input = op.input();
  const Value filter = op.filter();
  const TensorType input_ty = input.getType().template cast<TensorType>();
  const TensorType filter_ty = filter.getType().template cast<TensorType>();
  const StringRef paddings = op.padding().getValue();

  ArrayRef<Attribute> strides = op.strides().getValue();
  StringRef data_format = op.data_format().getValue();
  ArrayRef<Attribute> dilations = op.dilations().getValue();

  tensorflow::TensorFormat format;
  auto data_format_is_valid = FormatFromString(data_format.str(), &format);
  if (!data_format_is_valid) {
    return emitOptionalError(location, "Invalid data format provided");
  }
  tensorflow::Padding padding;
  auto padding_is_valid = GetPaddingFromString(paddings.str(), &padding);
  if (!padding_is_valid.ok()) {
    return emitOptionalError(location, "Invalid padding format provided");
  }
  auto get_int = [](Attribute attr) {
    return attr.template cast<IntegerAttr>().getInt();
  };

  // Necessary sanity checks.
  // Verifies that,
  // * Ranks of operands and result are valid
  // * Length of explicit_paddings attribute is valid and has non negative
  //   elements
  // * strides and dilations attributes have positive elements
  if (!IsOfRankOrUnranked(input, num_dims) ||
      !IsOfRankOrUnranked(filter, num_dims))
    return emitOptionalError(location, "requires operands to be ", num_dims,
                             "D tensor");

  if (padding == tensorflow::Padding::EXPLICIT) {
    if (explicit_padding.size() == 0) {
      return emitOptionalError(location,
                               "requires attribute 'explicit_paddings' with "
                               "'EXPLICIT' padding mode");
    }
    if (explicit_padding.size() != num_dims * 2) {
      return emitOptionalError(
          location, "requires explicit_paddings attribute length to be ",
          num_dims * 2);
    }
    auto is_negative = [](Attribute val) {
      return val.cast<IntegerAttr>().getValue().getSExtValue() < 0;
    };
    if (llvm::any_of(explicit_padding, is_negative))
      return emitOptionalError(location,
                               "requires non negative explicit paddings");
  }

  if (failed(VerifyConvOpAttributes(num_dims, strides, dilations, location))) {
    return failure();
  }

  // Output always have `num_dims` rank. All dimensions are initialized to
  // dynamic size and can be partially inferred.
  SmallVector<int64_t, 4> return_shape(num_dims, ShapedType::kDynamicSize);
  // Output batch and channel dimension can be obtained using utilities from
  // tensorflow/core/util/tensor_format.h.
  if (input_ty.hasRank()) {
    return_shape[GetTensorBatchDimIndex(num_dims, format)] =
        input_ty.getDimSize(GetTensorBatchDimIndex(num_dims, format));
  }
  if (filter_ty.hasRank()) {
    return_shape[GetTensorFeatureDimIndex(num_dims, format)] =
        filter_ty.getDimSize(GetFilterTensorOutputChannelsDimIndex(
            num_dims, tensorflow::FORMAT_HWIO));
  }
  // Spatial dimensions can be inferred only when both input and filter are
  // ranked because we need to get their spatial dimensions.
  if (input_ty.hasRank() && filter_ty.hasRank()) {
    // Checks the size of each of the output spatial dimensions.
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, format, i);
      int64_t stride = get_int(strides[dim]);
      tensorflow::int64 expected_output_size;
      tensorflow::int64 pad_low;
      tensorflow::int64 pad_high;
      // Retrieve padding, if defined explicitly.
      if (padding == tensorflow::Padding::EXPLICIT) {
        pad_low = get_int(explicit_padding[2 * dim]);
        pad_high = get_int(explicit_padding[2 * dim + 1]);
      }
      // Skip if input or filter size is dynamic.
      if (input_ty.isDynamicDim(dim) || filter_ty.isDynamicDim(i)) continue;
      // Calculate the expected_output_size.
      tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
          input_ty.getDimSize(dim), filter_ty.getDimSize(i),
          get_int(dilations[dim]), stride, padding, &expected_output_size,
          &pad_low, &pad_high);
      // Return failure if expected_output_size could not be calculated.
      if (!status.ok()) return failure();
      return_shape[dim] = expected_output_size;
    }
  }

  inferredReturnTypes.assign(
      {RankedTensorType::get(return_shape, input_ty.getElementType())});
  return success();
}

LogicalResult Conv2DOp::inferReturnTypes(
    mlir::MLIRContext *context, llvm::Optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  Conv2DOpAdaptor op(operands, attributes);
  ArrayRef<Attribute> explicit_padding;
  ArrayAttr explicit_pad =
      attributes.get("explicit_paddings").dyn_cast_or_null<::mlir::ArrayAttr>();
  if (!explicit_pad) {
    explicit_pad = ::mlir::Builder(context).getI64ArrayAttr({});
  }
  explicit_padding = explicit_pad.getValue();

  return inferConvReturnTypes(op, inferredReturnTypes, location,
                              explicit_padding);
}

StringRef Conv2DOp::GetOptimalLayout(const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // Input must be a tensor.
  auto input_ty = input().getType().dyn_cast<TensorType>();
  if (!input_ty) return data_format();

  // For f16 data type on devices with Tensor Cores support NHWC data format
  // is up to ~2x faster.
  const bool is_f16 = input_ty.getElementType().isF16();
  if (is_f16 && CanUseTensorCores(devices)) return "NHWC";

  // For f32/f16 data type decision depends on the filter size in spatial
  // dimensions, for other data types we keep current data format.
  if (!input_ty.getElementType().isF32() && !input_ty.getElementType().isF16())
    return data_format();

  // Keep current data format if filter rank is unknown or not equal to 4.
  auto filter_ty = filter().getType().dyn_cast<RankedTensorType>();
  if (!filter_ty || filter_ty.getRank() != 4) return data_format();

  const int64_t d0 = filter_ty.getDimSize(0);
  const int64_t d1 = filter_ty.getDimSize(1);

  auto all_ones = [](ArrayAttr arr) -> bool {
    return llvm::all_of(arr, [](Attribute attr) -> bool {
      return attr.cast<IntegerAttr>().getInt() == 1;
    });
  };

  // Convolutions with 1x1 filter and with strides and dilations all ones, can
  // be computed as a GEMM in NHWC data format, and can be up to ~2x times
  // faster than convolution in NCHW.
  const bool one_by_one = d0 == 1 && d1 == 1;
  const bool trivial_strides = all_ones(strides());
  const bool trivial_dilations = all_ones(dilations());

  // TODO(ezhulenev): This might lead to excessive transposes in the final IR,
  // if the ratio of 1x1 convolutions to regular convolutions is close to 1:1.
  // Also FusedBatchNorm in training mode prefers NCHW data format. Check if all
  // users can efficiently use NHWC data format?
  if (one_by_one && trivial_strides && trivial_dilations) {
    return "NHWC";
  }

  // If filter spatial dimensions are unknown or not 1x1 we prefer NCHW, because
  // it's the fastest option on NVIDIA GPUs with cuDNN library support.
  return "NCHW";
}

//===----------------------------------------------------------------------===//
// Conv2dBackpropFilterOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DBackpropFilterOp::UpdateDataFormat(StringRef data_format) {
  StringRef src_data_format = this->data_format();

  auto perm = GetDataFormatPermutation(src_data_format, data_format);
  if (perm.empty()) return failure();

  // Update data_format attribute and result types.
  if (failed(::mlir::TF::UpdateDataFormat(data_format, this))) return failure();

  // Update convolution attributes.
  (*this)->setAttr("dilations", ShuffleArrayAttr(dilations(), perm));
  (*this)->setAttr("strides", ShuffleArrayAttr(strides(), perm));
  (*this)->setAttr("explicit_paddings",
                   ShuffleArrayAttr(explicit_paddings(), perm, 2));

  // Permute filter sizes operand.
  OpBuilder builder(getOperation());
  auto filter_sizes_permuted = builder.create<TF::DataFormatVecPermuteOp>(
      getLoc(), filter_sizes(), StringAttr::get(getContext(), src_data_format),
      StringAttr::get(getContext(), data_format));
  setOperand(1, filter_sizes_permuted);

  return success();
}

StringRef Conv2DBackpropFilterOp::GetOptimalLayout(
    const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // Input must be a tensor.
  auto input_ty = input().getType().dyn_cast<TensorType>();
  if (!input_ty) return data_format();

  // For f16 data type on devices with Tensor Cores support NHWC data format
  // is up to ~2x faster.
  const bool is_f16 = input_ty.getElementType().isF16();
  if (is_f16 && CanUseTensorCores(devices)) return "NHWC";

  // Otherwise always use "NCHW".
  return "NCHW";
}

//===----------------------------------------------------------------------===//
// Conv2DBackpropInputOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(Conv2DBackpropInputOp op) {
  int num_spatial_dims = 2;
  int num_dims = 2 + num_spatial_dims;

  if (!IsOfRankOrUnranked(op.out_backprop(), num_dims) ||
      !IsOfRankOrUnranked(op.filter(), num_dims))
    return op.emitOpError()
           << "requires operands to be " << num_dims << "D tensor";
  if (!IsOfRankOrUnranked(op.getResult(), num_dims))
    return op.emitOpError()
           << "requires result to be " << num_dims << "D tensor";

  llvm::Optional<mlir::Location> location = op.getLoc();
  ArrayRef<Attribute> strides = op.strides().getValue();
  ArrayRef<Attribute> dilations = op.dilations().getValue();
  LogicalResult verify_result =
      VerifyConvOpAttributes(num_dims, strides, dilations, location);
  if (failed(verify_result)) {
    return verify_result;
  }

  return success();
}

LogicalResult Conv2DBackpropInputOp::UpdateDataFormat(StringRef data_format) {
  StringRef src_data_format = this->data_format();

  auto perm = GetDataFormatPermutation(src_data_format, data_format);
  if (perm.empty()) return failure();

  // Update data_format attribute and result types.
  if (failed(::mlir::TF::UpdateDataFormat(data_format, this))) return failure();

  // Update convolution attributes.
  (*this)->setAttr("dilations", ShuffleArrayAttr(dilations(), perm));
  (*this)->setAttr("strides", ShuffleArrayAttr(strides(), perm));
  (*this)->setAttr("explicit_paddings",
                   ShuffleArrayAttr(explicit_paddings(), perm, 2));

  // Permute input sizes operand.
  OpBuilder builder(getOperation());
  auto input_sizes_permuted = builder.create<TF::DataFormatVecPermuteOp>(
      getLoc(), input_sizes(), StringAttr::get(getContext(), src_data_format),
      StringAttr::get(getContext(), data_format));
  setOperand(0, input_sizes_permuted);

  return success();
}

StringRef Conv2DBackpropInputOp::GetOptimalLayout(
    const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // Filter must be a tensor.
  auto filter_ty = filter().getType().dyn_cast<TensorType>();
  if (!filter_ty) return data_format();

  // For f16 data type on devices with Tensor Cores support NHWC data format
  // is up to ~2x faster.
  const bool is_f16 = filter_ty.getElementType().isF16();
  if (is_f16 && CanUseTensorCores(devices)) return "NHWC";

  // Otherwise always use "NCHW".
  return "NCHW";
}

//===----------------------------------------------------------------------===//
// Conv3DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv3DOp::inferReturnTypes(
    mlir::MLIRContext *context, llvm::Optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  Conv3DOpAdaptor op(operands, attributes);
  ArrayRef<Attribute> explicit_padding;
  ArrayAttr explicit_pad =
      attributes.get("explicit_paddings").dyn_cast_or_null<::mlir::ArrayAttr>();
  if (!explicit_pad) {
    explicit_pad = ::mlir::Builder(context).getI64ArrayAttr({});
  }
  explicit_padding = explicit_pad.getValue();

  return inferConvReturnTypes(op, inferredReturnTypes, location,
                              explicit_padding);
}

//===----------------------------------------------------------------------===//
// DataFormatVecPermuteOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DataFormatVecPermuteOp op) {
  auto input_ty = op.x().getType().dyn_cast<RankedTensorType>();
  if (!input_ty) return success();

  int rank = input_ty.getRank();
  if (rank != 1 && rank != 2)
    return op.emitOpError("requires input of rank 1 or 2");

  if (rank == 1) {
    int64_t dim0 = input_ty.getDimSize(0);
    if (dim0 != ShapedType::kDynamicSize && dim0 != 4 && dim0 != 2)
      return op.emitOpError("requires 1D input of size 4 or size 2");
  }

  if (rank == 2) {
    int64_t dim0 = input_ty.getDimSize(0);
    if (dim0 != ShapedType::kDynamicSize && dim0 != 4)
      return op.emitOpError(
          "requires first dimensions of 2D input to be of size 4");

    int64_t dim1 = input_ty.getDimSize(1);
    if (dim1 != ShapedType::kDynamicSize && dim1 != 2)
      return op.emitOpError(
          "requires second dimensions of 2D input to be of size 2");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<DivWithSqrtDivisor>(context);
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
  return IdentityArithmeticOpFolder<DivOp>(*this, operands);
}

//===----------------------------------------------------------------------===//
// DynamicStitchOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DynamicStitchOp op) {
  if (op.N() < 1) return op.emitOpError("requires attribute N with value >= 1");

  if (RankedTensorType out_ty = op.getType().dyn_cast<RankedTensorType>()) {
    if (out_ty.getRank() == 0) {
      return op.emitOpError("requires non scalar output");
    }
  }

  llvm::SmallDenseSet<int64_t, 8> index_values;
  bool all_indices_const = true;
  int32_t max_index = -1;
  llvm::Optional<SmallVector<int64_t, 4>> inferred_item_shape;
  for (auto it : llvm::zip(op.indices(), op.data())) {
    Value index = std::get<0>(it);

    DenseIntElementsAttr index_attr;
    if (matchPattern(index, m_Constant(&index_attr))) {
      for (int32_t index : index_attr.getValues<int32_t>()) {
        if (index < 0)
          return op.emitOpError()
                 << "requires non-negative index values; found " << index;
        max_index = std::max(index, max_index);
        index_values.insert(index);
      }
    } else {
      all_indices_const = false;
    }

    Value data = std::get<1>(it);
    RankedTensorType index_ty = index.getType().dyn_cast<RankedTensorType>();
    RankedTensorType data_ty = data.getType().dyn_cast<RankedTensorType>();
    if (!index_ty || !data_ty) continue;

    int64_t index_rank = index_ty.getRank();
    ArrayRef<int64_t> data_shape = data_ty.getShape();
    ArrayRef<int64_t> index_shape = index_ty.getShape();
    if (failed(mlir::verifyCompatibleShape(index_shape,
                                           data_shape.take_front(index_rank))))
      return op.emitOpError() << "requires shape of data with type " << data_ty
                              << " to have prefix matching with shape of the "
                                 "corresponding index type "
                              << index_ty;

    ArrayRef<int64_t> item_shape = data_shape.drop_front(index_rank);
    if (!inferred_item_shape) {
      inferred_item_shape = llvm::to_vector<4>(item_shape);
      continue;
    }

    if (failed(mlir::verifyCompatibleShape(item_shape, *inferred_item_shape)))
      return op.emitOpError() << "has inconsistent shaped data and index "
                                 "pairs; inferred item shapes ["
                              << llvm::makeArrayRef(*inferred_item_shape)
                              << "] and [" << item_shape << "] don't match";
    for (int i = 0, e = item_shape.size(); i < e; ++i) {
      int64_t &inferred_dim = (*inferred_item_shape)[i];
      int64_t dim = item_shape[i];
      if (ShapedType::isDynamic(inferred_dim)) inferred_dim = dim;
    }
  }

  // If all indices are constants, then verify that they cover all indices in
  // the range [0, max_index] and the output type is legal.
  if (all_indices_const) {
    for (int32_t i = 0; i <= max_index; i++) {
      if (!index_values.count(i))
        return op.emitOpError() << "missing index " << i;
    }

    if (inferred_item_shape) {
      SmallVector<int64_t, 4> expected_shape;
      expected_shape.push_back(max_index + 1);
      expected_shape.append(inferred_item_shape->begin(),
                            inferred_item_shape->end());

      auto out_ty = op.getType().cast<TensorType>();
      auto expected_out_ty =
          RankedTensorType::get(expected_shape, out_ty.getElementType());

      if (!AreCastCompatible({out_ty, expected_out_ty})) {
        return op.emitOpError() << "has invalid output type; should be "
                                   "compatible with inferred type "
                                << expected_out_ty;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EinsumOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * Arity of the op is at most two.
//
// TODO(hinsu): Verify einsum equation attribute.
static LogicalResult Verify(EinsumOp op) {
  if (op.N() > 2) {
    return op.emitOpError("supports at most two operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

OpFoldResult EmptyOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "empty op has one operand");

  Attribute attr = operands.front();
  if (!attr) return {};

  auto int_attr = attr.cast<DenseIntElementsAttr>();
  SmallVector<int64_t, 6> out_shape;
  for (const auto val : int_attr.getValues<int32_t>()) {
    out_shape.push_back(val);
  }

  auto type = getResult().getType().cast<ShapedType>();
  auto etype = type.getElementType();

  // We can not fold if the result is not static.
  if (!type.hasStaticShape()) return {};

  if (auto float_type = etype.dyn_cast<FloatType>()) {
    auto out_type = RankedTensorType::get(out_shape, float_type);
    return DenseElementsAttr::get(out_type,
                                  {APFloat(float_type.getFloatSemantics())});
  }

  if (auto int_type = etype.dyn_cast<IntegerType>()) {
    auto out_type = RankedTensorType::get(out_shape, etype);
    APInt val(int_type.getWidth(), 0, int_type.getSignedness());
    return DenseElementsAttr::get(out_type, val);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// EmptyTensorListOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(EmptyTensorListOp op) {
  if (!IsOfRankOrUnranked(op.element_shape(), 0) &&
      !IsOfRankOrUnranked(op.element_shape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }

  if (!IsOfRankOrUnranked(op.max_num_elements(), 0)) {
    return op.emitOpError("requires max_num_elements operand to be 0D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EnsureShapeOp
//===----------------------------------------------------------------------===//

OpFoldResult EnsureShapeOp::fold(llvm::ArrayRef<mlir::Attribute>) {
  ShapedType type = input().getType().dyn_cast<ShapedType>();
  if (!type || !type.hasRank()) return {};
  // If shape attribute equals input operand's type's shape, fold it to input.
  if (type.getShape() == shape()) return input();
  // Else retain to enable failing dynamically.
  return {};
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(EqualOp op) {
  // If we allow inputs to have incompatible type, then nothing to do.
  if (!op.incompatible_shape_error()) return success();

  // Otherwise, check inputs are broadcastable.
  return mlir::OpTrait::impl::verifyCompatibleOperandBroadcast(
      op.getOperation());
}

void EqualOp::build(OpBuilder &builder, OperationState &result, Value x,
                    Value y, BoolAttr incompatible_shape_error) {
  auto result_type = DeduceEqualCmpOpType(&builder, result.location, x, y,
                                          incompatible_shape_error);
  return build(builder, result, result_type, x, y, incompatible_shape_error);
}

//===----------------------------------------------------------------------===//
// ExpandDimsOp
//===----------------------------------------------------------------------===//

Type InferExpandDimsOpType(Value input, Value dim) {
  Type element_ty = input.getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);

  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  if (!input_ty) return unranked_ty;

  DenseIntElementsAttr dim_attr;
  if (!matchPattern(dim, m_Constant(&dim_attr)) ||
      dim_attr.getNumElements() != 1)
    return unranked_ty;
  int64_t dim_val = (*dim_attr.begin()).getSExtValue();
  int64_t input_rank = input_ty.getRank();

  if (dim_val < -input_rank - 1 || dim_val > input_rank + 1) return unranked_ty;
  if (dim_val < 0) dim_val += input_rank + 1;

  SmallVector<int64_t, 4> shape = llvm::to_vector<4>(input_ty.getShape());
  shape.insert(shape.begin() + dim_val, 1);
  return RankedTensorType::get(shape, element_ty);
}

void ExpandDimsOp::build(OpBuilder &builder, OperationState &result,
                         Value input, Value dim) {
  return build(builder, result, InferExpandDimsOpType(input, dim), input, dim);
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxArgsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxArgsOp op) {
  // TODO(fengliuai): moving the following to an utility method.
  const llvm::fltSemantics &semantics = op.min().getSemantics();
  float rmin, rmax;
  if (&semantics == &APFloat::IEEEsingle()) {
    rmin = op.min().convertToFloat();
    rmax = op.max().convertToFloat();
  } else {
    rmin = op.min().convertToDouble();
    rmax = op.max().convertToDouble();
  }
  // Range boundaries must be valid.
  if (rmin >= rmax) {
    return op.emitOpError("range is invalid: [" + Twine(std::to_string(rmin)) +
                          "," + Twine(std::to_string(rmax)) + "]");
  }
  int64_t num_bits = op.num_bits();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxVarsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxVarsOp op) {
  auto min = GetRankedTensorTypeForOperand(op.min());
  if (min && !IsOfRankedFloatTensorType(min, 0))
    return op.emitOpError("requires min to be a 0d float tensor");

  auto max = GetRankedTensorTypeForOperand(op.max());
  if (max && !IsOfRankedFloatTensorType(max, 0))
    return op.emitOpError("requires max to be a 0d float tensor");

  int64_t num_bits = op.num_bits();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxVarsPerChannelOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxVarsPerChannelOp op) {
  auto min = GetRankedTensorTypeForOperand(op.min());
  if (min && !IsOfRankedFloatTensorType(min, 1))
    return op.emitOpError("requires min to be a 1d float tensor");

  auto max = GetRankedTensorTypeForOperand(op.max());
  if (max && !IsOfRankedFloatTensorType(max, 1))
    return op.emitOpError("requires max to be a 1d float tensor");

  Value inputs = op.inputs();
  if (!HasRankAtLeast(inputs, 1))
    return op.emitError("requires inputs to be at least 1d float tensor");

  int64_t num_bits = op.num_bits();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }

  auto inputs_type = inputs.getType().dyn_cast<RankedTensorType>();
  if (!inputs_type) return success();
  int depth = inputs_type.getDimSize(inputs_type.getRank() - 1);
  if ((min && min.getDimSize(0) != depth) ||
      (max && max.getDimSize(0) != depth)) {
    return op.emitOpError(
        "requires min and max to have same size as last dimension of inputs");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(FillOp op) {
  if (!IsOfRankOrUnranked(op.dims(), 1))
    return op.emitOpError() << "requires dims to be a 1D tensor";
  if (!IsOfRankOrUnranked(op.value(), 0))
    return op.emitOpError() << "requires value to be a scalar";

  return success();
}

static ShapedType InferFillOpType(Value dims, Value value) {
  Type etype = value.getType().cast<ShapedType>().getElementType();

  DenseIntElementsAttr dims_attr;
  if (!matchPattern(dims, m_Constant(&dims_attr))) {
    return UnrankedTensorType::get(etype);
  }

  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(dims_attr.getNumElements());
  for (const APInt dim : dims_attr.getValues<APInt>()) {
    shape.push_back(dim.getSExtValue());
  }
  return RankedTensorType::get(shape, etype);
}

void FillOp::build(OpBuilder &builder, OperationState &result, Value dims,
                   Value value) {
  FillOp::build(builder, result, InferFillOpType(dims, value), dims, value);
}

OpFoldResult FillOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "fill op has two operand");

  auto type = getType().cast<ShapedType>();
  // DenseElementsAttr that is used in this folder only supports int and float
  // types.
  // TODO(hinsu): Handle complex types once there is a attribute kind for
  // complex.
  if (!type.getElementType().isIntOrFloat()) return {};

  auto value = operands[1].dyn_cast_or_null<ElementsAttr>();
  if (!value) return {};

  if (type.hasStaticShape())
    return DenseElementsAttr::get(type, value.getValue({}));

  auto dims = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!dims) return {};

  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(dims.getNumElements());
  for (const APInt dim : dims.getValues<APInt>()) {
    shape.push_back(dim.getSExtValue());
  }
  type = RankedTensorType::get(shape, type.getElementType());

  return DenseElementsAttr::get(type, value.getValue({}));
}

//===----------------------------------------------------------------------===//
// FusedBatchNormGradOp
//===----------------------------------------------------------------------===//

// TODO(b/150954845): Add benchmarks to verify that layout preference didn't
// change in the latest GPU generations.

LogicalResult FusedBatchNormGradV3Op::UpdateDataFormat(StringRef data_format) {
  return ::mlir::TF::UpdateDataFormat(data_format, this);
}

StringRef FusedBatchNormGradV3Op::GetOptimalLayout(
    const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // For f16 data type on devices with Tensor Cores support NHWC data format
  // is up to ~2x faster.
  auto x_ty = x().getType().cast<TensorType>();
  const bool is_f16 = x_ty.getElementType().isF16();
  if (is_f16 && CanUseTensorCores(devices)) return "NHWC";

  // For all other data types prefer NCHW.
  return "NCHW";
}

//===----------------------------------------------------------------------===//
// FusedBatchNormOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(FusedBatchNormOp op) {
  auto x = GetRankedTensorTypeForOperand(op.x());
  if (x && !IsOfRankedFloatTensorType(x, 4))
    return op.emitOpError("requires x to be a 4D float tensor");

  auto scale = GetRankedTensorTypeForOperand(op.scale());
  if (scale && !IsOfRankedFloatTensorType(scale, 1))
    return op.emitOpError("requires scale to be a 1D float tensor");

  auto offset = GetRankedTensorTypeForOperand(op.offset());
  if (offset && !IsOfRankedFloatTensorType(offset, 1))
    return op.emitOpError("requires offset to be a 1D float tensor");

  auto mean = GetRankedTensorTypeForOperand(op.mean());
  if (mean && !IsOfRankedFloatTensorType(mean, 1))
    return op.emitOpError("requires mean to be a 1D float tensor");

  auto variance = GetRankedTensorTypeForOperand(op.variance());
  if (variance && !IsOfRankedFloatTensorType(variance, 1))
    return op.emitOpError("requires variance to be a 1D float tensor");

  // TODO(antiagainst): check attributes

  return success();
}

//===----------------------------------------------------------------------===//
// FusedBatchNormV2Op / FusedBatchNormV3Op
//===----------------------------------------------------------------------===//

template <class Op>
static LogicalResult InferenceFoldOperandsPermutation(
    ArrayRef<int64_t> permutation, Op *op) {
  // FusedBatchNorm in training mode is a layout sentitive operation, and should
  // have already assigned an optimal data format.
  if (op->is_training()) return failure();
  return ::mlir::TF::FoldOperandsPermutation(permutation, op);
}

template <class Op>
static StringRef GetOptimalLayout(const RuntimeDevices &devices, Op *op) {
  // In inference mode FusedBatchNorm is not sensitive to data layout.
  if (!op->is_training()) return op->data_format();

  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(op->getOperation()))
    return op->data_format();

  // For f16 data type on devices with Tensor Cores support NHWC data format
  // is up to ~2x faster.
  auto x_ty = op->x().getType().template cast<TensorType>();
  const bool is_f16 = x_ty.getElementType().isF16();
  if (is_f16 && CanUseTensorCores(devices)) return "NHWC";

  // For all other data types prefer NCHW.
  return "NCHW";
}

LogicalResult FusedBatchNormV2Op::FoldOperandsPermutation(
    ArrayRef<int64_t> permutation) {
  return ::mlir::TF::InferenceFoldOperandsPermutation(permutation, this);
}

LogicalResult FusedBatchNormV2Op::UpdateDataFormat(StringRef data_format) {
  return ::mlir::TF::UpdateDataFormat(data_format, this);
}

StringRef FusedBatchNormV2Op::GetOptimalLayout(const RuntimeDevices &devices) {
  return ::mlir::TF::GetOptimalLayout(devices, this);
}

LogicalResult FusedBatchNormV3Op::FoldOperandsPermutation(
    ArrayRef<int64_t> permutation) {
  return ::mlir::TF::InferenceFoldOperandsPermutation(permutation, this);
}

LogicalResult FusedBatchNormV3Op::UpdateDataFormat(StringRef data_format) {
  return ::mlir::TF::UpdateDataFormat(data_format, this);
}

StringRef FusedBatchNormV3Op::GetOptimalLayout(const RuntimeDevices &devices) {
  return ::mlir::TF::GetOptimalLayout(devices, this);
}

//===----------------------------------------------------------------------===//
// GatherV2Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(GatherV2Op op) {
  int64_t batch_dims = op.batch_dims();
  if (auto ty = op.indices().getType().dyn_cast<RankedTensorType>()) {
    int64_t rank = ty.getRank();
    if (batch_dims > rank || batch_dims < -rank)
      return op.emitOpError()
             << "batch_dims (" << batch_dims << ") must be in range [" << -rank
             << ", " << rank + 1 << ")";
    if (batch_dims < 0) batch_dims += rank;
  }

  if (!HasRankAtMost(op.axis(), 1))
    return op.emitOpError("requires axis to have rank at most 1");

  DenseIntElementsAttr axis_attr;
  if (matchPattern(op.axis(), m_Constant(&axis_attr))) {
    int64_t axis = (*axis_attr.begin()).getSExtValue();
    if (auto ty = op.params().getType().dyn_cast<RankedTensorType>()) {
      int64_t rank = ty.getRank();
      if (axis >= rank || axis < -rank)
        return op.emitOpError() << "axis (" << axis << ") must be in range ["
                                << -rank << ", " << rank << ")";
      if (axis < 0) axis += rank;
    }

    if (batch_dims >= 0 && axis >= 0 && axis < batch_dims) {
      return op.emitOpError() << "requires axis (" << axis
                              << ") to be greater than or equal to batch_dims ("
                              << batch_dims << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verifySymbolUses(SymbolTableCollection &symbol_table) {
  auto branch_name = [](unsigned index) -> std::string {
    return index == 0 ? "'then_branch'" : "'else_branch'";
  };
  // TODO(jpienaar): Remove.
  if (failed(IfOpAdaptor(*this).verify(getLoc()))) return failure();
  return VerifyCaseOrIfOpBranchFunctions(
      symbol_table, *this, {then_branchAttr(), else_branchAttr()}, branch_name);
}

//===----------------------------------------------------------------------===//
// IfOp canonicalization.
//===----------------------------------------------------------------------===//

namespace {
class FoldConstantIfOp : public OpRewritePattern<TF::IfOp> {
 public:
  explicit FoldConstantIfOp(MLIRContext *context)
      : OpRewritePattern<TF::IfOp>(context) {}
  LogicalResult matchAndRewrite(TF::IfOp op,
                                PatternRewriter &rewriter) const override;

 private:
  template <typename T>
  struct CallOpType {
    using CallOp = T;
  };
};

LogicalResult FoldConstantIfOp::matchAndRewrite(
    TF::IfOp op, PatternRewriter &rewriter) const {
  // Extract the constant cond value.
  DenseIntElementsAttr cond_attr;
  if (!matchPattern(op.cond(), m_Constant(&cond_attr))) return failure();

  // Cond value must be a scalar.
  if (cond_attr.getNumElements() != 1) return failure();

  // Select a branch function.
  bool cond = cond_attr.getSplatValue<BoolAttr>().getValue();
  FlatSymbolRefAttr func = cond ? op.then_branchAttr() : op.else_branchAttr();

  // Replace IfOp with PartitionedCallOp or StatefulPartitionedCallOp.
  auto rewrite = [&](auto op_type) {
    auto empty = rewriter.getStringAttr("");
    auto call_op = rewriter.create<typename decltype(op_type)::CallOp>(
        op.getLoc(), op.getResultTypes(), op.input(), func,
        /*config=*/empty, /*config_proto=*/empty, /*executor_type=*/empty);
    CopyDeviceAndUnderscoredAttributes(op.getOperation(), call_op);
    rewriter.replaceOp(op, call_op.getResults());
  };

  if (op.is_stateless())
    rewrite(CallOpType<PartitionedCallOp>{});
  else
    rewrite(CallOpType<StatefulPartitionedCallOp>{});

  return success();
}
}  // anonymous namespace

void IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *context) {
  results.insert<FoldConstantIfOp, DropAttributes<IfOp>>(context);
}

//===----------------------------------------------------------------------===//
// IfRegionOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(IfRegionOp op) {
  TypeRange then_types =
      op.then_branch().front().getTerminator()->getOperandTypes();
  TypeRange else_types =
      op.else_branch().front().getTerminator()->getOperandTypes();

  TypeRangeWithDesc results{op.getResultTypes(), "result"};
  TypeRangeWithDesc then_results{then_types, "then result"};
  TypeRangeWithDesc else_results{else_types, "else result"};

  if (failed(VerifyTypeRangesAreCompatible(op, then_results, results)))
    return failure();
  if (failed(VerifyTypeRangesAreCompatible(op, else_results, results)))
    return failure();
  return success();
}

namespace {
class FoldConstantIfRegionOp : public OpRewritePattern<TF::IfRegionOp> {
 public:
  explicit FoldConstantIfRegionOp(MLIRContext *context)
      : OpRewritePattern<TF::IfRegionOp>(context) {}
  LogicalResult matchAndRewrite(TF::IfRegionOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult FoldConstantIfRegionOp::matchAndRewrite(
    TF::IfRegionOp op, PatternRewriter &rewriter) const {
  // Extract the constant cond value.
  DenseIntElementsAttr cond_attr;
  if (!matchPattern(op.cond(), m_Constant(&cond_attr))) return failure();

  // IfRegion condition should always be a scalar. Select the region to fold to.
  bool cond = cond_attr.getSplatValue<BoolAttr>().getValue();
  Region &region = cond ? op.then_branch() : op.else_branch();

  // If the IfRegion is stateless but the region being inlined itself is not
  // stateless, then inlining the region could cause a loss of information.
  // However, its probably better to fold the IfRegion instead of having the
  // dead branch stay.

  // Inline the region in place of the IfRegion op, and forward the yield
  // inputs to the IfRegion op results. This is possible only if the yield
  // types match the result types.
  auto yield = cast<YieldOp>(region.front().getTerminator());
  auto updated_results = llvm::to_vector<4>(yield.getOperands());

  // If the yield types do not match the IfRegion result types, add appropriate
  // casts.
  rewriter.setInsertionPoint(yield);
  for (auto it : llvm::zip(op.getResultTypes(), updated_results)) {
    auto &updated_result = std::get<1>(it);
    Type result_type = std::get<0>(it);
    if (result_type != updated_result.getType()) {
      updated_result =
          rewriter.create<TF::CastOp>(op.getLoc(), result_type, updated_result,
                                      /*Truncate=*/rewriter.getBoolAttr(false));
    }
  }
  // Inline the region into the block containing the IfRegion.
  rewriter.mergeBlockBefore(&region.front(), op);
  rewriter.eraseOp(yield);
  rewriter.replaceOp(op, updated_results);
  return success();
}
}  // anonymous namespace

void IfRegionOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldConstantIfRegionOp,
                 CaseOrIfRegionEliminatePassThrough<TF::IfRegionOp>>(context);
}

//===----------------------------------------------------------------------===//
// InvertPermutationOp
//===----------------------------------------------------------------------===//

// Verifies that the input is 1D.
static LogicalResult Verify(InvertPermutationOp op) {
  auto x_type = op.x().getType().cast<TensorType>();
  if (!x_type.hasRank()) return success();
  if (x_type.getShape().size() != 1)
    return op.emitOpError() << "requires input x to be 1-dimensional";

  return success();
}

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//

OpFoldResult LeakyReluOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "leaky relu has one operand");

  // leaky_relu(x, alpha: 1) -> x
  if (alpha().convertToFloat() == 1.0f) return getOperand();

  auto calculate = [&](FloatAttr arg) {
    APFloat val = arg.getValue();
    if (val.isNegative()) val = alpha() * val;
    return FloatAttr::get(arg.getType(), val);
  };

  if (auto arg = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return calculate(arg);
  } else if (auto arg = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    if (auto elementAttr = arg.getSplatValue().dyn_cast<FloatAttr>())
      return DenseElementsAttr::get(arg.getType(), calculate(elementAttr));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

void LogOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<LogOfSoftmax, LogToLog1p>(context);
}

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

void LogicalNotOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<LogicalNotOfEqual, LogicalNotOfNotEqual, LogicalNotOfGreater,
                 LogicalNotOfGreaterEqual, LogicalNotOfLess,
                 LogicalNotOfLessEqual>(context);
}

//===----------------------------------------------------------------------===//
// MatrixBandPartOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(MatrixBandPartOp op) {
  if (!HasRankAtLeast(op.input(), 2)) {
    return op.emitOpError()
           << "requires `input` to have rank of at least 2, but found "
           << op.input().getType();
  }
  if (!IsOfRankOrUnranked(op.num_lower(), 0)) {
    return op.emitOpError()
           << "requires `num_lower` to have 0 dimensions, but found "
           << op.num_lower().getType();
  }
  if (!IsOfRankOrUnranked(op.num_upper(), 0)) {
    return op.emitOpError()
           << "requires `num_upper` to have 0 dimensions, but found "
           << op.num_upper().getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatrixSetDiagOp
//===----------------------------------------------------------------------===//
//
void MatrixSetDiagOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MatrixSetDiagToV3>(context);
}

//===----------------------------------------------------------------------===//
// MatrixSetDiagV2Op
//===----------------------------------------------------------------------===//

void MatrixSetDiagV2Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MatrixSetDiagV2ToV3>(context);
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

void MaxOp::build(OpBuilder &builder, OperationState &result, Value input,
                  Value reduction_indices, BoolAttr keep_dims) {
  Type out_ty =
      InferReductionOpType(input, reduction_indices, keep_dims, &builder);
  build(builder, result, out_ty, input, reduction_indices, keep_dims);
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

LogicalResult MaxPoolOp::FoldOperandsPermutation(
    ArrayRef<int64_t> permutation) {
  return ::mlir::TF::FoldOperandsPermutation(
      permutation, this, {{"strides", strides()}, {"ksize", ksize()}});
}

LogicalResult MaxPoolOp::UpdateDataFormat(StringRef new_data_format) {
  StringRef src_data_format = data_format();

  auto perm = GetDataFormatPermutation(src_data_format, new_data_format);
  if (perm.empty()) return failure();

  // Update data_format attribute and result types.
  if (failed(::mlir::TF::UpdateDataFormat(new_data_format, this)))
    return failure();

  stridesAttr(ShuffleArrayAttr(strides(), perm));
  explicit_paddingsAttr(ShuffleArrayAttr(explicit_paddings(), perm, 2));
  ksizeAttr(ShuffleArrayAttr(ksize(), perm));

  return success();
}

StringRef MaxPoolOp::GetOptimalLayout(const RuntimeDevices &devices) {
  // Keep current data format if no GPUs are available or if explicit placement
  // does not allow to use GPU for this operation.
  if (!CanUseGpuDevice(devices) || !CanUseGpuDevice(getOperation()))
    return data_format();

  // Defaults to NCHW.
  return "NCHW";
}

//===----------------------------------------------------------------------===//
// MaxPoolGradOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(MaxPoolGradOp op) {
  if (!IsOfRankOrUnranked(op.orig_input(), 4)) {
    return op.emitOpError() << "requires orig_input to be rank 4";
  }
  if (!IsOfRankOrUnranked(op.orig_output(), 4)) {
    return op.emitOpError() << "requires orig_output to be rank 4";
  }
  if (!IsOfRankOrUnranked(op.grad(), 4)) {
    return op.emitOpError() << "requires grad to be rank 4";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

LogicalResult MeanOp::FoldOperandsPermutation(ArrayRef<int64_t> permutation) {
  // Reduction indices must be defined by a constant operation.
  auto reduction_op =
      dyn_cast_or_null<TF::ConstOp>(reduction_indices().getDefiningOp());
  if (!reduction_op) return failure();

  auto reductions_value = reduction_op.value().dyn_cast<DenseElementsAttr>();
  if (!reductions_value) return failure();

  // Prepare new reduction indices according to operand permutation.
  SmallVector<int32_t, 4> shuffled_reduction;
  llvm::transform(reductions_value.getIntValues(),
                  std::back_inserter(shuffled_reduction),
                  [&](APInt idx) { return permutation[idx.getSExtValue()]; });

  // Add constant operation with a new reduction indices.
  OpBuilder builder(getOperation());
  auto type = mlir::RankedTensorType::get(shuffled_reduction.size(),
                                          builder.getIntegerType(32));
  auto values = mlir::DenseIntElementsAttr::get(type, shuffled_reduction);
  auto shuffled_reduction_op = builder.create<TF::ConstOp>(getLoc(), values);

  // Use new reduction indices.
  setOperand(1, shuffled_reduction_op);

  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  return IdentityArithmeticOpFolder<MulOp>(*this, operands);
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.cc.inc"
