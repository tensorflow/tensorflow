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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gather.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

std::optional<bool> IsGatherLegal(mhlo::GatherOp op) { return std::nullopt; }

class LegalizeGatherToSlice : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeGatherToSlice::matchAndRewrite(
    mhlo::GatherOp gather_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
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
  if (!operand_type.hasStaticShape() || !start_indices_type.hasStaticShape() ||
      !result_type.hasStaticShape()) {
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
    if (offset_index >= offset_dims.size() || offset_dims[offset_index] != i) {
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
  if (failed(NormalizeIndexVector(gather_op, start_indices, start_indices_type,
                                  index_vector_dim, rewriter))) {
    return failure();
  }

  ImplicitLocOpBuilder builder(gather_op.getLoc(), rewriter);
  // Clamp the start indices to ensure it is in bounds.
  auto max_start_indices = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter,
      llvm::SmallVector<int64_t>(
          {operand_type.getDimSize(0) - slice_sizes_vector[0],
           operand_type.getDimSize(1) - slice_sizes_vector[1]}),
      start_indices_type.getElementType());
  auto min_start_indices = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter, llvm::SmallVector<int64_t>({0, 0}),
      start_indices_type.getElementType());
  auto start_indices_max_op = rewriter.create<TFL::MaximumOp>(
      gather_op.getLoc(), start_indices, min_start_indices);
  auto clamped_start_indices_op = rewriter.create<TFL::MinimumOp>(
      gather_op.getLoc(), start_indices_max_op, max_start_indices);

  int64_t batch_size = start_indices_type.getDimSize(batch_dim);
  auto slice_size = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter, slice_sizes_vector, rewriter.getI32Type());
  if (batch_size == 1) {
    auto squeeze_op = rewriter.create<TFL::SqueezeOp>(
        gather_op.getLoc(),
        RankedTensorType::get({rank_two}, start_indices_type.getElementType()),
        clamped_start_indices_op,
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
    auto slice_op =
        rewriter.create<TFL::SliceOp>(gather_op.getLoc(), gather_op.getType(),
                                      operand, squeeze_op, slice_size);
    rewriter.replaceOp(gather_op, slice_op);
    return mlir::success();
  }

  llvm::SmallVector<Value, 1> slices;
  slices.reserve(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    auto zero = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter, llvm::SmallVector<int64_t>({i, 0}),
        rewriter.getI32Type());
    auto two = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter, llvm::SmallVector<int64_t>({1, 2}),
        rewriter.getI32Type());
    // TODO maybe cast to i32 here
    auto begin = rewriter.create<TFL::SliceOp>(
        gather_op.getLoc(),
        RankedTensorType::get({1, 2}, start_indices_type.getElementType()),
        clamped_start_indices_op, zero, two);
    // TODO maybe cast to i32 here
    auto squeeze_op = rewriter.create<TFL::SqueezeOp>(
        gather_op.getLoc(),
        RankedTensorType::get({rank_two}, start_indices_type.getElementType()),
        begin, rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
    auto slice_op = rewriter.create<TFL::SliceOp>(
        gather_op.getLoc(),
        RankedTensorType::get({1, slice_sizes_vector[1]},
                              operand_type.getElementType()),
        operand, squeeze_op, slice_size);
    slices.push_back(slice_op);
  }
  auto concat_op = rewriter.create<TFL::ConcatenationOp>(
      gather_op.getLoc(), result_type, slices, 0,
      rewriter.getStringAttr("NONE"));
  rewriter.replaceOp(gather_op, concat_op);
  return mlir::success();
}

namespace {

// Transform the canonicalized result produced by tf.GatherNd with the
// canonicalized operand and start indices back into the original result.
// The canonicalized result will have the start indices batching dimensions
// flattened as leading dimension, and the offset dimensions as trailing
// dimensions. To transform back, we:
// - Unflatten the start indices batching dimensions.
// - Transpose dimensions back based on `offset_dims` and
//   `start_indices_batching_dims`.
Value UncanonicalizeResult(mhlo::GatherOp gather_op, Value canonical_result,
                           ShapedType canonical_result_type,
                           ShapedType original_result_type,
                           ArrayRef<int64_t> offset_dims,
                           ArrayRef<int64_t> start_indices_batching_dims,
                           ConversionPatternRewriter& rewriter) {
  // For those dims NOT inside the original_offset_dims are considered "batch
  // dims".
  std::vector<int64_t> batch_dims;
  // Offset dims are guaranteed to be sorted.
  int offset_index = 0;
  for (int64_t i = 0; i < original_result_type.getRank(); ++i) {
    if (offset_index >= offset_dims.size() || offset_dims[offset_index] != i) {
      batch_dims.push_back(i);
    } else {
      ++offset_index;
    }
  }

  // Determine the canonical shape after unflattening the start indices
  // batching dimensions (if they exist), and the permutation to transform
  // the original shape to the unflattened canonical shape.
  llvm::SmallVector<int64_t> permutation_to_canonical;
  llvm::SmallVector<int64_t> unflattened_shape;
  for (int64_t i : start_indices_batching_dims) {
    int64_t dim = batch_dims[i];
    permutation_to_canonical.push_back(dim);
    unflattened_shape.push_back(original_result_type.getDimSize(dim));
  }
  for (int64_t i = 0; i < batch_dims.size(); ++i) {
    if (llvm::count(start_indices_batching_dims, i) == 0) {
      int64_t dim = batch_dims[i];
      permutation_to_canonical.push_back(dim);
      unflattened_shape.push_back(original_result_type.getDimSize(dim));
    }
  }
  for (int64_t dim : offset_dims) {
    permutation_to_canonical.push_back(dim);
    unflattened_shape.push_back(original_result_type.getDimSize(dim));
  }

  // Unflatten the canonical result if necessary, and transpose back to the
  // original result shape.
  if (!start_indices_batching_dims.empty()) {
    auto unflattened_result_type = RankedTensorType::get(
        unflattened_shape, original_result_type.getElementType());
    canonical_result = rewriter.create<mhlo::ReshapeOp>(
        gather_op.getLoc(), unflattened_result_type, canonical_result);
  }
  return rewriter.create<mhlo::TransposeOp>(
      gather_op.getLoc(), original_result_type, canonical_result,
      rewriter.getI64TensorAttr(
          GetInversePermutationArray(permutation_to_canonical)));
}

// Canonicalize `operand` to handle operand batching dimensions and non-iota
// start index map, so it can be used by tf.GatherNd:
// - Transpose so that the leading dimensions are the operand batching
//   dimensions followed by the indexed dimensions (in order).
// - Flatten the batching dimensions.
Value CanonicalizeOperand(mhlo::GatherOp gather_op, Value operand,
                          ShapedType operand_type,
                          ArrayRef<int64_t> operand_batching_dims,
                          ArrayRef<int64_t> start_index_map,
                          ConversionPatternRewriter& rewriter) {
  int batch_size = 1;
  llvm::SmallVector<int64_t> permutation;
  llvm::SmallVector<int64_t> transposed_shape;
  llvm::SmallVector<int64_t> flattened_shape;
  // First add the batching dimensions.
  for (int64_t batch_dim : operand_batching_dims) {
    permutation.push_back(batch_dim);
    transposed_shape.push_back(operand_type.getDimSize(batch_dim));
    batch_size *= operand_type.getDimSize(batch_dim);
  }
  if (!operand_batching_dims.empty()) {
    flattened_shape.push_back(batch_size);
  }
  // Add the indexed dimensions.
  for (int64_t s : start_index_map) {
    permutation.push_back(s);
    transposed_shape.push_back(operand_type.getDimSize(s));
    flattened_shape.push_back(operand_type.getDimSize(s));
  }
  // Finally, add the remaining dimensions.
  for (int64_t i = 0; i < operand_type.getRank(); i++) {
    if (llvm::count(operand_batching_dims, i) == 0 &&
        llvm::count(start_index_map, i) == 0) {
      permutation.push_back(i);
      transposed_shape.push_back(operand_type.getDimSize(i));
      flattened_shape.push_back(operand_type.getDimSize(i));
    }
  }

  // Transpose the dimensions and flatten the batching dimensions.
  RankedTensorType transposed_type =
      RankedTensorType::get(transposed_shape, operand_type.getElementType());
  auto transposed_operand = rewriter.create<mhlo::TransposeOp>(
      gather_op.getLoc(), transposed_type, operand,
      rewriter.getI64TensorAttr(permutation));
  auto flattened_type =
      RankedTensorType::get(flattened_shape, operand_type.getElementType());
  auto flattened_operand = rewriter.create<mhlo::ReshapeOp>(
      gather_op.getLoc(), flattened_type, transposed_operand);
  return flattened_operand;
}

// Canonicalize `start_indices` to handle start indices batching dimensions so
// it can be used by tf.GatherNd:
// - Transpose so that the batching dimensions are the leading dimensions.
// - Flatten the batching dimensions if they exist.
// - Add iota index values for the operand batching dimensions.
Value CanonicalizeStartIndices(mhlo::GatherOp gather_op, Value start_indices,
                               ShapedType start_indices_type,
                               ArrayRef<int64_t> start_indices_batching_dims,
                               ConversionPatternRewriter& rewriter) {
  if (start_indices_batching_dims.empty()) {
    // Don't need to do anything if there are no batching dimensions. This
    // assumes that `index_vector_dim` is already the last dimension.
    return start_indices;
  }
  int batch_size = 1;
  llvm::SmallVector<int64_t> permutation;
  llvm::SmallVector<int64_t> transposed_shape;
  llvm::SmallVector<int64_t> flattened_shape;
  // First add the batching dimensions.
  for (int64_t batch_dim : start_indices_batching_dims) {
    permutation.push_back(batch_dim);
    transposed_shape.push_back(start_indices_type.getDimSize(batch_dim));
    batch_size *= start_indices_type.getDimSize(batch_dim);
  }
  flattened_shape.push_back(batch_size);
  // Add remaining dimensions.
  for (int64_t i = 0; i < start_indices_type.getRank(); i++) {
    if (llvm::count(start_indices_batching_dims, i) == 0) {
      permutation.push_back(i);
      transposed_shape.push_back(start_indices_type.getDimSize(i));
      flattened_shape.push_back(start_indices_type.getDimSize(i));
    }
  }

  // Transpose the dimensions and flatten the batching dimensions.
  auto transposed_start_indices = rewriter.create<mhlo::TransposeOp>(
      gather_op.getLoc(),
      RankedTensorType::get(transposed_shape,
                            start_indices_type.getElementType()),
      start_indices, rewriter.getI64TensorAttr(permutation));
  auto flattened_start_indices = rewriter.create<mhlo::ReshapeOp>(
      gather_op.getLoc(),
      RankedTensorType::get(flattened_shape,
                            start_indices_type.getElementType()),
      transposed_start_indices);

  // Concat iota values for indexing into the batching dimensions of the
  // operand.
  llvm::SmallVector<int64_t> offsets_shape = flattened_shape;
  offsets_shape.back() = 1;
  auto offsets = rewriter.create<mhlo::IotaOp>(
      gather_op.getLoc(),
      RankedTensorType::get(offsets_shape, start_indices_type.getElementType()),
      rewriter.getI64IntegerAttr(0));

  llvm::SmallVector<int64_t> new_start_indices_shape = flattened_shape;
  new_start_indices_shape.back()++;
  auto new_start_indices = rewriter.create<mhlo::ConcatenateOp>(
      gather_op.getLoc(),
      RankedTensorType::get(new_start_indices_shape,
                            start_indices_type.getElementType()),
      ValueRange{offsets, flattened_start_indices},
      rewriter.getI32IntegerAttr(new_start_indices_shape.size() - 1));

  return new_start_indices;
}
}  // namespace

// Tries to convert an mhlo::GatherOp into a TFL::GatherNdOp.
//
// Consider the following example:
//  operand_shape = [B1, I1, O1, B2, I2, O2]
//  operand_batching_dims = [0, 3]
//
//  start_indices_shape = [B2, B3, B1, 2]
//  start_indices_batching_dims = [3, 0]
//  index_vector_dim = 3
//  start_index_map = [4, 1]
//
//  offset_dims: [2, 4]
//  slice_sizes = [1, 1, O1, 1, 1, O2]
//  collapsed_slice_dims = [1, 4]
//  result_shape = [B2, B3, O1, B3, O2]
//
// To implement this with a tfl.GatherNd, we canonicalize the operand s.t. the
// operand batching dimensions are flattened into the leading dimensions,
// followed by the indexed dimensions in order:
//  canonical_operand_shape = [B1 * B2, I2, I1, O1, O2]
//
// We canonicalize the start indices so the start indices batching dimensions
// are flattened (in order) into a leading dimension. In addition, we add iota
// indices to appropriately offset into the flattened operand batching
// dimension:
//  canonical_start_indices_shape = [B1 * B2, B3, 3]
//    (index_vector_dim is expanded to included indices for the operand
//     batching dimensions)
//
// The result of tf.GatherNd(canonical_operand, canonical_start_indices) has the
// following shape:
//  canonical_result_shape = [B1 * B2, B3, O1, O2]
//
// The canonical result is unflattened and transpose as needed to get back to
// the original result shape.
class LegalizeGatherToGatherND : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeGatherToGatherND::matchAndRewrite(
    mhlo::GatherOp gather_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
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

  llvm::ArrayRef<int64_t> operand_batching_dims =
      gather_op.getDimensionNumbers().getOperandBatchingDims();
  llvm::ArrayRef<int64_t> start_indices_batching_dims =
      gather_op.getDimensionNumbers().getStartIndicesBatchingDims();
  if (!start_indices_type.hasStaticShape()) {
    // Dynamic dimensions in the start indices aren't supported in certain
    // cases that require reshaping the indices or result.
    if (!start_indices_batching_dims.empty()) {
      gather_op.emitOpError()
          << "Dynamic shaped start indices aren't supported when there are "
             "batching dimensions.";
    }
  }

  // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
  int64_t index_vector_dim =
      gather_op.getDimensionNumbers().getIndexVectorDim();
  if (failed(NormalizeIndexVector(gather_op, start_indices, start_indices_type,
                                  index_vector_dim, rewriter))) {
    return failure();
  }
  start_indices_type = mlir::cast<ShapedType>(start_indices.getType());

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

  // Verify that slice_sizes is 1 for the batching and indexed dimensions and
  // the full shape for the rest of the dimensions.
  auto slice_sizes = gather_op.getSliceSizes();
  int64_t index = 0;
  for (int64_t s : slice_sizes.getValues<int64_t>()) {
    if (llvm::count(start_index_map, index) ||
        llvm::count(start_indices_batching_dims, index)) {
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

  // Canonicalize the operand and start indices.
  auto canonical_operand =
      CanonicalizeOperand(gather_op, operand, operand_type,
                          operand_batching_dims, start_index_map, rewriter);
  auto canonical_operand_type =
      mlir::cast<ShapedType>(canonical_operand.getType());

  auto canonical_start_indices =
      CanonicalizeStartIndices(gather_op, start_indices, start_indices_type,
                               start_indices_batching_dims, rewriter);
  auto canonical_start_indices_type =
      mlir::cast<ShapedType>(canonical_start_indices.getType());

  TFL::CastOp cast_op = nullptr;
  if (canonical_start_indices_type.getElementType().isUnsignedInteger(32)) {
    cast_op = rewriter.create<TFL::CastOp>(
        gather_op->getLoc(),
        RankedTensorType::get(canonical_start_indices_type.getShape(),
                              rewriter.getI64Type()),
        canonical_start_indices);
  }

  llvm::SmallVector<int64_t> canonical_result_shape;
  for (int64_t i = 0; i < canonical_start_indices_type.getRank() - 1; ++i) {
    canonical_result_shape.push_back(
        canonical_start_indices_type.getDimSize(i));
  }
  for (int64_t i = canonical_start_indices_type.getDimSize(
           canonical_start_indices_type.getRank() - 1);
       i < canonical_operand_type.getRank(); ++i) {
    canonical_result_shape.push_back(canonical_operand_type.getDimSize(i));
  }

  auto canonical_result_type = RankedTensorType::get(
      canonical_result_shape, result_type.getElementType());
  auto canonical_result = rewriter.create<TFL::GatherNdOp>(
      gather_op->getLoc(), canonical_result_type, canonical_operand,
      cast_op ? cast_op.getResult() : canonical_start_indices);

  auto offset_dims = gather_op.getDimensionNumbers().getOffsetDims();
  auto final_result = UncanonicalizeResult(
      gather_op, canonical_result, canonical_result_type, result_type,
      offset_dims, start_indices_batching_dims, rewriter);

  rewriter.replaceOp(gather_op, final_result);
  return success();
}

void PopulateGatherPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                            ConversionTarget& target) {
  patterns.add<LegalizeGatherToSlice, LegalizeGatherToGatherND>(ctx);
  target.addDynamicallyLegalOp<mhlo::GatherOp>(IsGatherLegal);
}

}  // namespace mlir::odml
