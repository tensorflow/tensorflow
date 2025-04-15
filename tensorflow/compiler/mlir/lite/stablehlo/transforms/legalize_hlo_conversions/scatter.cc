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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/scatter.h"

#include <cstdint>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

LogicalResult CanonicalizeScatterUpdates(
    Operation* scatter_op, llvm::ArrayRef<int64_t> update_window_dims,
    const Value& indices, const ShapedType& indices_type, Value& updates,
    ShapedType& updates_type, ConversionPatternRewriter& rewriter) {
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

template <typename BinaryOp, typename TfOp>
LogicalResult ConvertScatterOp<BinaryOp, TfOp>::matchAndRewrite(
    mhlo::ScatterOp scatter_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  OperandRange operands = scatter_op.getInputs();
  Value indices = scatter_op.getScatterIndices();
  OperandRange updates = scatter_op.getUpdates();
  if (operands.size() != 1 || updates.size() != 1) return failure();

  ShapedType operand_type = mlir::cast<ShapedType>(operands[0].getType());
  ShapedType indices_type = mlir::cast<ShapedType>(indices.getType());
  ShapedType updates_type = mlir::cast<ShapedType>(updates[0].getType());

  Value new_updates = updates[0];

  // Can only convert with static shaped scatter.
  if (!operand_type.hasStaticShape() || !indices_type.hasStaticShape() ||
      !updates_type.hasStaticShape()) {
    return failure();
  }

  // Match the scatter computation against computations supported by TF.
  if (failed(MatchBinaryReduceFunction<BinaryOp>(
          scatter_op.getUpdateComputation()))) {
    return failure();
  }

  auto scatter_dimension_numbers = scatter_op.getScatterDimensionNumbers();

  // Normalize indices so index_vector_dim == indices.rank() - 1.
  int64_t index_vector_dim = scatter_dimension_numbers.getIndexVectorDim();
  if (failed(NormalizeIndexVector(scatter_op, indices, indices_type,
                                  index_vector_dim, rewriter))) {
    return failure();
  }

  // Transform updates so that update window dims are the trailing
  // dimensions in the update tensor.
  auto update_window_dims = scatter_dimension_numbers.getUpdateWindowDims();
  if (failed(CanonicalizeScatterUpdates(scatter_op, update_window_dims, indices,
                                        indices_type, new_updates, updates_type,
                                        rewriter))) {
    return failure();
  }

  auto inserted_window_dims = scatter_dimension_numbers.getInsertedWindowDims();
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

  Value new_indices = indices;
  int64_t index_depth =
      permutation_and_shape.shape.getRank() - inserted_window_dims.size();
  int64_t num_updates = indices_type.getDimSize(0);
  // For TF::TensorScatterUpdateOp, `indices` must have at least 2 axes:
  // `(num_updates, index_depth)`. Reshape indices and updates if necessary.
  if (std::is_same<TfOp, TF::TensorScatterUpdateOp>::value &&
      indices_type.getRank() == 1 && updates_type.getRank() == 1 &&
      index_depth == 1 && num_updates == 1) {
    ImplicitLocOpBuilder builder(loc, rewriter);
    auto indices_shape = BuildIntArrayConstOp(
        builder, rewriter,
        llvm::SmallVector<int64_t>({num_updates, index_depth}),
        rewriter.getI32Type());
    new_indices = rewriter.create<TF::ReshapeOp>(
        loc,
        RankedTensorType::get({num_updates, index_depth},
                              indices_type.getElementType()),
        indices, indices_shape);
    auto updates_shape = BuildIntArrayConstOp(
        builder, rewriter,
        llvm::SmallVector<int64_t>({num_updates, updates_type.getDimSize(0)}),
        rewriter.getI32Type());
    new_updates = rewriter.create<TF::ReshapeOp>(
        loc,
        RankedTensorType::get({1, updates_type.getDimSize(0)},
                              updates_type.getElementType()),
        new_updates, updates_shape);
  }

  // Apply TF scatter to update the trailing dimensions of the
  // transposed operand.
  auto tf_scatter_op =
      rewriter.create<TfOp>(loc, permutation_and_shape.shape,
                            transposed_operand, new_indices, new_updates);

  // Reverse the earlier transpose.
  auto inverse_permutation = GetInversePermutation(permutation_array, rewriter);
  rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
      scatter_op, scatter_op.getResult(0).getType(), tf_scatter_op,
      inverse_permutation);

  return success();
}

}  // namespace odml
}  // namespace mlir
