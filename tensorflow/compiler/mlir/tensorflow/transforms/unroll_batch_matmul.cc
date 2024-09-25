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

#include "tensorflow/compiler/mlir/tensorflow/transforms/unroll_batch_matmul.h"

#include <climits>
#include <cstdint>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {

template <typename BatchMatMulOpType>
class ConvertTFBatchMatMulOp : public OpRewritePattern<BatchMatMulOpType> {
  using OpRewritePattern<BatchMatMulOpType>::OpRewritePattern;

  static TF::ReshapeOp createReshapeOp(Value value, ArrayRef<int64_t> shape,
                                       Type element_type, Location loc,
                                       PatternRewriter& rewriter);

  static std::vector<Value> sliceInput(Value value, int batch_size,
                                       Location loc, PatternRewriter& rewriter);

  LogicalResult matchAndRewrite(BatchMatMulOpType op,
                                PatternRewriter& rewriter) const override;
};

#define GEN_PASS_DEF_UNROLLBATCHMATMULPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Unrolls a BatchMatMul on the batch dimension. We need to slice each batch out
// of the inputs, matmul them individually, then stack them all back together at
// the end.
struct UnrollBatchMatMulPass
    : public impl::UnrollBatchMatMulPassBase<UnrollBatchMatMulPass> {
  void runOnOperation() override;
};

void UnrollBatchMatMulPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  PopulateUnrollTfBatchMatMul(&getContext(), patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

template <typename BatchMatMulOpType>
TF::ReshapeOp ConvertTFBatchMatMulOp<BatchMatMulOpType>::createReshapeOp(
    Value value, ArrayRef<int64_t> shape, Type element_type, Location loc,
    PatternRewriter& rewriter) {
  int64_t shape_rank = shape.size();
  auto shape_spec_type =
      RankedTensorType::get({shape_rank}, rewriter.getIntegerType(64));
  Type resultType = RankedTensorType::get(shape, element_type);
  auto constant_attr = DenseElementsAttr::get(shape_spec_type, shape);
  auto shape_tensor =
      rewriter.create<TF::ConstOp>(loc, shape_spec_type, constant_attr);
  return rewriter.create<TF::ReshapeOp>(loc, resultType, /*tensor=*/value,
                                        /*shape=*/shape_tensor);
}

template <typename BatchMatMulOpType>
std::vector<Value> ConvertTFBatchMatMulOp<BatchMatMulOpType>::sliceInput(
    Value value, int batch_size, Location loc, PatternRewriter& rewriter) {
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(value.getType());
  Type element_type = tensorType.getElementType();

  int rank = tensorType.getShape().size();
  int num_rows = tensorType.getShape()[rank - 2];
  int num_cols = tensorType.getShape()[rank - 1];

  std::vector<Value> sliced;

  if (batch_size == 1) {
    // Batch size is 1, no splitting is required
    // Squeeze the batch dimension, i.e. reshape
    // [1, num_rows, num_cols] -> [num_rows, num_cols]
    auto reshape_op = createReshapeOp(value, {num_rows, num_cols}, element_type,
                                      loc, rewriter);
    sliced.emplace_back(reshape_op.getOutput());
  } else {
    // Reshape to rank-3 tensor with first dimension as the batch size.
    auto reshape_op = createReshapeOp(value, {batch_size, num_rows, num_cols},
                                      element_type, loc, rewriter);

    // Create a constant op for the split axis (=0)
    auto split_dimension_type =
        RankedTensorType::get({}, rewriter.getIntegerType(32));
    auto split_dimension_attr = DenseElementsAttr::get(split_dimension_type, 0);
    auto split_dimension_op = rewriter.create<TF::ConstOp>(
        loc, split_dimension_type, split_dimension_attr);

    // Split along each batch.
    SmallVector<int64_t, 3> slice_size = {1, num_rows, num_cols};
    Type slice_result_type = RankedTensorType::get(slice_size, element_type);
    llvm::SmallVector<Type, 4> output_types(batch_size, slice_result_type);
    auto split_op = rewriter.create<TF::SplitOp>(loc, output_types,
                                                 split_dimension_op.getOutput(),
                                                 reshape_op.getOutput());

    // Squeeze each batch, i.e. reshape
    // [1, num_rows, num_cols] -> [num_rows, num_cols]
    for (const auto& split_value : split_op.getOutput()) {
      auto reshape_op = createReshapeOp(split_value, {num_rows, num_cols},
                                        element_type, loc, rewriter);

      sliced.emplace_back(reshape_op.getOutput());
    }
  }
  return sliced;
}

template <typename BatchMatMulOpType>
LogicalResult ConvertTFBatchMatMulOp<BatchMatMulOpType>::matchAndRewrite(
    BatchMatMulOpType op, PatternRewriter& rewriter) const {
  Value input_lhs = op.getX();
  Value input_rhs = op.getY();

  if (!mlir::isa<RankedTensorType>(input_lhs.getType())) {
    // LHS must be a ranked tensor type
    return failure();
  }
  if (!mlir::isa<RankedTensorType>(input_rhs.getType())) {
    // RHS must be a ranked tensor type
    return failure();
  }

  auto lhs_type = mlir::cast<RankedTensorType>(input_lhs.getType());
  auto rhs_type = mlir::cast<RankedTensorType>(input_rhs.getType());

  // Skip int8 x int8 => int32.
  if (lhs_type.getElementType().isInteger(8) &&
      rhs_type.getElementType().isInteger(8)) {
    return rewriter.notifyMatchFailure(op,
                                       "skip unrolling for int8 BatchMatMulV3");
  }

  auto element_type = lhs_type.getElementType();

  if (element_type != rhs_type.getElementType()) {
    // The element type of LHS must be the same with element type of RHS
    return failure();
  }

  std::vector<int64_t> lhs_shape = lhs_type.getShape();
  std::vector<int64_t> rhs_shape = rhs_type.getShape();

  Location loc = op.getLoc();

  // Ensure that input ranks are at least 2.
  const int lhs_dims = lhs_shape.size();
  const int rhs_dims = rhs_shape.size();
  if (lhs_dims < 2 || rhs_dims < 2) {
    // Both inputs must have rank >= 2
    return failure();
  }

  // Replace the last 2 dimensions of LHS and RHS if necessary.
  // The actual transpose is done by MatMulOp.
  if (op.getAdjX()) {
    std::swap(lhs_shape[lhs_dims - 1], lhs_shape[lhs_dims - 2]);
  }
  if (op.getAdjY()) {
    std::swap(rhs_shape[rhs_dims - 1], rhs_shape[rhs_dims - 2]);
  }

  const int64_t rows = lhs_shape[lhs_dims - 2];
  const int64_t cols = rhs_shape[rhs_dims - 1];

  if (lhs_shape[lhs_dims - 1] != rhs_shape[rhs_dims - 2]) {
    // Input dimensions must be compatible for multiplication.
    return failure();
  }

  const auto matmul_type = RankedTensorType::get({rows, cols}, element_type);

  if (lhs_dims == 2 && rhs_dims == 2) {
    // When both inputs are matrices, just replace the op with a matmul op.
    rewriter.replaceOpWithNewOp<TF::MatMulOp>(op, matmul_type,
                                              /*a=*/input_lhs,
                                              /*b=*/input_rhs,
                                              /*transpose_a=*/op.getAdjX(),
                                              /*transpose_b=*/op.getAdjY());
    return success();
  }

  // Input dimensions must be defined. MatMulBCast does not support partial
  // shapes.
  for (auto dim : lhs_shape) {
    if (dim == mlir::ShapedType::kDynamic) {
      return failure();
    }
  }
  for (auto dim : rhs_shape) {
    if (dim == mlir::ShapedType::kDynamic) {
      return failure();
    }
  }
  // Ensure that batch shapes are broadcastable.
  tensorflow::MatMulBCast bcast(
      absl::InlinedVector<int64_t, 4>(lhs_shape.begin(), lhs_shape.end()),
      absl::InlinedVector<int64_t, 4>(rhs_shape.begin(), rhs_shape.end()));

  if (!bcast.IsValid()) {
    // Input batch dimensions must be broadcastable
    return failure();
  }

  // Compute slices for each batch in the LHS and RHS.
  std::vector<Value> sliced_lhs =
      sliceInput(input_lhs, bcast.x_batch_size(), loc, rewriter);
  std::vector<Value> sliced_rhs =
      sliceInput(input_rhs, bcast.y_batch_size(), loc, rewriter);

  // Compute (single batch) MatMul for each output batch.
  std::vector<Value> matmuls;
  matmuls.reserve(bcast.output_batch_size());
  for (int batch_idx : llvm::seq<int>(0, bcast.output_batch_size())) {
    int lhs_batch_idx, rhs_batch_idx;
    if (bcast.IsBroadcastingRequired()) {
      lhs_batch_idx = bcast.x_batch_indices()[batch_idx];
      rhs_batch_idx = bcast.y_batch_indices()[batch_idx];
    } else {
      lhs_batch_idx = batch_idx;
      rhs_batch_idx = batch_idx;
    }
    auto matmul = rewriter.create<TF::MatMulOp>(loc, matmul_type,
                                                /*a=*/sliced_lhs[lhs_batch_idx],
                                                /*b=*/sliced_rhs[rhs_batch_idx],
                                                /*transpose_a=*/op.getAdjX(),
                                                /*transpose_b=*/op.getAdjY());
    matmuls.emplace_back(matmul.getProduct());
  }

  // Combine the result of each individual MatMul into a rank-3 tensor.
  Type packed_type = RankedTensorType::get(
      {bcast.output_batch_size(), rows, cols}, element_type);
  const auto axis = rewriter.getI64IntegerAttr(0);
  auto pack_op =
      rewriter.create<TF::PackOp>(loc, packed_type, /*values=*/matmuls, axis);

  // Reshape the rank-3 tensor into the correct output shape.
  const auto& result_batch_shape = bcast.output_batch_shape().dim_sizes();
  std::vector<int64_t> result_shape(result_batch_shape.begin(),
                                    result_batch_shape.end());
  result_shape.push_back(rows);
  result_shape.push_back(cols);

  auto reshape_op = createReshapeOp(pack_op.getOutput(), result_shape,
                                    element_type, loc, rewriter);
  rewriter.replaceOp(op, reshape_op.getOutput());
  return success();
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateUnrollBatchMatMulPassPass() {
  return std::make_unique<UnrollBatchMatMulPass>();
}

}  // namespace TF
}  // namespace mlir

void mlir::TF::PopulateUnrollTfBatchMatMul(MLIRContext* context,
                                           RewritePatternSet& patterns) {
  patterns.add<ConvertTFBatchMatMulOp<TF::BatchMatMulOp>,
               ConvertTFBatchMatMulOp<TF::BatchMatMulV2Op>,
               ConvertTFBatchMatMulOp<TF::BatchMatMulV3Op>>(context);
}
