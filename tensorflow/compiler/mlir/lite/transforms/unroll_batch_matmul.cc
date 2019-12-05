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

#include "tensorflow/compiler/mlir/lite/transforms/unroll_batch_matmul.h"

#include <climits>
#include <cstdint>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/UniformSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TFL {

namespace {
// Unrolls a BatchMatMul on the batch dimension. We need to slice each batch out
// of the inputs, matmul them individually, then stack them all back together at
// the end.
struct UnrollBatchMatMulPass : public FunctionPass<UnrollBatchMatMulPass> {
  void runOnFunction() override;
};

void UnrollBatchMatMulPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFBatchMatMulOp<TF::BatchMatMulOp>,
                  ConvertTFBatchMatMulOp<TF::BatchMatMulV2Op>>(&getContext());
  applyPatternsGreedily(func, patterns);
}

}  // namespace

template <typename BatchMatMulOpType>
TF::ReshapeOp ConvertTFBatchMatMulOp<BatchMatMulOpType>::createReshapeOp(
    Value* value, ArrayRef<int64_t> shape, Type element_type, Location loc,
    PatternRewriter& rewriter) {
  int64_t shape_rank = shape.size();
  auto shape_spec_type =
      RankedTensorType::get({shape_rank}, rewriter.getIntegerType(64));
  Type resultType = RankedTensorType::get(shape, element_type);
  auto constant_attr = DenseElementsAttr::get(shape_spec_type, shape);
  auto shape_tensor =
      rewriter.create<ConstantOp>(loc, shape_spec_type, constant_attr);
  return rewriter.create<TF::ReshapeOp>(loc, resultType, /*tensor=*/value,
                                        /*shape=*/shape_tensor);
}

template <typename BatchMatMulOpType>
std::vector<Value*> ConvertTFBatchMatMulOp<BatchMatMulOpType>::sliceInput(
    Value* value, int batch_size, Location loc, PatternRewriter& rewriter) {
  RankedTensorType tensorType = value->getType().cast<RankedTensorType>();
  Type element_type = tensorType.getElementType();

  int rank = tensorType.getShape().size();
  int num_rows = tensorType.getShape()[rank - 2];
  int num_cols = tensorType.getShape()[rank - 1];

  // Reshape to rank-3 Tensor with first dimension as the batch size.
  auto reshape_op = createReshapeOp(value, {batch_size, num_rows, num_cols},
                                    element_type, loc, rewriter);

  SmallVector<int64_t, 3> slice_size = {1, num_rows, num_cols};

  std::vector<Value*> sliced;
  Type int64_type = rewriter.getIntegerType(64);
  Type slice_result_type = RankedTensorType::get(slice_size, element_type);

  // Slice along each batch index and remember the slice output for future
  // use.
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto vector3_type = RankedTensorType::get({3}, int64_type);

    auto begin_attr =
        DenseElementsAttr::get<int64_t>(vector3_type, {batch_idx, 0, 0});
    auto size_attr = DenseElementsAttr::get<int64_t>(vector3_type, slice_size);
    auto begin = rewriter.create<ConstantOp>(loc, vector3_type, begin_attr);
    auto size = rewriter.create<ConstantOp>(loc, vector3_type, size_attr);
    auto slice_op = rewriter.create<TF::SliceOp>(loc, slice_result_type,
                                                 /*input=*/reshape_op.output(),
                                                 begin, size);

    // Squeeze matrix, i.e. reshape [1, num_rows, num_cols] -> [num_rows,
    // num_cols]
    auto squeeze_op = createReshapeOp(slice_op.output(), {num_rows, num_cols},
                                      element_type, loc, rewriter);

    sliced.emplace_back(squeeze_op.output());
  }
  return sliced;
}

template <typename BatchMatMulOpType>
TF::TransposeOp ConvertTFBatchMatMulOp<BatchMatMulOpType>::createTransposeOp(
    Value* value, Location loc, PatternRewriter& rewriter) {
  auto value_type = value->getType().cast<RankedTensorType>();
  auto shape = value_type.getShape();
  int dims = shape.size();

  std::vector<int32_t> perm(dims);
  for (int i = 0; i < dims - 2; i++) {
    perm[i] = i;
  }
  perm[dims - 2] = dims - 1;
  perm[dims - 1] = dims - 2;

  auto perm_type = RankedTensorType::get({static_cast<int32_t>(perm.size())},
                                         rewriter.getIntegerType(32));

  auto perm_attr = DenseElementsAttr::get(perm_type, llvm::makeArrayRef(perm));
  auto perm_op = rewriter.create<ConstantOp>(loc, perm_type, perm_attr);

  std::vector<int64_t> transposed_shape(shape.begin(), shape.end());
  int64_t r = transposed_shape[dims - 1];
  int64_t c = transposed_shape[dims - 2];

  transposed_shape[dims - 1] = c;
  transposed_shape[dims - 2] = r;

  auto transposed_type =
      RankedTensorType::get(transposed_shape, value_type.getElementType());
  return rewriter.create<TF::TransposeOp>(loc, transposed_type, value, perm_op);
}

template <typename BatchMatMulOpType>
TF::PackOp ConvertTFBatchMatMulOp<BatchMatMulOpType>::createMatMulOps(
    const std::vector<Value*>& sliced_lhs,
    const std::vector<Value*>& sliced_rhs, const tensorflow::MatMulBCast& bcast,
    int rows, int cols, Type element_type, Location loc,
    PatternRewriter& rewriter) {
  auto matmul_type = RankedTensorType::get({rows, cols}, element_type);

  std::vector<Value*> matmuls;
  for (int batch_idx = 0; batch_idx < bcast.output_batch_size(); ++batch_idx) {
    int lhs_batch_idx, rhs_batch_idx;
    if (bcast.IsBroadcastingRequired()) {
      lhs_batch_idx = bcast.x_batch_indices()[batch_idx];
      rhs_batch_idx = bcast.y_batch_indices()[batch_idx];
    } else {
      lhs_batch_idx = batch_idx;
      rhs_batch_idx = batch_idx;
    }
    auto false_attr = rewriter.getBoolAttr(false);
    auto matmul = rewriter.create<TF::MatMulOp>(loc, matmul_type,
                                                /*a=*/sliced_lhs[lhs_batch_idx],
                                                /*b=*/sliced_rhs[rhs_batch_idx],
                                                /*transpose_a=*/false_attr,
                                                /*transpose_b=*/false_attr);
    matmuls.emplace_back(matmul.product());
  }

  // Combine the result of each individual MatMul into a rank-3 Tensor.
  Type packed_type = RankedTensorType::get(
      {bcast.output_batch_size(), rows, cols}, element_type);

  auto axis = rewriter.getI64IntegerAttr(0);
  return rewriter.create<TF::PackOp>(loc, packed_type,
                                     /*values=*/matmuls, axis);
}

template <typename BatchMatMulOpType>
PatternMatchResult ConvertTFBatchMatMulOp<BatchMatMulOpType>::matchAndRewrite(
    BatchMatMulOpType op, PatternRewriter& rewriter) const {
  Value* input_lhs = op.x();
  Value* input_rhs = op.y();

  if (!input_lhs->getType().isa<RankedTensorType>()) {
    // LHS must be a ranked tensor type
    return this->matchFailure();
  }
  if (!input_rhs->getType().isa<RankedTensorType>()) {
    // RHS must be a ranked tensor type
    return this->matchFailure();
  }

  auto lhs_type = input_lhs->getType().cast<RankedTensorType>();
  auto rhs_type = input_rhs->getType().cast<RankedTensorType>();

  auto element_type = lhs_type.getElementType();

  if (element_type != rhs_type.getElementType()) {
    // The element type of LHS must be the same with element type of RHS
    return this->matchFailure();
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  Location loc = op.getLoc();

  // Ensure that input ranks are at least 2.
  const int dims_a = lhs_shape.size();
  const int dims_b = rhs_shape.size();
  if (dims_a < 2 || dims_b < 2) {
    // Both inputs must have rank >= 2
    return this->matchFailure();
  }

  // Transpose LHS input if necessary.
  if (op.adj_x()) {
    input_lhs = createTransposeOp(input_lhs, loc, rewriter);

    lhs_type = input_lhs->getType().cast<RankedTensorType>();
    lhs_shape = lhs_type.getShape();
  }

  // Transpose RHS input if necessary.
  if (op.adj_y()) {
    input_rhs = createTransposeOp(input_rhs, loc, rewriter);

    rhs_type = input_rhs->getType().cast<RankedTensorType>();
    rhs_shape = rhs_type.getShape();
  }

  if (lhs_shape[dims_a - 1] != rhs_shape[dims_b - 2]) {
    // Input dimensions must be compatible for multiplication.
    return this->matchFailure();
  }

  if (dims_a == 2 && dims_b == 2) {
    // When both inputs are matrices, just replace the op to a matmul op.
    Type result_type =
        RankedTensorType::get({lhs_shape[0], rhs_shape[1]}, element_type);
    auto false_attr = rewriter.getBoolAttr(false);
    rewriter.replaceOpWithNewOp<TF::MatMulOp>(op, result_type,
                                              /*a=*/input_lhs,
                                              /*b=*/input_rhs,
                                              /*transpose_a=*/false_attr,
                                              /*transpose_b=*/false_attr);
    return this->matchSuccess();
  }

  // Ensure that batch shapes are broadcastable.
  tensorflow::MatMulBCast bcast(absl::InlinedVector<tensorflow::int64, 4>(
                                    lhs_shape.begin(), lhs_shape.end()),
                                absl::InlinedVector<tensorflow::int64, 4>(
                                    rhs_shape.begin(), rhs_shape.end()));

  if (!bcast.IsValid()) {
    // Input batch dimensions must be broadcastable
    return this->matchFailure();
  }

  // Compute slices for each batch in the LHS and RHS.
  std::vector<Value*> sliced_lhs =
      sliceInput(input_lhs, bcast.x_batch_size(), loc, rewriter);
  std::vector<Value*> sliced_rhs =
      sliceInput(input_rhs, bcast.y_batch_size(), loc, rewriter);

  // Compute (single batch) MatMul for each output batch. The MatMul outputs
  // are then packed together into one output Tensor.
  auto pack_op =
      createMatMulOps(sliced_lhs, sliced_rhs, bcast, lhs_shape[dims_a - 2],
                      rhs_shape[dims_b - 1], element_type, loc, rewriter);

  // Reshape the rank-3 Tensor into the correct output shape.
  const auto& result_batch_shape = bcast.output_batch_shape().dim_sizes();
  std::vector<int64_t> result_shape(result_batch_shape.begin(),
                                    result_batch_shape.end());
  result_shape.push_back(lhs_shape[dims_a - 2]);
  result_shape.push_back(rhs_shape[dims_b - 1]);

  auto reshape_op = createReshapeOp(pack_op.output(), result_shape,
                                    element_type, loc, rewriter);
  rewriter.replaceOp(op, reshape_op.output());
  return this->matchSuccess();
}

static PassRegistration<UnrollBatchMatMulPass> pass(
    "tfl-unroll-batch-matmul",
    "Unroll TF BatchMatMul op into Reshape, Slice, MatMul, Pack ops.");

}  // namespace TFL
}  // namespace mlir
