/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/sparse_expansions/dynamic_enqueue_sparse_expander.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/mlir/sparse_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Indices tensor should be transformed from a shape [?, 2] tensor to a
// [?, 3] tensor padded with 0's because
// EnqueueTPUEmbeddingArbitraryTensorBatchOp expects a [?, 3] indices tensor.
StatusOr<mlir::Value> ExpandIndices(mlir::OpBuilder& builder,
                                    mlir::Value indices) {
  int64_t num_dim =
      indices.getType().dyn_cast<mlir::RankedTensorType>().getDimSize(1);
  if (num_dim != 2)
    return errors::Unimplemented(
        "Sparse tensors with dense rank not equal to 2 is not yet supported in "
        "DTensor.");
  mlir::Location loc = indices.getLoc();
  auto indices_padded_type = mlir::RankedTensorType::get(
      {-1, 3},
      indices.getType().dyn_cast<mlir::RankedTensorType>().getElementType());
  // Little trick to make a rank-2 tensor of [[0,0], [0,1]] using rank 1
  // constants.
  mlir::Value indices_padding = builder.create<mlir::TF::ReshapeOp>(
      loc,
      mlir::TF::collection_ops_util::GetR1Const({0, 0, 0, 1}, builder, loc),
      mlir::TF::collection_ops_util::GetR1Const({2, 2}, builder, loc));
  mlir::Value indices_padded =
      builder.create<mlir::TF::PadOp>(loc, indices_padded_type,
                                      /*input=*/indices,
                                      /*paddings=*/indices_padding);
  return indices_padded;
}

}  // namespace

StatusOr<mlir::Operation*> DynamicEnqueueSparseExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::TF::DynamicEnqueueTPUEmbeddingArbitraryTensorBatchOp dense_enqueue_op =
      mlir::cast<mlir::TF::DynamicEnqueueTPUEmbeddingArbitraryTensorBatchOp>(
          op);

  mlir::OpBuilder builder(dense_enqueue_op);
  mlir::Location location = dense_enqueue_op->getLoc();

  mlir::OperandRange feature = dense_enqueue_op.embedding_indices();
  llvm::SmallVector<mlir::Value, 4> indices;
  llvm::SmallVector<mlir::Value, 4> values;

  for (mlir::Value sparse_feature_value : feature) {
    if (!IsSparseValue(sparse_feature_value)) {
      return errors::Internal(
          "Expected feature input to DynamicEnqueueOp to be a sparse input, "
          "but was not. This should not happen.");
    }
    // Indices tensor may need to be expanded to a different shape
    // for Enqueue op to work properly.
    TF_ASSIGN_OR_RETURN(
        mlir::Value expanded_indices,
        ExpandIndices(
            builder,
            GetIndicesFromSparseTensor(sparse_feature_value).ValueOrDie()));
    indices.push_back(expanded_indices);
    values.push_back(
        GetValuesFromSparseTensor(sparse_feature_value).ValueOrDie());
  }
  // Insert a new op with new sparse operands, and delete the old one.
  // This op does not have a return value so we do not need to replace any
  // consumers.
  mlir::Operation* sparse_enqueue_op =
      builder
          .create<mlir::TF::DynamicEnqueueTPUEmbeddingArbitraryTensorBatchOp>(
              location,
              /*sample_indices_or_row_splits_list=*/indices,
              /*embedding_indices=*/values,
              /*aggregation_weights=*/dense_enqueue_op.aggregation_weights(),
              /*mode_override=*/
              dense_enqueue_op.mode_override(),
              /*device_ordinal=*/dense_enqueue_op.device_ordinal(),
              /*combiners=*/dense_enqueue_op.combiners());
  dense_enqueue_op.erase();
  return sparse_enqueue_op;
}

}  // namespace dtensor
}  // namespace tensorflow
