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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_data_optimization.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

namespace {

struct FuseParallelMapAndBatch : public OpRewritePattern<BatchDatasetV2Op> {
  using OpRewritePattern<BatchDatasetV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchDatasetV2Op op,
                                PatternRewriter &rewriter) const override {
    auto batchInputDataset = op.input_dataset();

    ParallelMapDatasetOp batchInputOp = dyn_cast_or_null<ParallelMapDatasetOp>(
        batchInputDataset.getDefiningOp());
    if (!batchInputOp) return failure();

    // The type of the `num_parallel_calls` argument in ParallelMapDataset
    // and MapAndBatchDataset is different (int32 and int64 respectively)
    auto num_parallel_calls_op = rewriter.create<CastOp>(
        op.getLoc(), UnrankedTensorType::get(rewriter.getIntegerType(64)),
        batchInputOp.num_parallel_calls(), rewriter.getBoolAttr(false));

    auto fused_op = rewriter.create<MapAndBatchDatasetOp>(
        op.getLoc(), op.getType(), batchInputOp.input_dataset(),
        batchInputOp.other_arguments(), op.batch_size(),
        num_parallel_calls_op.y(), op.drop_remainder(), batchInputOp.f(),
        op.output_types(), op.output_shapes(),
        batchInputOp.preserve_cardinality());
    rewriter.replaceOp(op, {fused_op.handle()});
    return failure();
  }
};

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_tf_data_optimization.inc"
}  // namespace

void PopulateTFDataOptimizationPatterns(MLIRContext *context,
                                        OwningRewritePatternList *patterns) {
  patterns->insert<FuseParallelMapAndBatch>(context);
  populateWithGenerated(context, *patterns);
}

}  // namespace TF
}  // namespace mlir
