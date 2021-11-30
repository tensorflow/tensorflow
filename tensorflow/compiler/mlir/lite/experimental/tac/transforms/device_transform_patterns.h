/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_PATTERNS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_PATTERNS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace tac {

// TODO(renjieliu): add more patterns.

// This basically:
// Pack => (Concat -> Reshape)
struct LowerPackIntoConcatReshape : public OpRewritePattern<TFL::PackOp> {
  using OpRewritePattern<TFL::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::PackOp pack_op,
                                PatternRewriter& rewriter) const override;
};

struct SquaredDifference : public OpRewritePattern<TFL::SquaredDifferenceOp> {
  using OpRewritePattern<TFL::SquaredDifferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SquaredDifferenceOp squared_diff_op,
                                PatternRewriter& rewriter) const override;
};

// Unroll split into a bunch of slice ops.
struct UnrollSplit : public OpRewritePattern<TFL::SplitOp> {
  using OpRewritePattern<TFL::SplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SplitOp split_op,
                                PatternRewriter& rewriter) const override;
};

// Unroll splitv into a bunch of slice ops.
struct UnrollSplitV : public OpRewritePattern<TFL::SplitVOp> {
  using OpRewritePattern<TFL::SplitVOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SplitVOp splitv_op,
                                PatternRewriter& rewriter) const override;
};

// Ensure bias for conv2d op.
struct EnsureBiasForConv2d : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv_op,
                                PatternRewriter& rewriter) const override;
};

// Pad slice to 4d.
struct PadSlice : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp slice_op,
                                PatternRewriter& rewriter) const override;
};

// Fully connected to conv2d.
struct FullyConnectedToConv : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter& rewriter) const override;
};

// Pad concat to 4d.
struct PadConcat : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat_op,
                                PatternRewriter& rewriter) const override;
};

// Convert reduce mean 4d to avg pool.
struct ReduceMeanToAvgPool : public OpRewritePattern<TFL::MeanOp> {
  using OpRewritePattern<TFL::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MeanOp mean_op,
                                PatternRewriter& rewriter) const override;
};

// Insert Requant ops for reduce_mean.
struct InsertRequantForReduceMean : public OpRewritePattern<TFL::MeanOp> {
  using OpRewritePattern<TFL::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MeanOp mean_op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_DEVICE_TRANSFORM_PATTERNS_H_
