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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UNROLL_BATCH_MATMUL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UNROLL_BATCH_MATMUL_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TFL {

// Unroll tf.BatchMatMulV2 op into a sequence of TF ops. Since TFLite does not
// support BatchMatMul operation, it unrolls a BatchMatMul op into tf.Reshape,
// tf.Slice, tf.MatMul, tf.Pack, and tf.Reshape ops.
template <typename BatchMatMulOpType>
class ConvertTFBatchMatMulOp : public OpRewritePattern<BatchMatMulOpType> {
  using OpRewritePattern<BatchMatMulOpType>::OpRewritePattern;

  static TF::ReshapeOp createReshapeOp(Value value, ArrayRef<int64_t> shape,
                                       Type element_type, Location loc,
                                       PatternRewriter& rewriter);

  static std::vector<Value> sliceInput(Value value, int batch_size,
                                       Location loc, PatternRewriter& rewriter);

  static TF::TransposeOp createTransposeOp(Value value, Location loc,
                                           PatternRewriter& rewriter);

  static TF::PackOp createMatMulOps(const std::vector<Value>& sliced_lhs,
                                    const std::vector<Value>& sliced_rhs,
                                    const tensorflow::MatMulBCast& bcast,
                                    int rows, int cols, Type element_type,
                                    Location loc, PatternRewriter& rewriter);

  PatternMatchResult matchAndRewrite(BatchMatMulOpType op,
                                     PatternRewriter& rewriter) const override;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UNROLL_BATCH_MATMUL_H_
