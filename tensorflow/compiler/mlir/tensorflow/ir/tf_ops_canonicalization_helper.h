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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_CANONICALIZATION_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_CANONICALIZATION_HELPER_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace mlir {
namespace TF {

// Eliminate attributes that are not needed, but can get attached to Ops
// during import.
template <typename Op>
struct DropAttributes : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  // Drop the "output_shapes" attribute.
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool found = !!op->removeAttr("output_shapes");
    return success(found);
  }
};

// Helper function to create TF op while copying all underscore attributes from
// another TF op.
// TODO(jpienaar): This is a workaround until behavior is established.
template <typename OpTy, typename... Args>
OpTy CreateTfOp(RewriterBase &b, Operation *op, Args &&...args) {
  auto ret = b.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  CopyDeviceAndUnderscoredAttributes(op, ret.getOperation());
  return ret;
}

// Helper function to replace TF op with another op while copying all underscore
// attributes from the TF op.
// TODO(jpienaar): This is a workaround until behavior is established.
template <typename OpTy, typename... Args>
OpTy ReplaceTfOpWithNewOp(RewriterBase &b, Operation *op, Args &&...args) {
  auto ret = CreateTfOp<OpTy>(b, op, std::forward<Args>(args)...);
  b.replaceOp(op, ret.getOperation()->getResults());
  return ret;
}

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_CANONICALIZATION_HELPER_H_
