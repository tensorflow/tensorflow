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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_REMOVE_IDENTITY_OP_PATTERN_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_REMOVE_IDENTITY_OP_PATTERN_H_

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

// Copied from tensorflow/compiler/mlir/lite/transforms/prepare_tf.cc.
// By removing identity ops, constant operands with dynamic shapes have static
// shape information which is necessary for correct pattern matching in this
// pass.
struct RemoveIdentity : public OpRewritePattern<TF::IdentityOp> {
  using OpRewritePattern<TF::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::IdentityOp identity,
                                PatternRewriter &rewriter) const override;
};

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_REMOVE_IDENTITY_OP_PATTERN_H_
