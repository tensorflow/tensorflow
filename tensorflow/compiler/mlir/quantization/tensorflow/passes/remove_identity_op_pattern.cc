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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/remove_identity_op_pattern.h"

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

LogicalResult RemoveIdentity::matchAndRewrite(TF::IdentityOp identity,
                                              PatternRewriter &rewriter) const {
  for (Operation *user : identity->getUsers()) {
    // Replace the op with the input if output is only used by TF ops.
    // Currently this is more on the conservative side since we need to ensure
    // every consumer op to be a TF op before applying this pattern. We can
    // consider to revisit this in the future if this turns out to be too
    // restrictive.
    if (user->getDialect()->getNamespace() != "tf") {
      return failure();
    }
    // Identity ops of returning values might be helpful for some other
    // compilers, so avoid removing these Identity ops.
    if (user->hasTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
  }

  rewriter.replaceOp(identity, identity.getInput());
  return success();
}

}  // namespace quant
}  // namespace mlir
