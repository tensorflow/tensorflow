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

#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/einsum.h"

namespace mlir {
namespace quant {
namespace {

class PrepareLiftingPass
    : public PassWrapper<PrepareLiftingPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLiftingPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-prepare-lifting";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply graph optimizations such as fusing and constant folding to "
           "prepare lifting.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

bool HasEqualElementSize(Value filter, Attribute val,
                         mlir::ArrayRef<unsigned> filter_indices,
                         mlir::ArrayRef<unsigned> val_indices) {
  int filter_result = 1;
  int val_result = 1;

  mlir::ShapedType shaped_filter = filter.getType().cast<ShapedType>();
  mlir::ShapedType shaped_val = val.dyn_cast<DenseElementsAttr>().getType();

  for (auto idx : filter_indices) {
    if (idx >= shaped_filter.getRank()) return false;
    filter_result *= shaped_filter.getDimSize(idx);
  }

  for (auto idx : val_indices) {
    if (idx >= shaped_val.getRank()) return false;
    val_result *= shaped_val.getDimSize(idx);
  }

  return filter_result == val_result;
}

// Copied from tensorflow/compiler/mlir/lite/transforms/prepare_tf.cc.
// By removing identity ops, constant operands with dynamic shapes have static
// shape information which is necessary for correct pattern matching in this
// pass.
struct RemoveIdentity : public OpRewritePattern<TF::IdentityOp> {
  using OpRewritePattern<TF::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::IdentityOp identity,
                                PatternRewriter &rewriter) const override {
    // Replace the op with the input if input and result have the same type.
    if (identity.input().getType() == identity.getType()) {
      rewriter.replaceOp(identity, identity.input());
      return success();
    }
    // Replace the op with the input if output is only used by TF ops.
    // Currently this is more on the conservative side since we need to ensure
    // every consumer op to be a TF op before applying this pattern. We can
    // consider to revisit this in the future if this turns out to be too
    // restrictive.
    for (Operation *user : identity->getUsers()) {
      if (user->getDialect()->getNamespace() != "tf") {
        return failure();
      }
    }

    rewriter.replaceOp(identity, identity.input());
    return success();
  }
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_lifting.inc"

void PrepareLiftingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();

  // The pattern includes decomposing batch normalization ops, fusing add/mul
  // with a constant operand to a preceding affine operation.
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  patterns.add<TF::ConvertTFEinsumOp, RemoveIdentity>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass() {
  return std::make_unique<PrepareLiftingPass>();
}

static PassRegistration<PrepareLiftingPass> pass;

}  // namespace quant
}  // namespace mlir
