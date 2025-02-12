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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

#define DEBUG_TYPE "tf-gpu-op-fusion"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_TENSORFLOWGPUFUSION
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// GpuOpFusionPass is a pass performing fusion specific to GPU targets.
// This is an ad-hoc pass for now, but should be integrated with some notion
// of "target" in the MLIR pipeline in the future.
class GpuOpFusionPass : public impl::TensorflowGPUFusionBase<GpuOpFusionPass> {
 public:
  void runOnOperation() final;
};

//   %y:6 = "tf.FusedBatchNormV3"(%x, %scale, %offset, %mean, %variance)
//   %0 = "tf.Relu"(%y#0)
// ->
//   %y:6 = "tf._FusedBatchNormEx"(%x, %scale, %offset, %mean, %variance)
//
// Or:
//   %y:6 = "tf.FusedBatchNormV3"(%x, %scale, %offset, %mean, %variance)
//   %0 = "tf.AddV2"(%y#0, %side_input)
//   %1 = "tf.Relu"(%0)
// ->
//  %y:6 = "tf._FusedBatchNormEx"(%x, %scale, %offset, %mean, %variance,
//                                %side_input)
// TODO(aminim): we should revisit this as a declarative pattern.
// For the second pattern, there is not good way in the framework to handle the
// commutativity of the AddV2: we want the FusedBatchNormV3 on any side.
// Also we need some native calls to handle the "hasOneUse" aspects and the
// optional extra operands for the AddV2 case.
struct ReluToFusedBatchNorm : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu_op,
                                PatternRewriter &rewriter) const override {
    Operation *relu_input = relu_op.getFeatures().getDefiningOp();
    if (!relu_input) return failure();
    auto batch_norm = dyn_cast_or_null<FusedBatchNormV3Op>(relu_input);
    AddV2Op add_op;
    Value side_input;
    if (!batch_norm) {
      // We don't have a FusedBatchNorm as input to the ReLu, but we can get
      // through an AddV2 as well.
      add_op = dyn_cast_or_null<AddV2Op>(relu_input);
      if (!add_op) return failure();

      batch_norm =
          dyn_cast_or_null<FusedBatchNormV3Op>(add_op.getX().getDefiningOp());
      if (batch_norm) {
        side_input = add_op.getY();
      } else {
        // Didn't get a FusedBatchNorm on the LHS of the AddV2, try the RHS.
        batch_norm =
            dyn_cast_or_null<FusedBatchNormV3Op>(add_op.getY().getDefiningOp());
        if (!batch_norm) return failure();
        side_input = add_op.getX();
      }
    }
    assert(batch_norm);
    if (batch_norm.getIsTraining()) return failure();
    if (!batch_norm.getY().hasOneUse()) return failure();

    // Build the newly fused operation to replace the batch norm
    OperationState state(batch_norm.getLoc(),
                         _FusedBatchNormExOp::getOperationName());
    state.addOperands(batch_norm.getOperands());
    if (side_input) state.operands.push_back(side_input);
    state.addTypes(batch_norm.getResultTypes());
    state.addAttributes(batch_norm->getAttrs());
    Operation *op = rewriter.create(state);
    rewriter.replaceOp(batch_norm, op->getResults());

    // Depending on the case, we may fuse the add, the relu, or both.
    if (!add_op || add_op.getZ().hasOneUse()) {
      // We fuse the Relu only if the add has a single use, otherwise we only
      // fuse the add itself.
      op->setAttr("activation_mode", rewriter.getStringAttr("Relu"));
      rewriter.replaceOp(relu_op, op->getResult(0));
    }
    if (add_op) {
      rewriter.replaceOp(add_op, op->getResult(0));
    }

    return success();
  }
};

void GpuOpFusionPass::runOnOperation() {
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<ReluToFusedBatchNorm>(&getContext());
  (void)applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateGpuOpFusionPass() {
  return std::make_unique<GpuOpFusionPass>();
}

}  // namespace TF
}  // namespace mlir
