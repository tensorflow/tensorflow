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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/smuggle_disallowed_ops.h"

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace odml {

namespace {

// Convert op to stablehlo.custom_call
//   "tf.ResizeBilinear"(%341, %138) {
//      align_corners = false, device = "", half_pixel_centers = true}
//   ==>
//   stablehlo.custom_call @tf.ResizeBilinear(%arg0, %arg1) {
//      align_corners = false, device = "", half_pixel_centers = true}
LogicalResult SmuggleOp(Operation* op, PatternRewriter& rewriter) {
  auto call_target =
      rewriter.getNamedAttr("call_target_name", op->getName().getIdentifier());
  SmallVector<NamedAttribute> attrs{op->getAttrs()};
  attrs.push_back(call_target);
  auto custom_call = rewriter.create<mlir::stablehlo::CustomCallOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(), attrs);
  rewriter.replaceOp(op, custom_call.getResults());
  return success();
}

}  // namespace

template <typename OpTy>
class SmuggleOpPattern : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    return SmuggleOp(op, rewriter);
  }
};

class SmuggleDisallowedOpsPass
    : public PassWrapper<SmuggleDisallowedOpsPass,
                         OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "smuggle-disallowed-ops-pass"; }
  StringRef getDescription() const final {
    return "Smuggle disallowed ops via stablehlo.custom_calls";
  }
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<mlir::stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SmuggleOpPattern<TF::ResizeBilinearOp>>(&getContext());
    patterns.add<SmuggleOpPattern<TF::ResizeNearestNeighborOp>>(&getContext());

    ConversionTarget target(getContext());
    target.addIllegalOp<TF::ResizeBilinearOp, TF::ResizeNearestNeighborOp>();
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> CreateSmuggleDisallowedOpsPass() {
  return std::make_unique<SmuggleDisallowedOpsPass>();
}

static PassRegistration<SmuggleDisallowedOpsPass> pass;

}  // namespace odml
}  // namespace mlir
