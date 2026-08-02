/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // build_cleaner: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // build_cleaner: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir::odml {

#define GEN_PASS_DEF_DROPSHAPEASSERTIONSPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

struct DropShapeAssertionPattern : public RewritePattern {
  explicit DropShapeAssertionPattern(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (!op->getName().getStringRef().contains("custom_call")) {
      return failure();
    }

    Attribute raw_attr = op->getAttr("call_target_name");
    if (!raw_attr) return failure();

    StringRef target_name;
    if (auto attr = mlir::dyn_cast<vhlo::StringV1Attr>(raw_attr)) {
      target_name = attr.getValue();
    } else if (auto attr = mlir::dyn_cast<StringAttr>(raw_attr)) {
      target_name = attr.getValue();
    }

    if (!target_name.empty() && target_name.contains("shape_assertion")) {
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class DropShapeAssertionsPass
    : public impl::DropShapeAssertionsPassBase<DropShapeAssertionsPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DropShapeAssertionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static mlir::PassRegistration<DropShapeAssertionsPass> pass;

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateDropShapeAssertionsPass() {
  return std::make_unique<DropShapeAssertionsPass>();
}

}  // namespace mlir::odml
