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

#include <utility>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h.inc"

constexpr llvm::StringRef kWasPeeledAttr = "PeelTiledLoopsPeeledAttr";

struct PeelTiledLoop
    : public mlir::OpRewritePattern<mlir::linalg::TiledLoopOp> {
  using mlir::OpRewritePattern<mlir::linalg::TiledLoopOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::TiledLoopOp loop,
      mlir::PatternRewriter &rewriter) const override {
    if (loop->hasAttr(kWasPeeledAttr)) return mlir::failure();
    auto true_attr = mlir::BoolAttr::get(rewriter.getContext(), true);
    loop->setAttr(kWasPeeledAttr, true_attr);
    for (int peeled_idx = loop.getNumLoops() - 1; peeled_idx >= 0;
         peeled_idx--) {
      mlir::linalg::TiledLoopOp peel;
      // Mark the new loop if one was created
      if (mlir::linalg::peelAndCanonicalizeTiledLoop(rewriter, loop, peeled_idx,
                                                     peel)
              .succeeded())
        peel->setAttr(kWasPeeledAttr, true_attr);
    }
    return mlir::success();
  }
};

struct PeelTiledLoopsPass : public PeelTiledLoopsBase<PeelTiledLoopsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {}

  void runOnFunction() override {
    auto func_op = getFunction();

    // Apply some canonicalizations before loop splitting confuses the
    // situation.
    // TODO(tpopp): See if this is still necessary in the integrated version.
    mlir::OwningRewritePatternList canonicalizations(func_op.getContext());
    mlir::linalg::TiledLoopOp::getCanonicalizationPatterns(
        canonicalizations, func_op.getContext());
    mlir::linalg::populateLinalgTilingCanonicalizationPatterns(
        canonicalizations);
    (void)applyPatternsAndFoldGreedily(func_op, std::move(canonicalizations));

    mlir::OwningRewritePatternList loopPeeling(func_op.getContext());
    loopPeeling.insert<PeelTiledLoop>(func_op.getContext());
    (void)applyPatternsAndFoldGreedily(func_op, std::move(loopPeeling));

    func_op->walk([&](mlir::linalg::TiledLoopOp op) {
      if (op->hasAttr(kWasPeeledAttr)) op->removeAttr(kWasPeeledAttr);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreatePeelTiledLoopsPass() {
  return std::make_unique<PeelTiledLoopsPass>();
}
}  // namespace tensorflow
