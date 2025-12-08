/* Copyright 2025 The OpenXLA Authors.

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

#include <cassert>
#include <memory>
#include <utility>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_FUSEELEMENTWISEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

class FuseElementwisePass
    : public impl::FuseElementwisePassBase<FuseElementwisePass> {
 public:
  using FuseElementwisePassBase::FuseElementwisePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);

    // Only fuse op with one use.
    mlir::linalg::ControlFusionFn fuse_control_fn =
        [](mlir::OpOperand* fused_operand) {
          mlir::Operation* producer = fused_operand->get().getDefiningOp();
          return producer && producer->hasOneUse();
        };

    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                       fuse_control_fn);

    // Long chains of elementwise ops can require many iterations, so we have to
    // increase the limit from the default 10.
    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(1000);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFuseElementwisePass() {
  return std::make_unique<FuseElementwisePass>();
}

}  // namespace xla::cpu
