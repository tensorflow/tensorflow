/* Copyright 2026 The OpenXLA Authors.

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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERPDLWAITPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

struct LowerPdlWaitPattern
    : public mlir::OpRewritePattern<xla::gpu::PdlWaitOp> {
  using OpRewritePattern<xla::gpu::PdlWaitOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xla::gpu::PdlWaitOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::NVVM::GriddepcontrolOp>(
        op, mlir::NVVM::GridDepActionKind::wait);
    return mlir::success();
  }
};

class LowerPdlWaitPass : public impl::LowerPdlWaitPassBase<LowerPdlWaitPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerPdlWaitPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerPdlWaitPass() {
  return std::make_unique<LowerPdlWaitPass>();
}

}  // namespace emitters
}  // namespace xla
