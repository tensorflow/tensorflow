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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_SHLOTOVECTORPASS
#define GEN_PASS_DEF_SHLOTOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct LowerTranspose : mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType source_tensor_type = op.getOperand().getType();
    auto source_vector_type = mlir::VectorType::get(
        source_tensor_type.getShape(), source_tensor_type.getElementType());
    mlir::Value source_vector =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                op->getLoc(), source_vector_type, op.getOperand())
            .getResult(0);

    mlir::Value dest_vector = rewriter.create<mlir::vector::TransposeOp>(
        op->getLoc(), source_vector, op.getPermutation());

    mlir::RankedTensorType dest_tensor_type = op.getResult().getType();
    mlir::Value dest_tensor =
        rewriter
            .create<mlir::UnrealizedConversionCastOp>(
                op->getLoc(), dest_tensor_type, dest_vector)
            .getResult(0);

    rewriter.replaceAllUsesWith(op, dest_tensor);
    return mlir::success();
  }
};

class ShloToVectorPass : public impl::ShloToVectorPassBase<ShloToVectorPass> {
 public:
  using ShloToVectorPassBase::ShloToVectorPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerTranspose>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateShloToVectorPass() {
  return std::make_unique<ShloToVectorPass>();
}

}  // namespace xla::cpu
