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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_TENSOROPSTOVECTORPASS
#define GEN_PASS_DEF_TENSOROPSTOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct LowerFromElements
    : mlir::OpRewritePattern<mlir::tensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::FromElementsOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::VectorType vector_type = GetVectorType(op.getType());
    mlir::Value vector_from_elements =
        rewriter.create<mlir::vector::FromElementsOp>(op.getLoc(), vector_type,
                                                      op->getOperands());
    rewriter.replaceOp(op, WriteVectorToTensor(rewriter, vector_from_elements));
    return mlir::success();
  }
};

struct LowerExtract : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value vector_input = ReadTensorToVector(rewriter, op.getTensor());
    llvm::SmallVector<mlir::OpFoldResult> indices(op.getIndices());
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(op, vector_input,
                                                         indices);
    return mlir::success();
  }
};

class TensorOpsToVectorPass
    : public impl::TensorOpsToVectorPassBase<TensorOpsToVectorPass> {
 public:
  using TensorOpsToVectorPassBase::TensorOpsToVectorPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerFromElements, LowerExtract>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTensorOpsToVectorPass() {
  return std::make_unique<TensorOpsToVectorPass>();
}

}  // namespace xla::cpu
