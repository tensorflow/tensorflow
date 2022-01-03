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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h.inc"

using mlir::failure;
using mlir::success;
using mlir::arith::ConstantIndexOp;
using mlir::linalg::CodegenStrategy;
using mlir::linalg::TiledLoopOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransferReadOp vector_read,
      mlir::PatternRewriter &rewriter) const override {
    auto expand = vector_read.source().getDefiningOp<ExpandShapeOp>();
    if (!expand) return failure();

    auto expand_src = expand.src();
    auto expand_src_type = expand.getSrcType();
    auto expand_dst_type = expand.getResultType();
    if (expand_src_type.getRank() != 1 || expand_dst_type.getRank() != 2)
      return failure();

    auto zero = rewriter.create<ConstantIndexOp>(vector_read.getLoc(), 0);
    auto map = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                    vector_read.getContext());

    auto new_read = rewriter.create<TransferReadOp>(
        vector_read.getLoc(),
        mlir::VectorType::get(expand_src_type.getShape(),
                              expand_src_type.getElementType()),
        expand_src, mlir::ValueRange{zero}, mlir::AffineMapAttr::get(map),
        vector_read.padding(),
        /*mask=*/mlir::Value(), rewriter.getBoolArrayAttr({true}));
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        vector_read, vector_read.getType(), new_read);
    return success();
  }
};

struct VectorizeTiledOpsPass
    : public VectorizeTiledOpsBase<VectorizeTiledOpsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnFunction() override {
    auto funcOp = getFunction();

    // Vector transfer options.
    mlir::VectorTransferToSCFOptions vector_transfer_opts;

    // Vectorize linalg.fill and linalg.generic operations.
    mlir::OpPassManager dynamicPM("builtin.func");
    CodegenStrategy strategy;
    strategy.vectorize(mlir::linalg::FillOp::getOperationName())
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .setVectorTransferToSCFOptions(vector_transfer_opts));
    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();

    mlir::OpPassManager dynamicPM2("builtin.func");
    CodegenStrategy strategy2;
    strategy2
        .vectorize(
            mlir::linalg::GenericOp::getOperationName(),
            [](mlir::Operation *op) {
              // TODO(b/206986898): Allow vectorization of non-tiled ops.
              return success(op->getParentOfType<TiledLoopOp>() != nullptr);
            })
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .setVectorTransferToSCFOptions(vector_transfer_opts));
    strategy2.configurePassPipeline(dynamicPM2, funcOp.getContext());
    if (failed(runPipeline(dynamicPM2, funcOp))) return signalPassFailure();

    // Vectorize padding.
    mlir::OwningRewritePatternList patterns(funcOp.getContext());
    mlir::linalg::populatePadTensorOpVectorizationPatterns(patterns);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
        patterns);
    patterns.insert<TransferReadOfOneDimExpandShape>(funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateVectorizeTiledOpsPass() {
  return std::make_unique<VectorizeTiledOpsPass>();
}

}  // namespace tensorflow
