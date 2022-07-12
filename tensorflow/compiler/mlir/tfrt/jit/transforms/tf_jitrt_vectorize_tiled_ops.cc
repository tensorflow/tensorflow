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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::failure;
using mlir::success;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::CodegenStrategy;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;

// The upper limit for vectorization of untiled `linalg.fill`. If a tensor has a
// static shape with more elements, then `linalg.fill` won't be vectorized. It
// is expected that such operations are tiled to get to small static shapes.
constexpr int64_t kNumElementsThreshold = 1024;

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransferReadOp vector_read,
      mlir::PatternRewriter &rewriter) const override {
    auto expand = vector_read.getSource().getDefiningOp<ExpandShapeOp>();
    if (!expand) return failure();

    auto expand_src = expand.getSrc();
    auto expand_src_type = expand.getSrcType();
    auto expand_dst_type = expand.getResultType();
    if (expand_src_type.getRank() != 1 || expand_dst_type.getRank() != 2)
      return failure();

    auto result_type = vector_read.getType().dyn_cast<mlir::ShapedType>();
    if (!result_type || result_type.getShape() != expand_dst_type.getShape())
      return failure();

    auto zero = rewriter.create<ConstantIndexOp>(vector_read.getLoc(), 0);
    auto map = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                    vector_read.getContext());
    // TODO(pifon): Also support canonicalization in case the map is not an
    // identity.
    if (!map.isIdentity()) return failure();

    auto new_read = rewriter.create<TransferReadOp>(
        vector_read.getLoc(),
        mlir::VectorType::get(expand_src_type.getShape(),
                              expand_src_type.getElementType()),
        expand_src, mlir::ValueRange{zero}, mlir::AffineMapAttr::get(map),
        vector_read.getPadding(),
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

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Vectorize linalg.fill and linalg.generic operations.
    mlir::OpPassManager dynamicPM("func.func");
    CodegenStrategy strategy;
    strategy.vectorize(FillOp::getOperationName(), [](mlir::Operation *op) {
      auto fill = mlir::dyn_cast<FillOp>(op);
      if (!fill) return failure();

      if (op->getParentOfType<LoopOp>()) return success();

      // Allow vectorization for static shapes with low number of elements.
      auto output_type = fill.output().getType().cast<mlir::RankedTensorType>();
      if (output_type.hasStaticShape() &&
          output_type.getNumElements() < kNumElementsThreshold)
        return success();

      return failure();
    });

    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();

    mlir::OpPassManager dynamicPM2("func.func");
    CodegenStrategy strategy2;
    strategy2.vectorize(GenericOp::getOperationName(), [](mlir::Operation *op) {
      auto generic = mlir::dyn_cast<GenericOp>(op);
      if (!generic) return failure();

      if (op->getParentOfType<LoopOp>()) return success();

      // Allow vectorization of 1D reductions.
      return success(generic.getNumLoops() == 1 &&
                     generic.getNumReductionLoops() == 1);
    });

    strategy2.configurePassPipeline(dynamicPM2, funcOp.getContext());
    if (failed(runPipeline(dynamicPM2, funcOp))) return signalPassFailure();

    // Vectorize padding.
    mlir::RewritePatternSet patterns(funcOp.getContext());
    mlir::linalg::populatePadOpVectorizationPatterns(patterns);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
        patterns);
    patterns.add<TransferReadOfOneDimExpandShape>(funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVectorizeTiledOpsPass() {
  return std::make_unique<VectorizeTiledOpsPass>();
}

}  // namespace tensorflow
