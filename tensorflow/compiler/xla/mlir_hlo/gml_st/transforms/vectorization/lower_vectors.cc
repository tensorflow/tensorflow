/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_LOWERVECTORSPASS
#include "gml_st/transforms/passes.h.inc"

LogicalResult rewriteVectorContract(MLIRContext* ctx, Operation* funcOp) {
  // Reduce vector.contract dimensions to fit one of the lowering patterns to
  // vector.outerproduct.
  {
    RewritePatternSet castAwayUnitDimPatterns(ctx);
    vector::populateCastAwayVectorLeadingOneDimPatterns(
        castAwayUnitDimPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(castAwayUnitDimPatterns)))) {
      return failure();
    }

    RewritePatternSet reductionToContractPatterns(ctx);
    vector::populateVectorReductionToContractPatterns(
        reductionToContractPatterns);
    vector::ExtractOp::getCanonicalizationPatterns(reductionToContractPatterns,
                                                   ctx);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(reductionToContractPatterns)))) {
      return failure();
    }
  }

  RewritePatternSet patterns(ctx);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);

  // Currently we always lower vector.contract into vector.outerproduct.
  patterns.add<vector::ContractionOpToOuterProductOpLowering,
               vector::ContractionOpLowering>(
      vector::VectorTransformsOptions().setVectorTransformsOptions(
          vector::VectorContractLowering::OuterProduct),
      ctx, 2);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

// Rewrite `vector.transpose` into vector.shuffle ops.
LogicalResult rewriteVectorTranspose(MLIRContext* ctx, Operation* funcOp) {
  // Options for controlling specialized AVX2 lowerings. These lowerings may
  // either use intrin or inline_asm depending on needs. So they won't work for
  // SSE.
  auto avxLoweringOptions =
      x86vector::avx2::LoweringOptions().setTransposeOptions(
          x86vector::avx2::TransposeLoweringOptions()
              .lower4x8xf32()
              .lower8x8xf32());

  RewritePatternSet patterns(ctx);
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions = vectorTransformOptions.setVectorTransposeLowering(
      vector::VectorTransposeLowering::EltWise);
  vector::populateVectorTransposeLoweringPatterns(patterns,
                                                  vectorTransformOptions);
  x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
      patterns, avxLoweringOptions, /*benefit=*/10);

  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

// Rewrite N-D reductions as the sequence of vector operations without
// horizontal reduction, i.e. `vector.reduction`.
LogicalResult rewriteVectorReductionsND(MLIRContext* ctx, Operation* funcOp) {
  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, vector::VectorDialect>();
  target.addDynamicallyLegalOp<vector::MultiDimReductionOp>(
      [&](vector::MultiDimReductionOp op) {
        return op.getSourceVectorType().getRank() == 1;
      });

  RewritePatternSet patterns(ctx);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerParallel);
  return applyPartialConversion(funcOp, target, std::move(patterns));
}

// Rewrite 1D reductions as a `vector.reduction`.
LogicalResult rewriteVectorReductions1D(MLIRContext* ctx, Operation* op) {
  RewritePatternSet patterns(ctx);
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction);
  return applyPatternsAndFoldGreedily(op, std::move(patterns));
}

struct LowerVectorsPass : public impl::LowerVectorsPassBase<LowerVectorsPass> {
  LowerVectorsPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = &getContext();

    if (failed(rewriteVectorContract(ctx, funcOp))) signalPassFailure();
    if (failed(rewriteVectorTranspose(ctx, funcOp))) signalPassFailure();
    if (failed(rewriteVectorReductionsND(ctx, funcOp))) signalPassFailure();
    if (failed(rewriteVectorReductions1D(ctx, funcOp))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorsPass() {
  return std::make_unique<LowerVectorsPass>();
}

}  // namespace gml_st
}  // namespace mlir
