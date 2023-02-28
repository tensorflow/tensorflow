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
#include <optional>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

using func::FuncOp;

LogicalResult rewriteVectorContract(MLIRContext* ctx, FuncOp funcOp) {
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
LogicalResult rewriteVectorTranspose(MLIRContext* ctx, FuncOp funcOp) {
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
LogicalResult rewriteVectorReductionsND(MLIRContext* ctx, FuncOp funcOp) {
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

// Return the uses of op if they all are either StoreOp, TransferWriteOp, or
// SubviewOp with only StoreOp/TransferWriteOp users.
std::optional<llvm::SmallVector<Operation*>> getUsesIfAllStores(Operation* op) {
  llvm::SmallVector<Operation*> opUses;
  for (OpOperand& use : op->getUses()) {
    Operation* useOp = use.getOwner();
    if (isa<vector::TransferWriteOp, memref::StoreOp>(useOp)) {
      opUses.push_back(useOp);
      continue;
    }
    if (isa<memref::SubViewOp>(useOp)) {
      if (auto subviewUses = getUsesIfAllStores(useOp)) {
        opUses.insert(opUses.end(), subviewUses->begin(), subviewUses->end());
        opUses.push_back(useOp);
        continue;
      }
    }
    return std::nullopt;
  }
  return opUses;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
void eraseDeadAllocAndStores(func::FuncOp func) {
  SmallVector<Operation*> opToErase;
  func.walk([&](memref::AllocOp op) {
    if (auto uses = getUsesIfAllStores(op)) {
      // Insert the uses first,
      opToErase.insert(opToErase.end(), uses->begin(), uses->end());
      // then the op itself, since we will be erasing from opToErase's start.
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation* op : opToErase) {
    op->erase();
  }
}

// Pattern to canonialize tranpose where only one dimension is not unit
// dimension. In this case the transpose is a no-op and should be simplified
// before getting to the conversion to llvm/spirv.
class TransposeUnitDimToShapeCast
    : public OpRewritePattern<vector::TransposeOp> {
 public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    unsigned numNonUnitSrcDim =
        llvm::count_if(op.getSourceVectorType().getShape(),
                       [](int64_t dim) { return dim != 1; });
    if (numNonUnitSrcDim != 1) return failure();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), op.getVector());
    return success();
  }
};

// Run optimization transformations on vector transfer operations.
LogicalResult optimizeVectorTransfers(MLIRContext* ctx, FuncOp funcOp) {
  // Generate vector.shape_cast for dropping leading one dimensions in vector
  // ops. This increases the chance that we can forward more transfer writes
  // to transfer reads.
  {
    RewritePatternSet patterns(ctx);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.add<TransposeUnitDimToShapeCast>(ctx);
    mlir::vector::populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }

  // Move bitcast inwards from loop region boundaries to increase chances to
  // cancel them.
  {
    RewritePatternSet patterns(ctx);
    vector::populateBubbleVectorBitCastOpPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }

  // Third stage of patterns to flatten transfer ops.
  {
    RewritePatternSet patterns(ctx);
    mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    mlir::vector::populateFlattenVectorTransferPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }
  // Delete potential dead alloc and associated ops after store to load
  // forwarding.
  eraseDeadAllocAndStores(funcOp);
  return success();
}

LogicalResult lowerVectorOpsToSCF(MLIRContext* ctx, FuncOp funcOp) {
  RewritePatternSet patterns(ctx);
  auto vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll(true).setTargetRank(1);

  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    if (failed(optimizeVectorTransfers(ctx, funcOp))) signalPassFailure();
    if (failed(lowerVectorOpsToSCF(ctx, funcOp))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorsPass() {
  return std::make_unique<LowerVectorsPass>();
}

}  // namespace gml_st
}  // namespace mlir
