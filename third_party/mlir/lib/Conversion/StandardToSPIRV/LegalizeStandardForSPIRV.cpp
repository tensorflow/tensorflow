//===- LegalizeStandardForSPIRV.cpp - Legalize ops for SPIR-V lowering ----===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This transformation pass legalizes operations before the conversion to SPIR-V
// dialect to handle ops that cannot be lowered directly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Merges subview operation with load operation.
class LoadOpOfSubViewFolder final : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(LoadOp loadOp,
                                     PatternRewriter &rewriter) const override;
};

/// Merges subview operation with store operation.
class StoreOpOfSubViewFolder final : public OpRewritePattern<StoreOp> {
public:
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(StoreOp storeOp,
                                     PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions for op legalization.
//===----------------------------------------------------------------------===//

/// Given the 'indices' of an load/store operation where the memref is a result
/// of a subview op, returns the indices w.r.t to the source memref of the
/// subview op. For example
///
/// %0 = ... : memref<12x42xf32>
/// %1 = subview %0[%arg0, %arg1][][%stride1, %stride2] : memref<12x42xf32> to
///          memref<4x4xf32, offset=?, strides=[?, ?]>
/// %2 = load %1[%i1, %i2] : memref<4x4xf32, offset=?, strides=[?, ?]>
///
/// could be folded into
///
/// %2 = load %0[%arg0 + %i1 * %stride1][%arg1 + %i2 * %stride2] :
///          memref<12x42xf32>
static LogicalResult
resolveSourceIndices(Location loc, PatternRewriter &rewriter,
                     SubViewOp subViewOp, ValueRange indices,
                     SmallVectorImpl<ValuePtr> &sourceIndices) {
  // TODO: Aborting when the offsets are static. There might be a way to fold
  // the subview op with load even if the offsets have been canonicalized
  // away.
  if (subViewOp.getNumOffsets() == 0)
    return failure();

  ValueRange opOffsets = subViewOp.offsets();
  SmallVector<ValuePtr, 2> opStrides;
  if (subViewOp.getNumStrides()) {
    // If the strides are dynamic, get the stride operands.
    opStrides = llvm::to_vector<2>(subViewOp.strides());
  } else {
    // When static, the stride operands can be retrieved by taking the strides
    // of the result of the subview op, and dividing the strides of the base
    // memref.
    SmallVector<int64_t, 2> staticStrides;
    if (failed(subViewOp.getStaticStrides(staticStrides))) {
      return failure();
    }
    opStrides.reserve(opOffsets.size());
    for (auto stride : staticStrides) {
      auto constValAttr = rewriter.getIntegerAttr(
          IndexType::get(rewriter.getContext()), stride);
      opStrides.emplace_back(rewriter.create<ConstantOp>(loc, constValAttr));
    }
  }
  assert(opOffsets.size() == opStrides.size());

  // New indices for the load are the current indices * subview_stride +
  // subview_offset.
  assert(indices.size() == opStrides.size());
  sourceIndices.resize(indices.size());
  for (auto index : llvm::enumerate(indices)) {
    auto offset = opOffsets[index.index()];
    auto stride = opStrides[index.index()];
    auto mul = rewriter.create<MulIOp>(loc, index.value(), stride);
    sourceIndices[index.index()] =
        rewriter.create<AddIOp>(loc, offset, mul).getResult();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Folding SubViewOp and LoadOp.
//===----------------------------------------------------------------------===//

PatternMatchResult
LoadOpOfSubViewFolder::matchAndRewrite(LoadOp loadOp,
                                       PatternRewriter &rewriter) const {
  auto subViewOp =
      dyn_cast_or_null<SubViewOp>(loadOp.memref()->getDefiningOp());
  if (!subViewOp) {
    return matchFailure();
  }
  SmallVector<ValuePtr, 4> sourceIndices;
  if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                  loadOp.indices(), sourceIndices)))
    return matchFailure();

  rewriter.replaceOpWithNewOp<LoadOp>(loadOp, subViewOp.source(),
                                      sourceIndices);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// Folding SubViewOp and StoreOp.
//===----------------------------------------------------------------------===//

PatternMatchResult
StoreOpOfSubViewFolder::matchAndRewrite(StoreOp storeOp,
                                        PatternRewriter &rewriter) const {
  auto subViewOp =
      dyn_cast_or_null<SubViewOp>(storeOp.memref()->getDefiningOp());
  if (!subViewOp) {
    return matchFailure();
  }
  SmallVector<ValuePtr, 4> sourceIndices;
  if (failed(resolveSourceIndices(storeOp.getLoc(), rewriter, subViewOp,
                                  storeOp.indices(), sourceIndices)))
    return matchFailure();

  rewriter.replaceOpWithNewOp<StoreOp>(storeOp, storeOp.value(),
                                       subViewOp.source(), sourceIndices);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// Hook for adding patterns.
//===----------------------------------------------------------------------===//

void mlir::populateStdLegalizationPatternsForSPIRVLowering(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpOfSubViewFolder, StoreOpOfSubViewFolder>(context);
}

//===----------------------------------------------------------------------===//
// Pass for testing just the legalization patterns.
//===----------------------------------------------------------------------===//

namespace {
struct SPIRVLegalization final : public OperationPass<SPIRVLegalization> {
  void runOnOperation() override;
};
} // namespace

void SPIRVLegalization::runOnOperation() {
  OwningRewritePatternList patterns;
  auto *context = &getContext();
  populateStdLegalizationPatternsForSPIRVLowering(context, patterns);
  applyPatternsGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<Pass> mlir::createLegalizeStdOpsForSPIRVLoweringPass() {
  return std::make_unique<SPIRVLegalization>();
}

static PassRegistration<SPIRVLegalization>
    pass("legalize-std-for-spirv", "Legalize standard ops for SPIR-V lowering");
