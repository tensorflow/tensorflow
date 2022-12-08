/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/utils/vector_utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_GMLSTTOGPUPASS
#include "gml_st/transforms/passes.h.inc"

using namespace mlir;
using namespace mlir::gml_st;
using mlir::memref::SubViewOp;
using mlir::vector::CombiningKind;
using mlir::vector::ExtractOp;
using mlir::vector::MultiDimReductionOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

namespace {

struct MultiDimReductionOpToWarpReductionPattern
    : OpRewritePattern<MultiDimReductionOp> {
  using OpRewritePattern<MultiDimReductionOp>::OpRewritePattern;

  MultiDimReductionOpToWarpReductionPattern(MLIRContext* context,
                                            StringRef warpDistributionLabel)
      : OpRewritePattern<MultiDimReductionOp>(context),
        warpDistributionLabel(warpDistributionLabel) {}

  LogicalResult matchAndRewrite(MultiDimReductionOp reductionOp,
                                PatternRewriter& rewriter) const override;

 private:
  std::string warpDistributionLabel;
};

struct EliminateMaterializeOfTransferReadPattern
    : OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp materialize,
                                PatternRewriter& rewriter) const override;
};

struct EliminateDistributeIntoTransferWritePattern
    : OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferWriteOp transferWrite,
                                PatternRewriter& rewriter) const override;
};

/// Implements the GmlStToGpuPass declared in
/// gml_st/transforms/passes.td.
struct GmlStToGpuPass : public ::impl::GmlStToGpuPassBase<GmlStToGpuPass> {
  using GmlStToGpuPassBase<GmlStToGpuPass>::GmlStToGpuPassBase;

  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    RewritePatternSet patterns(&ctx);

    patterns.add<EliminateMaterializeOfTransferReadPattern,
                 EliminateDistributeIntoTransferWritePattern>(&ctx);
    patterns.add<MultiDimReductionOpToWarpReductionPattern>(
        &ctx, warpDistributionLabel);

    func::FuncOp func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

static Value createCombineOp(Location loc, Value lhs, Value rhs,
                             CombiningKind kind, PatternRewriter& rewriter) {
  auto helper = [&](auto dummy) {
    return rewriter.create<decltype(dummy)>(loc, lhs, rhs);
  };
  switch (kind) {
    case CombiningKind::ADD:
      return helper(arith::AddFOp());
    case CombiningKind::MUL:
      return helper(arith::MulFOp());
    case CombiningKind::MINUI:
      return helper(arith::MinUIOp());
    case CombiningKind::MINSI:
      return helper(arith::MinSIOp());
    case CombiningKind::MINF:
      return helper(arith::MinFOp());
    case CombiningKind::MAXUI:
      return helper(arith::MaxUIOp());
    case CombiningKind::MAXSI:
      return helper(arith::MaxSIOp());
    case CombiningKind::MAXF:
      return helper(arith::MaxFOp());
    case CombiningKind::AND:
      return helper(arith::AndIOp());
    case CombiningKind::OR:
      return helper(arith::OrIOp());
    case CombiningKind::XOR:
      return helper(arith::XOrIOp());
  }
  llvm_unreachable("unhandled");
}

LogicalResult MultiDimReductionOpToWarpReductionPattern::matchAndRewrite(
    MultiDimReductionOp reductionOp, PatternRewriter& rewriter) const {
  auto distributionLevelAttr =
      reductionOp->getAttrOfType<StringAttr>(kDistributionLabelKey);

  if (!distributionLevelAttr ||
      distributionLevelAttr.getValue() != warpDistributionLabel)
    return rewriter.notifyMatchFailure(reductionOp,
                                       "expected warp-level operation");

  auto inType = reductionOp.getSourceVectorType();
  auto elementType = inType.getElementType();
  if (!elementType.isIntOrFloat() || elementType.getIntOrFloatBitWidth() > 32) {
    return rewriter.notifyMatchFailure(
        reductionOp, "expected int or float element type <= 32b");
  }
  int64_t width = inType.getNumElements();
  std::initializer_list<int64_t> supportedWidths = {1, 2, 4, 8, 16, 32};
  if (!llvm::is_contained(supportedWidths, width)) {
    return rewriter.notifyMatchFailure(
        reductionOp, "expected input vector with size 2^N, <=32");
  }
  auto hasOneElement = [](auto type) {
    return type && type.getNumElements() == 1;
  };
  auto outType = reductionOp.getDestType().dyn_cast<VectorType>();
  if (!hasOneElement(outType)) {
    return rewriter.notifyMatchFailure(reductionOp, "expected 1-vector output");
  }
  auto distribute = reductionOp.getSource().getDefiningOp<DistributeOp>();
  if (!distribute) {
    return rewriter.notifyMatchFailure(
        reductionOp, "source not defined by gml_st.distribute");
  }
  // Even if this value was not written into the tile corresponding to the
  // current thread's lane id, this is fine, since it doesn't matter which
  // thread processes which element within a reduction.
  TypedValue<VectorType> distributeSource = distribute.getSource();
  if (!hasOneElement(distributeSource.getType())) {
    return rewriter.notifyMatchFailure(distribute, "expected 1-vector input");
  }

  // Preamble: extract element from input.
  Location loc = reductionOp->getLoc();
  Value result = rewriter.create<ExtractOp>(
      loc, distributeSource,
      SmallVector<int64_t>(distributeSource.getType().getRank(), 0));

  auto createConstant = [&](int32_t value) {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(value));
  };
  // Always have all lanes participate. This assumes that the lanes are either
  // in convergence or that they have exited the kernel.
  Value cWarpWidth = createConstant(32);
  // Create warp shuffles of increasing offset and interleave with a clone of
  // the accumulate block.
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  for (int64_t i = 1; i < width; i *= 2) {
    Value shuffle = result;
    if (bitWidth < 32) {
      shuffle = rewriter.create<arith::ExtUIOp>(
          loc, rewriter.getI32Type(),
          rewriter.create<arith::BitcastOp>(
              loc, rewriter.getIntegerType(bitWidth), shuffle));
    }
    shuffle = rewriter
                  .create<gpu::ShuffleOp>(loc, shuffle, createConstant(i),
                                          cWarpWidth, gpu::ShuffleMode::XOR)
                  .getShuffleResult();
    if (bitWidth < 32) {
      shuffle = rewriter.create<arith::BitcastOp>(
          loc, elementType,
          rewriter.create<arith::TruncIOp>(
              loc, rewriter.getIntegerType(bitWidth), shuffle));
    }
    result =
        createCombineOp(loc, result, shuffle, reductionOp.getKind(), rewriter);
  }

  // Combine with init element and broadcast result back to vector.
  Value acc = rewriter.create<ExtractOp>(loc, reductionOp.getAcc(), 0);
  result = createCombineOp(loc, acc, result, reductionOp.getKind(), rewriter);
  rewriter.replaceOpWithNewOp<vector::BroadcastOp>(reductionOp, outType,
                                                   result);

  return success();
}

SubViewOp createSubView(Location loc, Value source, TileOp tile,
                        PatternRewriter& rewriter) {
  Type memRefType = SubViewOp::inferResultType(
      source.getType().cast<MemRefType>(), tile.getStaticOffsets(),
      tile.getStaticSizes(), tile.getStaticStrides());
  return rewriter.create<SubViewOp>(
      loc, memRefType, source, tile.getOffsets(), tile.getSizes(),
      tile.getStrides(), tile.getStaticOffsets(), tile.getStaticSizes(),
      tile.getStaticStrides());
}

LogicalResult EliminateMaterializeOfTransferReadPattern::matchAndRewrite(
    MaterializeOp materialize, PatternRewriter& rewriter) const {
  // Match the following pattern:
  //  gml_st.materialize(
  //  vector.transfer_read Memref:$src[(arith.constant 0)...]
  //  gml_st.tile [$offsets] [$sizes] [$strides])
  auto transferRead = materialize.getSource().getDefiningOp<TransferReadOp>();
  if (!transferRead) {
    return rewriter.notifyMatchFailure(
        materialize, "expected vector.transfer_read as source");
  }
  Value source = transferRead.getSource();
  if (!source.getType().isa<MemRefType>()) {
    return rewriter.notifyMatchFailure(transferRead,
                                       "expected memref as source");
  }
  if (failed(matchSimpleTransferOp(transferRead, rewriter))) return failure();

  auto tile = materialize.getSet().getDefiningOp<TileOp>();
  if (!tile) {
    return rewriter.notifyMatchFailure(materialize,
                                       "expected gml_st.tile as set");
  }

  // Rewrite the pattern as:
  // vector.transfer_read
  //   (memref.subview $src [$offsets] [$sizes] [$strides])
  //   [(arith.constant 0)...]
  // TODO(b/254271932): This might not be correct if there is someone writing
  // to `source` in between `transferRead` and `materialize`. This won't happen
  // for elementwise fusion and softmax, but might become a problem down the
  // line.
  auto subview = createSubView(materialize.getLoc(), source, tile, rewriter);
  Type resultType = materialize.getResult().getType();
  if (!resultType.isa<VectorType>()) {
    // We have a transfer to a single element: just use memref.load directly.
    rewriter.replaceOpWithNewOp<memref::LoadOp>(materialize, subview,
                                                transferRead.getIndices());
    return success();
  }
  rewriter.replaceOpWithNewOp<TransferReadOp>(
      materialize, resultType, subview, transferRead.getIndices(),
      transferRead.getPermutationMap(), transferRead.getPadding(),
      /*mask=*/nullptr, transferRead.getInBounds().value_or(nullptr));
  return success();
}

LogicalResult EliminateDistributeIntoTransferWritePattern::matchAndRewrite(
    TransferWriteOp transferWrite, PatternRewriter& rewriter) const {
  // Match the following pattern:
  //  vector.transfer_write
  //    (gml_st.distribute $src into
  //      [(gml_st.tile [$offsets] [$sizes] [$strides])])
  //    Memref:$dst[(arith.constant 0)]
  Value destination = transferWrite.getSource();
  if (!destination.getType().isa<MemRefType>()) {
    return rewriter.notifyMatchFailure(transferWrite,
                                       "expected memref as destination");
  }
  if (failed(matchSimpleTransferOp(transferWrite, rewriter))) return failure();

  auto distribute = transferWrite.getVector().getDefiningOp<DistributeOp>();
  if (!distribute) {
    return rewriter.notifyMatchFailure(transferWrite,
                                       "expected distribute as source");
  }
  Value source = distribute.getSource();

  auto tile = distribute.getSet().getDefiningOp<TileOp>();
  if (!tile) {
    return rewriter.notifyMatchFailure(distribute,
                                       "expected gml_st.tile as set");
  }

  // Rewrite the pattern as:
  // vector.transfer_write $src,
  //   (memref.subview $dst [$offsets] [$sizes] [$strides])
  //   [(arith.constant 0)...]
  auto subview =
      createSubView(transferWrite.getLoc(), destination, tile, rewriter);
  rewriter.replaceOpWithNewOp<TransferWriteOp>(
      transferWrite, /*resultType=*/llvm::None, source, subview,
      transferWrite.getIndices(), transferWrite.getPermutationMap(),
      /*mask=*/nullptr, transferWrite.getInBounds().value_or(nullptr));
  return success();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::gml_st::createGmlStToGpuPass(
    StringRef warpDistributionLabel) {
  const GmlStToGpuPassOptions passOptions = {
      /*.warpDistributionLabel=*/std::string(warpDistributionLabel)};
  return std::make_unique<GmlStToGpuPass>(passOptions);
}
