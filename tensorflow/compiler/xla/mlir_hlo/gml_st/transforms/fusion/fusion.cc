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

#include "gml_st/transforms/fusion/fusion.h"

#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/rewriters.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONPASS
#include "gml_st/transforms/passes.h.inc"

// TODO(frgossen): Move this to the shape reification pass.
struct DimOpFissionPattern : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    auto shapeDef = llvm::dyn_cast_or_null<shape::ShapeOfOp>(
        extract.getTensor().getDefiningOp());
    if (!shapeDef || extract.getIndices().size() != 1) return failure();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(extract, shapeDef.getArg(),
                                               extract.getIndices().front());
    return success();
  }
};

// TODO(frgossen): Implement this through the shape reification interface and
// move this pattern to the shape reification pass.
struct DimOpReificationPattern : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.getSource().getDefiningOp();
    if (!def) return failure();

    // TODO(pifon): Split this pattern into many.
    // Case MaterializeOp.
    if (auto materializeOp = llvm::dyn_cast<MaterializeOp>(def)) {
      assert(materializeOp->getNumResults() == 1 && "assume single result");
      auto dimConstantIndex = op.getConstantIndex();
      if (!dimConstantIndex.has_value()) return failure();

      rewriter.replaceOp(op, materializeOp.getSizes()[*dimConstantIndex]);
      return success();
    }
    // Case LinalgOp.
    if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(def)) {
      if (linalgOp->getNumResults() != 1 || !linalgOp.hasTensorSemantics()) {
        return failure();
      }
      Value outputOperand = linalgOp.getDpsInitOperand(0)->get();
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, outputOperand,
                                                 op.getIndex());
      return success();
    }

    // Case EmptyOp.
    if (auto emptyTensorOp = llvm::dyn_cast<tensor::EmptyOp>(def)) {
      if (auto indexConstantOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
              op.getIndex().getDefiningOp())) {
        int64_t idx =
            indexConstantOp.getValue().dyn_cast<IntegerAttr>().getInt();
        OpFoldResult dim = emptyTensorOp.getMixedSizes()[idx];
        Value dimValue;
        if (dim.is<Value>()) {
          dimValue = dim.get<Value>();
        } else {
          assert(dim.is<Attribute>() && "expected Value or Attribute");
          int64_t dimInt = dim.get<Attribute>().cast<IntegerAttr>().getInt();
          dimValue =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dimInt);
        }
        assert(dimValue);
        rewriter.replaceOp(op, ValueRange{dimValue});
        return success();
      }
    }

    // Case ConcatenateOp.
    if (auto concat = llvm::dyn_cast<thlo::ConcatenateOp>(def)) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, concat.getInit(),
                                                 op.getIndex());
      return success();
    }

    // Case DynamicBroadcastInDimOp.
    if (auto bcast = llvm::dyn_cast<thlo::DynamicBroadcastInDimOp>(def)) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, bcast.getInit(),
                                                 op.getIndex());
      return success();
    }

    return failure();
  }
};

// Finds the `dst` operand of `setYieldOp` that matches `currentDst` and then
// replaces it with the corresponding `init` operand of the defining op of
// `currentDst`. At the moment this update is restricted to `linalg.fill` only,
// later it can be relaxed to support fusion of transposes into
// `gml_st.parallel`.
LogicalResult replaceSetYieldDstByProducerInit(SetYieldOp setYieldOp,
                                               Value currentDst) {
  auto fillOp = currentDst.getDefiningOp<linalg::FillOp>();
  if (!fillOp) return failure();

  Value init = fillOp.getDpsInitOperand(0)->get();
  for (OpOperand& operand : setYieldOp->getOpOperands()) {
    if (operand.get() != currentDst) continue;
    operand.set(init);
    return success();
  }
  return failure();
}

class FusionPattern : public OpRewritePattern<MaterializeOp> {
 public:
  FusionPattern(MLIRContext* context,
                function_ref<LogicalResult(MaterializeOp)> filterFn,
                mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<MaterializeOp>(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(MaterializeOp materializeOp,
                                PatternRewriter& rewriter) const override {
    assert(filterFn && "expect filter function");
    if (failed(filterFn(materializeOp)))
      return rewriter.notifyMatchFailure(materializeOp, "filtered");
    return fuse(rewriter, materializeOp);
  }

 private:
  function_ref<LogicalResult(MaterializeOp)> filterFn;
};

struct FusionPass : public impl::FusionPassBase<FusionPass> {
  FusionPass(StringRef producer, StringRef consumer) {
    this->producerLabel = producer.str();
    this->consumerLabel = consumer.str();
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();

    auto filterFn = [&](MaterializeOp op) {
      Operation* producerOp = op.getSource().getDefiningOp();
      if (!producerOp || (!producerLabel.empty() &&
                          !hasMatchingLabel(producerOp, producerLabel))) {
        return failure();
      }

      Operation* consumerOp = nullptr;
      if (!consumerLabel.empty()) {
        for (Operation* user : op.getResult().getUsers()) {
          if (hasMatchingLabel(user, consumerLabel)) {
            consumerOp = user;
            break;
          }
        }
        return success(consumerOp != nullptr);
      }

      return success();
    };

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    populateFusionPatterns(ctx, filterFn, &patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

bool isEqualOp(const Operation* lhsC, const Operation* rhsC) {
  return OperationEquivalence::isEquivalentTo(
      const_cast<Operation*>(lhsC), const_cast<Operation*>(rhsC),
      OperationEquivalence::exactValueMatch,
      OperationEquivalence::ignoreValueEquivalence,
      OperationEquivalence::IgnoreLocations);
}

template <class OpTy>
void eliminateEqualOps(PatternRewriter& rewriter, Block& block) {
  SmallVector<OpTy> uniqueOps;
  for (auto op : llvm::make_early_inc_range(block.getOps<OpTy>())) {
    auto* it = llvm::find_if(
        uniqueOps, [&](OpTy uniqueOp) { return isEqualOp(uniqueOp, op); });
    if (it == uniqueOps.end())
      uniqueOps.push_back(op);
    else
      rewriter.replaceOp(op, it->getResult());
  }
}

void eliminateTriviallyDeadUsers(PatternRewriter& rewriter, Operation* op) {
  for (auto* user :
       DenseSet<Operation*>(op->getUsers().begin(), op->getUsers().end())) {
    if (isOpTriviallyDead(user)) rewriter.eraseOp(user);
  }
}

void reifyDimOp(PatternRewriter& rewriter, tensor::DimOp dimOp) {
  OpResult dimValue = dimOp.getSource().template dyn_cast<OpResult>();
  if (!dimValue) return;
  auto rankedShapeTypeOp =
      dyn_cast<ReifyRankedShapedTypeOpInterface>(dimValue.getOwner());
  if (!rankedShapeTypeOp) return;

  std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
  if (!dimIndex) return;

  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(
          rankedShapeTypeOp.reifyResultShapes(rewriter, reifiedResultShapes)))
    return;

  if (reifiedResultShapes.size() != rankedShapeTypeOp->getNumResults()) return;

  unsigned resultNumber = dimValue.getResultNumber();
  auto sourceType = dimValue.getType().dyn_cast<RankedTensorType>();
  if (reifiedResultShapes[resultNumber].size() !=
      static_cast<size_t>(sourceType.getRank()))
    return;

  rewriter.replaceOp(dimOp, reifiedResultShapes[resultNumber][*dimIndex]);
}

void reifyDimOpsUsers(PatternRewriter& rewriter, Operation* op) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  for (auto* user : llvm::make_early_inc_range(op->getUsers())) {
    auto dimOp = dyn_cast<tensor::DimOp>(user);
    if (dimOp) reifyDimOp(rewriter, dimOp);
  }
}

// Iterates over MaterializeOps inside the block, finds a suitable candidate for
// fusion and fuses it. The fusion candidate should satisfy the filter function
// and not have uses outside of the block. Fails if nothing can be fused.
LogicalResult fuseGreedilyOneOpIntoBlock(
    PatternRewriter& rewriter, Block& block,
    llvm::function_ref<bool(Operation*)> filterFn) {
  // Ad-hoc CSE to eliminate duplicate MatrializeOp that could have been added
  // after previous fusions. Running the whole CSE pass would be to expensive
  // here and unnecessary. Without removing those duplicate, some ops will be
  // fused multiple times resulting in exponential code growth.
  eliminateEqualOps<TileOp>(rewriter, block);
  eliminateEqualOps<MaterializeOp>(rewriter, block);

  for (auto materializeOp : block.getOps<MaterializeOp>()) {
    auto* fusionCandidate = materializeOp.getSource().getDefiningOp();
    // Do not fuse if there is no defining op. Of example if it's a materialize
    // from a function argument.
    if (!fusionCandidate) continue;

    if (filterFn && !filterFn(fusionCandidate)) continue;

    // Ad-hoc DCE to trim the fusion candidate from dead users that could have
    // been added in the previous fusion cycles. Normally those ops would be
    // garbage collected after the pattern rewriter driver finished working, but
    // here it requires manual handling.
    eliminateTriviallyDeadUsers(rewriter, fusionCandidate);

    // Push tensor.dim ops 'above' the fusion candidate. This is normally done
    // by canonicalization passes, but running the whole canonicalization
    // pipeline here is too expensive.
    reifyDimOpsUsers(rewriter, fusionCandidate);

    // After the previous steps, materializeOp should be only one user of the
    // fusion candidate. Otherwise this candidate should not be fused.
    auto fusionCandidateUsers = llvm::to_vector(fusionCandidate->getUsers());
    if (fusionCandidateUsers.size() != 1 ||
        fusionCandidateUsers[0] != materializeOp)
      continue;

    if (succeeded(fuse(rewriter, materializeOp))) return success();
  }
  return failure();
}

}  // namespace

FailureOr<Operation*> fuse(PatternRewriter& rewriter,
                           MaterializeOp materializeOp) {
  Location loc = materializeOp.getLoc();
  FailureOr<Value> fusedOr = createFusedOp(rewriter, materializeOp);
  if (failed(fusedOr)) return failure();  // Match failure already notified.

  // Insert cast if needed.
  Value fused = *fusedOr;
  if (fused.getType() != materializeOp.getType()) {
    if (!materializeOp.getType().isa<RankedTensorType>()) {
      // the result should be a scalar, insert tensor.extract
      auto tensorType = fused.getType().dyn_cast<RankedTensorType>();
      assert(tensorType && tensorType.getNumElements() == 1 &&
             "resulting tensor should contain a single element");
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      fused = rewriter.create<tensor::ExtractOp>(
          loc, fused, SmallVector<Value>(tensorType.getRank(), zero));
    } else {
      // The result should be a tensor, cast it to the correct shape
      fused =
          rewriter.create<tensor::CastOp>(loc, materializeOp.getType(), fused);
    }
  }

  // Update destination argument of SetYieldOp if we are fusing into the
  // output tile.
  if (auto parallelOp = dyn_cast<ParallelOp>(materializeOp->getParentOp())) {
    SetYieldOp setYieldOp = parallelOp.getTerminator();
    Value src = materializeOp.getSource();
    if (llvm::is_contained(src.getUsers(), setYieldOp)) {
      if (failed(replaceSetYieldDstByProducerInit(setYieldOp, src)))
        return failure();
    }
  }

  rewriter.replaceOp(materializeOp, fused);
  return fused.getDefiningOp();
}

void fuseGreedily(PatternRewriter& rewriter, Block& block,
                  llvm::function_ref<bool(Operation*)> filterFn) {
  while (succeeded(fuseGreedilyOneOpIntoBlock(rewriter, block, filterFn)))
    ;
}

FusionCluster findMapFusionCluster(Operation* op) {
  // Find the root operation in the chain of elementwise ops. Current approach
  // doesn't work well if maps don't form a chain.
  Operation* rootOp = op;
  while (true) {
    auto users = llvm::to_vector(rootOp->getUsers());

    if (users.size() != 1) break;
    if (!isa<linalg::MapOp>(users[0])) break;

    rootOp = users[0];
  }

  // Run a graph search to find all linalg.map and that can be fused in
  // the root op.
  DenseSet<Operation*> resultOps;
  SmallVector<Operation*> remainingProducers{rootOp};

  while (!remainingProducers.empty()) {
    Operation* curOp = remainingProducers.pop_back_val();
    if (!curOp) continue;

    if (auto mapOp = dyn_cast<linalg::MapOp>(curOp)) {
      resultOps.insert(curOp);
      for (auto* operand : mapOp.getDpsInputOperands())
        remainingProducers.push_back(operand->get().getDefiningOp());
    } else if (curOp->getName() == op->getName()) {
      for (auto* u : curOp->getUsers())
        // Do not fuse curOp that is used by another op of the same type.
        if (u->getName() == op->getName()) continue;
      resultOps.insert(curOp);
    }
  }
  return {resultOps, rootOp};
}

LogicalResult fuseOutputFill(PatternRewriter& rewriter, Operation* op) {
  // Fusion into the output.
  Operation* definingOp =
      cast<linalg::LinalgOp>(op).getDpsInitOperand(0)->get().getDefiningOp();

  // linalg.fill has already been fused for another matmul.
  if (isa<linalg::FillOp>(definingOp)) return success();

  auto materialize = dyn_cast<MaterializeOp>(definingOp);
  if (!materialize) {
    return rewriter.notifyMatchFailure(
        op, "has failed to 'materialize' output during 'linalg.fill' fusion.");
  }
  if (materialize.getSource().getDefiningOp<linalg::FillOp>()) {
    if (failed(fuse(rewriter, materialize))) return failure();
  }
  return success();
}

FailureOr<Operation*> tileAndFuseGreedily(
    PatternRewriter& rewriter, Operation* op,
    const mlir::gml_st::TilingOptions& opts, StringRef label,
    llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  auto tilingResult = tile(opts, rewriter, cast<TilingInterface>(op));
  if (failed(tilingResult)) return failure();

  // If we did not tile (e.g. when all tile sizes are 0), do not replace
  // original op and just mark it as transformed then return.
  if (tilingResult->loop != nullptr) {
    rewriter.replaceOp(op, tilingResult->loop->getResults());

    // Fuse ops into the loop.
    fuseGreedily(rewriter, *tilingResult->tiledOps.front()->getBlock(),
                 fuseFilterFn);
  }
  setLabel(tilingResult->tiledOps.front(), label);
  return tilingResult->loop;
}

LogicalResult tilePeeledOpsToScalars(
    PatternRewriter& rewriter, const PeelingResult& peelingResult,
    StringRef label, llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  for (auto* loop : peelingResult) {
    ParallelOp peeledLoop = dyn_cast<ParallelOp>(loop);
    auto* terminatorOp = peeledLoop->getRegion(0).front().getTerminator();
    if (!terminatorOp) return failure();

    auto* definingOp = terminatorOp->getOperand(0).getDefiningOp();
    if (!definingOp) return failure();

    mlir::gml_st::TilingOptions opts;
    opts.setTileSizeComputationFn(SmallVector<int64_t>(
        cast<linalg::LinalgOp>(definingOp).getNumLoops(), 1));

    if (failed(tileAndFuseGreedily(rewriter, definingOp, opts, label,
                                   fuseFilterFn)))
      return failure();
  }
  return success();
}

FailureOr<Value> createFusedOp(PatternRewriter& rewriter,
                               MaterializeOp materializeOp) {
  auto tileableOp = materializeOp.getSource().getDefiningOp<TilingInterface>();
  if (!tileableOp) {
    return rewriter.notifyMatchFailure(
        materializeOp, "expected source to be defined by tiling interface op ");
  }

  SmallVector<OpFoldResult> offsets = materializeOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = materializeOp.getMixedSizes();

  // Tile the producer.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(materializeOp);
  FailureOr<Value> tiledProducer = tileableOp.generateResultTileValue(
      rewriter, /*resultNumber=*/0, offsets, sizes);
  if (failed(tiledProducer)) {
    return rewriter.notifyMatchFailure(tileableOp,
                                       "failed to tile the producer");
  }

  return tiledProducer;
}

void populateFusionPatterns(MLIRContext* ctx,
                            function_ref<LogicalResult(MaterializeOp)> filterFn,
                            RewritePatternSet* patterns) {
  patterns->insert<FusionPattern>(ctx, filterFn);
  // clang-format off
  patterns->insert<
      DimOpFissionPattern,
      DimOpReificationPattern>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass(
    StringRef producer, StringRef consumer) {
  return std::make_unique<FusionPass>(producer, consumer);
}

}  // namespace gml_st
}  // namespace mlir
