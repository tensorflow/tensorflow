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
#include <optional>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
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
    // Case tensor::ExtractSliceOp.
    if (auto extractSliceOp = llvm::dyn_cast<tensor::ExtractSliceOp>(def)) {
      assert(extractSliceOp->getNumResults() == 1 && "assume single result");
      auto dimConstantIndex = op.getConstantIndex();
      if (!dimConstantIndex.has_value()) return failure();

      rewriter.replaceOp(op, extractSliceOp.getSizes()[*dimConstantIndex]);
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

class FusionPattern : public OpRewritePattern<tensor::ExtractSliceOp> {
 public:
  FusionPattern(MLIRContext* context,
                function_ref<LogicalResult(tensor::ExtractSliceOp)> filterFn,
                mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExtractSliceOp>(context, benefit),
        filterFn(filterFn) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter& rewriter) const override {
    assert(filterFn && "expect filter function");
    if (failed(filterFn(extractSliceOp)))
      return rewriter.notifyMatchFailure(extractSliceOp, "filtered");

    // If there is an output argument produced by `linalg.fill`, then we can
    // also fuse it into the parallel loop. To check that, we verify that the
    // `src` of `extract_slice` is the bbArg of `gml_st.parallel` and that the
    // corresponding operand of `gml_st.parallel` is defined by `linalg.fill`.
    if (auto bbArg = dyn_cast<BlockArgument>(extractSliceOp.getSource())) {
      if (auto parallelOp =
              dyn_cast_or_null<ParallelOp>(bbArg.getOwner()->getParentOp())) {
        Value loopOperand =
            parallelOp.getOpOperandForRegionOutputArg(bbArg).get();
        if (loopOperand.getDefiningOp<linalg::FillOp>())
          return fuseFillOpsIntoParallelOp(rewriter, parallelOp);
      }
    }
    return fuse(rewriter, extractSliceOp);
  }

 private:
  function_ref<LogicalResult(tensor::ExtractSliceOp)> filterFn;
};

struct FusionPass : public impl::FusionPassBase<FusionPass> {
  FusionPass(StringRef producer, StringRef consumer) {
    this->producerLabel = producer.str();
    this->consumerLabel = consumer.str();
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();

    auto filterFn = [&](tensor::ExtractSliceOp op) {
      Operation* producerOp = op.getSource().getDefiningOp();
      if (auto bbArg = dyn_cast<BlockArgument>(op.getSource())) {
        if (isa<ParallelOp>(bbArg.getOwner()->getParentOp())) return success();
      }
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
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

template <class OpTy>
void eliminateEqualOps(PatternRewriter& rewriter, Block& block) {
  SmallVector<OpTy> uniqueOps;
  for (auto op : llvm::make_early_inc_range(block.getOps<OpTy>())) {
    auto* it = llvm::find_if(
        uniqueOps, [&](OpTy uniqueOp) { return isEqualOp(uniqueOp, op); });
    if (it == uniqueOps.end()) {
      uniqueOps.push_back(op);
    } else {
      rewriter.replaceOp(op, it->getResult());
    }
  }
}

void eliminateTriviallyDeadUsers(PatternRewriter& rewriter, Operation* op) {
  for (auto* user :
       DenseSet<Operation*>(op->getUsers().begin(), op->getUsers().end())) {
    if (isOpTriviallyDead(user)) rewriter.eraseOp(user);
  }
}

void reifyDimOp(PatternRewriter& rewriter, tensor::DimOp dimOp) {
  auto dimValue = dimOp.getSource().template dyn_cast<OpResult>();
  if (!dimValue) return;
  auto rankedShapeTypeOp =
      dyn_cast<ReifyRankedShapedTypeOpInterface>(dimValue.getOwner());
  if (!rankedShapeTypeOp) return;

  std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
  if (!dimIndex) return;

  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(
          rankedShapeTypeOp.reifyResultShapes(rewriter, reifiedResultShapes))) {
    return;
  }

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

LogicalResult fuseTensorCast(PatternRewriter& rewriter, tensor::CastOp castOp,
                             tensor::ExtractSliceOp sliceOp) {
  if (!tensor::canFoldIntoConsumerOp(castOp)) return failure();

  /// Deduce the type of the result to use for the canonicalized operation.
  RankedTensorType resultType =
      tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
          sliceOp.getType().getRank(), sliceOp.getSourceType(),
          sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
          sliceOp.getMixedStrides());
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(sliceOp);
  Value newSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), resultType, castOp.getSource(), sliceOp.getOffsets(),
      sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
      sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
  rewriter.replaceOpWithNewOp<tensor::CastOp>(sliceOp, sliceOp.getType(),
                                              newSlice);
  return success();
}

// Iterates over tensor::ExtractSliceOp inside the block, finds a suitable
// candidate for fusion and fuses it. The fusion candidate should satisfy the
// filter function and not have uses outside of the block. Fails if nothing
// can be fused.
LogicalResult fuseGreedilyOneOpIntoBlock(
    PatternRewriter& rewriter, Block& block,
    llvm::function_ref<bool(Operation*)> filterFn) {
  // Ad-hoc CSE to eliminate duplicate MatrializeOp that could have been added
  // after previous fusions. Running the whole CSE pass would be to expensive
  // here and unnecessary. Without removing those duplicate, some ops will be
  // fused multiple times resulting in exponential code growth.
  eliminateEqualOps<tensor::ExtractSliceOp>(rewriter, block);

  SetVector<Value> valuesFromAbove;
  getUsedValuesDefinedAbove(*block.getParent(), valuesFromAbove);

  for (Value valueFromAbove : valuesFromAbove) {
    auto* fusionCandidate = valueFromAbove.getDefiningOp();
    // Do not fuse if there is no defining op. Of example if it's a
    // materialize from a function argument.
    if (!fusionCandidate) continue;

    if (filterFn && !filterFn(fusionCandidate)) continue;

    // Ad-hoc DCE to trim the fusion candidate from dead users that could have
    // been added in the previous fusion cycles. Normally those ops would be
    // garbage collected after the pattern rewriter driver finished working,
    // but here it requires manual handling.
    eliminateTriviallyDeadUsers(rewriter, fusionCandidate);

    // Push tensor.dim ops 'above' the fusion candidate. This is normally done
    // by canonicalization passes, but running the whole canonicalization
    // pipeline here is too expensive.
    reifyDimOpsUsers(rewriter, fusionCandidate);

    // After the previous steps, extractSliceOp should be only one user of the
    // fusion candidate. Otherwise this candidate should not be fused.
    auto fusionCandidateUsers = llvm::to_vector(fusionCandidate->getUsers());
    if (fusionCandidateUsers.size() != 1) continue;

    Operation* candidateUser = fusionCandidateUsers.front();

    // If the user of the fusion candidate is `tensor.extract_slice`, we use
    // TilingInterface to rewrite `tensor.extract_slice(fusionOp)` into
    // `tiledFusionOp(tensor.extract_slice)`.
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(candidateUser)) {
      if (auto castOp = dyn_cast<tensor::CastOp>(fusionCandidate)) {
        if (succeeded(fuseTensorCast(rewriter, castOp, extractSliceOp))) {
          return success();
        }
        continue;
      }
      if (succeeded(fuse(rewriter, extractSliceOp))) {
        return success();
      }
      continue;
    }

    // TODO(shyshkov): Implement fusion into `tensor.extract` using
    // TilingInterface.
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(candidateUser)) {
      continue;
    }

    // Otherwise, the fusion candidate op is moved inside of the region.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(candidateUser);
    Operation* clonedCandidate = rewriter.clone(*fusionCandidate);
    rewriter.replaceOp(fusionCandidate, clonedCandidate->getResults());
    return success();
  }
  return failure();
}

}  // namespace

FailureOr<Operation*> fuse(PatternRewriter& rewriter,
                           tensor::ExtractSliceOp extractSliceOp) {
  Location loc = extractSliceOp.getLoc();
  FailureOr<Value> fusedOr = createFusedOp(rewriter, extractSliceOp);
  if (failed(fusedOr)) return failure();  // Match failure already notified.

  // Insert cast if needed.
  Value fused = *fusedOr;
  if (fused.getType() != extractSliceOp.getType()) {
    // The result should be a tensor, cast it to the correct shape
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(fused.getDefiningOp());
    fused =
        rewriter.create<tensor::CastOp>(loc, extractSliceOp.getType(), fused);
  }

  rewriter.replaceOp(extractSliceOp, fused);
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
  SetVector<Operation*> resultOps;
  SmallVector<Operation*> remainingProducers{rootOp};

  while (!remainingProducers.empty()) {
    Operation* curOp = remainingProducers.pop_back_val();
    if (!curOp) continue;

    if (auto mapOp = dyn_cast<linalg::MapOp>(curOp)) {
      resultOps.insert(curOp);
      for (auto* operand : mapOp.getDpsInputOperands())
        remainingProducers.push_back(operand->get().getDefiningOp());
    } else if (curOp->getName() == op->getName()) {
      for (auto* u : curOp->getUsers()) {
        // Do not fuse curOp that is used by another op of the same type.
        if (u->getName() == op->getName()) continue;
      }
      resultOps.insert(curOp);
    }
  }
  return {resultOps, rootOp};
}

LogicalResult fuseFillOpsIntoParallelOp(PatternRewriter& rewriter,
                                        ParallelOp parallelOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(parallelOp.getBody());
  bool fillOpsWereFused = false;
  for (OpOperand& output :
       parallelOp->getOpOperands().take_back(parallelOp.getNumOutputs())) {
    auto fillOp = output.get().getDefiningOp<linalg::FillOp>();
    if (!fillOp) continue;

    fillOpsWereFused = true;

    // Clone `linalg.fill` op inside the loop, update the uses of bbArg.
    BlockArgument regionOutputArg =
        parallelOp.getRegionOutputArgForOpOperand(output);
    auto clonedFill = cast<linalg::FillOp>(
        mlir::clone(rewriter, fillOp, fillOp.getResultTypes(),
                    {fillOp.value(), regionOutputArg}));

    output.set(fillOp.output());

    SmallVector<tensor::ExtractSliceOp> sliceOps;
    regionOutputArg.replaceUsesWithIf(
        clonedFill.getResult(0), [&](OpOperand& operand) {
          Operation* owner = operand.getOwner();
          if (auto sliceOp = dyn_cast_or_null<tensor::ExtractSliceOp>(owner))
            sliceOps.push_back(sliceOp);
          return owner != clonedFill && !isa<SetYieldOp>(owner) &&
                 owner->getParentOfType<ParallelOp>() == parallelOp;
        });

    // Use standard fusion logic to swap extract_slice(fill) ->
    // fill(extract_slice).
    for (tensor::ExtractSliceOp sliceOp : sliceOps)
      (void)fuse(rewriter, sliceOp);
  }
  return success(fillOpsWereFused);
}

gml_st::TilingOptions getGmlStTilingOptions(ArrayRef<int64_t> tileSizes) {
  gml_st::TilingOptions opts;
  opts.distribute = true;
  opts.setTileSizeComputationFn(tileSizes);
  return opts;
}

FailureOr<ParallelOp> tileUsingGmlStParallelAndFuseGreedily(
    PatternRewriter& rewriter, Operation* op,
    const mlir::gml_st::TilingOptions& opts, StringRef label,
    llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  assert(opts.distribute == true &&
         "gml_st.for should not be used for CPU pipeline");
  auto tilingResult = tileUsingGmlSt(opts, rewriter, cast<TilingInterface>(op));
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
  return cast<ParallelOp>(tilingResult->loop);
}

scf::SCFTilingOptions getSCFTilingOptions(ArrayRef<int64_t> tileSizes) {
  scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);
  return opts;
}

FailureOr<scf::SCFTilingResult> tileUsingSCFForOpAndFuseGreedily(
    PatternRewriter& rewriter, Operation* op, const scf::SCFTilingOptions& opts,
    StringRef label, llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, opts);
  if (failed(tilingResult)) return failure();

  // If we did not tile (e.g. when all tile sizes are 0), do not replace
  // original op and just mark it as transformed then return.
  if (!tilingResult->loops.empty()) {
    rewriter.replaceOp(op, tilingResult->replacements);

    // Fuse ops into the loop.
    scf::ForOp innerLoop = tilingResult->loops.back();
    fuseGreedily(rewriter, *innerLoop.getBody(), fuseFilterFn);
  }
  setLabel(tilingResult->tiledOps.front(), label);
  return tilingResult;
}

LogicalResult tilePeeledOpsToScalars(
    PatternRewriter& rewriter, const GmlStPeelingResult& peelingResult,
    StringRef label, llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  for (ParallelOp peeledLoop : peelingResult.tailLoops) {
    auto* terminatorOp = peeledLoop->getRegion(0).front().getTerminator();
    if (!terminatorOp) return failure();

    auto* definingOp = terminatorOp->getOperand(0).getDefiningOp();
    if (!definingOp) return failure();

    mlir::gml_st::TilingOptions opts;
    opts.setTileSizeComputationFn(SmallVector<int64_t>(
        cast<linalg::LinalgOp>(definingOp).getNumLoops(), 1));

    if (failed(tileUsingGmlStParallelAndFuseGreedily(rewriter, definingOp, opts,
                                                     label, fuseFilterFn))) {
      return failure();
    }
  }
  return success();
}

FailureOr<gml_st::FusionOp> wrapFusionCluster(
    PatternRewriter& rewriter, const FusionCluster& fusionCluster) {
  auto loc = fusionCluster.root->getLoc();

  // 1. Find operands and results of the cluster op.
  SetVector<Value> clusterOperands;
  SmallVector<Value> clusterResults;
  auto visitOpOperand = [&](OpOperand* operand) {
    auto* definingOp = operand->get().getDefiningOp();

    if (fusionCluster.operations.contains(definingOp)) return;

    if (!isa_and_nonnull<arith::ConstantOp>(definingOp))
      clusterOperands.insert(operand->get());
  };

  for (Operation* op : fusionCluster.operations) {
    for (OpOperand& operand : op->getOpOperands()) visitOpOperand(&operand);

    visitUsedValuesDefinedAbove(op->getRegions(), visitOpOperand);

    for (Value result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation* user) {
            return !fusionCluster.operations.contains(user);
          }))
        clusterResults.push_back(result);
    }
  }

  // We assume that a cluster has only one result for simplity for now. This
  // restriction should be relaxed.
  if (clusterResults.size() != 1) return failure();

  // 2. Create an empty op.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(fusionCluster.root);
  auto fusionClusterOp = rewriter.create<gml_st::FusionOp>(
      loc, TypeRange(ValueRange(clusterResults)),
      clusterOperands.getArrayRef());

  // 3. Create block with mapping between operands and block arguments.
  SmallVector<Type, 4> blockArgTypes =
      llvm::to_vector(TypeRange(ValueRange(clusterOperands.getArrayRef())));
  SmallVector<Location, 4> blockArgLocs(blockArgTypes.size(), loc);

  Region& region = fusionClusterOp.getRegion();
  Block* block =
      rewriter.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);

  IRMapping mapper;
  mapper.map(clusterOperands, block->getArguments());

  // 4. Copy ops into the cluster region in topoligical order to avoid swapping
  // depending ops.
  SmallVector<Operation*> clusterOps(fusionCluster.operations.begin(),
                                     fusionCluster.operations.end());

  mlir::computeTopologicalSorting(clusterOps);
  for (Operation* op : clusterOps) {
    rewriter.clone(*op, mapper);
  }

  auto yieldOp = rewriter.create<gml_st::YieldOp>(
      loc, mapper.lookupOrDefault(clusterResults[0]));

  // 5. Replace all uses of ops in the cluster with results of the new fusion
  // cluster op.
  for (auto [fromV, toV] :
       llvm::zip(clusterResults, fusionClusterOp.getResults())) {
    rewriter.replaceAllUsesExcept(fromV, toV, yieldOp);
  }

  return fusionClusterOp;
}

LogicalResult inlineFusionCluster(FusionOp fusionOp,
                                  PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(fusionOp);

  IRMapping mapper;
  mapper.map(fusionOp.getRegion().getArguments(), fusionOp.getOperands());

  for (auto& op : fusionOp.getBody()->without_terminator()) {
    rewriter.clone(op, mapper);
  }

  rewriter.replaceOp(
      fusionOp, mapper.lookupOrDefault(fusionOp.getTerminator().getOperand()));

  return success();
}

FailureOr<Value> createFusedOp(PatternRewriter& rewriter,
                               tensor::ExtractSliceOp extractSliceOp) {
  Value src = extractSliceOp.getSource();
  if (!src) return failure();
  auto tileableOp = src.getDefiningOp<TilingInterface>();
  if (!tileableOp) {
    return rewriter.notifyMatchFailure(
        extractSliceOp,
        "expected source to be defined by tiling interface op ");
  }

  SmallVector<OpFoldResult> offsets = extractSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = extractSliceOp.getMixedSizes();

  // Tile the producer.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(extractSliceOp);
  FailureOr<Value> tiledProducer = tileableOp.generateResultTileValue(
      rewriter, /*resultNumber=*/0, offsets, sizes);
  if (failed(tiledProducer)) {
    return rewriter.notifyMatchFailure(tileableOp,
                                       "failed to tile the producer");
  }

  return tiledProducer;
}

void populateFusionPatterns(
    MLIRContext* ctx,
    function_ref<LogicalResult(tensor::ExtractSliceOp)> filterFn,
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
