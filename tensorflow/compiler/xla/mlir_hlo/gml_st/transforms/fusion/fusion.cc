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
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/tensor_utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetOperations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

namespace mlir::gml_st {
namespace {

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

  std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
  if (!dimIndex) return;

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(rewriter, dimValue.getOwner(),
                               reifiedResultShapes))) {
    return;
  }

  if (reifiedResultShapes.size() != dimValue.getOwner()->getNumResults())
    return;

  unsigned resultNumber = dimValue.getResultNumber();
  auto sourceType = dimValue.getType().dyn_cast<RankedTensorType>();
  if (reifiedResultShapes[resultNumber].size() !=
      static_cast<size_t>(sourceType.getRank()))
    return;

  rewriter.replaceOp(dimOp, getValueOrCreateConstantIndexOp(
                                rewriter, dimOp.getLoc(),
                                reifiedResultShapes[resultNumber][*dimIndex]));
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

// TODO(vuson): maybe overload this function instead of templating it.
// Fuse a reshape op being used by an extract_slice op (inside a loop) into the
// loop by reversing the order of these two ops (and fixing their
// operands/result accordingly). For example, fusing a tensor.collapse_shape:
//
//   %1 = tensor.collapse_shape %0 [[0], [1, 2]] :
//            tensor<10x10x1xf32> into tensor<10x10xf32>
//   some_scf_loop (%a1, %a2) = (0, 0) to (10, 10) step (1, 8) ...
//     %3 = tensor.extract_slice %1[%a1, %a2] [1, 8] [1, 1] :
//              tensor<10x10xf32> to tensor<1x?xf32>
//
// into
//
//   some_scf_loop (%a1, %a2) = (0, 0) to (10, 10) step (1, 8) ...
//     %3 = tensor.extract_slice %0[%a1, %a2, 0] [1, 8, 1] [1, 1, 1] :
//              tensor<10x10x1xf32> to tensor<1x?x1xf32>
//     %1 = tensor.collapse_shape %3 [[0], [1, 2]] :
//              tensor<1x?x1xf32> into tensor<1x?xf32>
//
// This also works for tensor.expand_shape instead of tensor.collapse_shape:
//
//   %1 = tensor.expand_shape %0 [[0], [1, 2]] :
//            tensor<10x10xf32> into tensor<10x10x1xf32>
//   some_scf_loop (%a1, %a2) = (0, 0) to (10, 10) step (1, 8) ...
//     %3 = tensor.extract_slice %1[%a1, %a2, 0] [1, 8, 1] [1, 1, 1] :
//              tensor<10x10x1xf32> to tensor<1x?x1xf32>
//
// into
//
//   some_scf_loop (%a1, %a2) = (0, 0) to (10, 10) step (1, 8) ...
//     %3 = tensor.extract_slice %0[%a1, %a2] [1, 8] [1, 1] :
//              tensor<10x10xf32> to tensor<1x?xf32>
//     %1 = tensor.expand_shape %2 [[0], [1, 2]] :
//              tensor<1x?xf32> into tensor<1x?x1xf32>
template <typename TensorReshapeOp>
LogicalResult fuseTensorReshape(PatternRewriter& rewriter,
                                TensorReshapeOp reshapeOp,
                                tensor::ExtractSliceOp sliceOp) {
  if (!isDegenerateReshapeOp(reshapeOp)) return failure();

  auto newSliceSrcType = reshapeOp.getSrcType();
  llvm::ArrayRef<int64_t> newSliceSrcShape = newSliceSrcType.getShape();
  auto newSliceRank = newSliceSrcType.getRank();
  // If the source type of reshape op is a rank-0 tensor, there will be no
  // extract_slice possible from that source value, let's bail out.
  if (newSliceRank == 0) return failure();

  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> newOffsets(newSliceRank, zero);
  SmallVector<OpFoldResult> newSizes(newSliceRank, one);
  SmallVector<OpFoldResult> newStrides(newSliceRank, one);

  llvm::ArrayRef<int64_t> sliceSrcShape = sliceOp.getSourceType().getShape();
  auto reassociation = reshapeOp.getReassociationIndices();
  constexpr bool isExpanding =
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value;
  llvm::ArrayRef<int64_t> shape =
      isExpanding ? sliceSrcShape : newSliceSrcShape;
  // For each reassociation indices, a degenerate reshape op only has at most
  // 1 non-unit-dimension. If there is none, it means the source shape already
  // has some unit-dimensions (e.g. tensor<1x1x1xf32> collapsed into
  // tensor<1xf32>)
  assert(
      static_cast<size_t>(
          llvm::count_if(reassociation,
                         [&](auto indices) {
                           return llvm::count_if(indices, [&](int64_t idx) {
                                    return shape[idx] != 1;
                                  }) <= 1;
                         })) == reassociation.size() &&
      "Degenerate reshape op should only have at most 1 non-unit dimension for "
      "each reassociation indices");
  for (const auto& [enumIdx, indices] : llvm::enumerate(reassociation)) {
    auto findIt =
        llvm::find_if(indices, [&](int64_t idx) { return shape[idx] != 1; });
    // No non-unit dimension, which means the source shape already has some
    // unit-dimensions. The default values for offset/size/stride (0/1/1) should
    // be usable. Skip updating offset/size/stride for this dimension.
    if (findIt == indices.end()) continue;
    auto newIdx = isExpanding ? enumIdx : *findIt;
    auto idx = isExpanding ? *findIt : enumIdx;
    newOffsets[newIdx] = sliceOp.getMixedOffsets()[idx];
    newSizes[newIdx] = sliceOp.getMixedSizes()[idx];
    newStrides[newIdx] = sliceOp.getMixedStrides()[idx];
  }

  RankedTensorType newSliceResultType =
      tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
          newSliceRank, newSliceSrcType, newOffsets, newSizes, newStrides);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(sliceOp);
  auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), newSliceResultType, reshapeOp.getSrc(), newOffsets,
      newSizes, newStrides);

  rewriter.replaceOpWithNewOp<TensorReshapeOp>(sliceOp, sliceOp.getResultType(),
                                               newSlice, reassociation);
  return success();
}

// Checks that there is at most one user in each given block.
bool atMostOneUserPerBlock(ArrayRef<Operation*> users,
                           ArrayRef<Block*> blocks) {
  if (users.size() == 1) return true;
  if (users.size() > blocks.size()) return false;

  llvm::SmallSetVector<Block*, 2> blocksWithUsers;

  Block* funcBlock = &users.front()
                          ->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
                          ->getRegion(0)
                          .front();
  // Return false if there two users in a block.
  for (Operation* user : users) {
    Block* block = user->getBlock();
    while (block != funcBlock) {
      if (llvm::is_contained(blocks, block)) {
        if (!blocksWithUsers.insert(block)) return false;
      }
      block = block->getParentOp()->getBlock();
    }
  }
  return true;
}

DenseSet<Operation*> getFusionCandidates(
    Region& region, llvm::function_ref<bool(Operation*)> filterFn) {
  DenseSet<Operation*> fusionCandidates;
  visitUsedValuesDefinedAbove(region, [&](OpOperand* operand) {
    auto* fusionCandidate = operand->get().getDefiningOp();
    // Do not fuse if there is no defining op. Of example, if it's an
    // extract_slice from a function argument.
    if (!fusionCandidate) return;

    // Filter candidates that we don't want to fuse.
    if (filterFn && !filterFn(fusionCandidate)) return;

    // Check that the candidate doesn't have users that will block fusion.
    if (!llvm::all_of(fusionCandidate->getUsers(), [](Operation* op) {
          // Fusion candidates can only be fused into tensor.extract_slice or
          // tensor.extract.
          return isa<tensor::ExtractSliceOp, tensor::ExtractOp>(op) ||
                 // tensor.dim is pushed 'above' the fusion candidate.
                 isa<tensor::DimOp>(op) ||
                 // Trivially dead ops will be removed.
                 isOpTriviallyDead(op);
        })) {
      return;
    }
    fusionCandidates.insert(fusionCandidate);
  });
  return fusionCandidates;
}

LogicalResult fuseIntoUser(PatternRewriter& rewriter,
                           Operation* fusionCandidate,
                           Operation* candidateUser) {
  // If the user of the fusion candidate is `tensor.extract_slice`, we use
  // TilingInterface to rewrite `tensor.extract_slice(fusionOp)` into
  // `tiledFusionOp(tensor.extract_slice)`.
  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(candidateUser)) {
    if (auto castOp = dyn_cast<tensor::CastOp>(fusionCandidate)) {
      return fuseTensorCast(rewriter, castOp, extractSliceOp);
    }
    if (auto collapseShapeOp =
            dyn_cast<tensor::CollapseShapeOp>(fusionCandidate)) {
      return fuseTensorReshape(rewriter, collapseShapeOp, extractSliceOp);
    }
    if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(fusionCandidate)) {
      return fuseTensorReshape(rewriter, expandShapeOp, extractSliceOp);
    }
    auto fusedOp = fuse(rewriter, extractSliceOp);
    if (succeeded(fusedOp)) {
      setLabel(*fusedOp, kTransformedLabel);
      return success();
    }
    return failure();
  }

  // TODO(shyshkov): Implement fusion into `tensor.extract` using
  // TilingInterface.
  if (auto extractOp = dyn_cast<tensor::ExtractOp>(candidateUser)) {
    return failure();
  }

  // Otherwise, the fusion candidate op is moved inside of the region.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateUser);
  Operation* clonedCandidate = rewriter.clone(*fusionCandidate);
  rewriter.replaceOp(fusionCandidate, clonedCandidate->getResults());
  return success();
}

LogicalResult fuseIntoUsers(PatternRewriter& rewriter,
                            Operation* fusionCandidate,
                            ArrayRef<Operation*> fusionCandidateUsers) {
  for (Operation* candidateUser : fusionCandidateUsers) {
    if (failed(fuseIntoUser(rewriter, fusionCandidate, candidateUser)))
      return failure();
  }
  return success();
}

// Iterates over tensor::ExtractSliceOp inside the block, finds a suitable
// candidate for fusion and fuses it. The fusion candidate should satisfy the
// filter function and not have uses outside of the block. Fails if nothing
// can be fused.
LogicalResult fuseGreedilyOneOpIntoBlock(
    PatternRewriter& rewriter, ArrayRef<Block*> blocks,
    llvm::function_ref<bool(Operation*)> filterFn) {
  if (blocks.empty()) return failure();

  // Ad-hoc CSE to eliminate duplicate MatrializeOp that could have been added
  // after previous fusions. Running the whole CSE pass would be to expensive
  // here and unnecessary. Without removing those duplicate, some ops will be
  // fused multiple times resulting in exponential code growth.
  DenseSet<Operation*> fusionCandidates;
  for (auto [index, block] : llvm::enumerate(blocks)) {
    eliminateEqualOps<tensor::ExtractSliceOp>(rewriter, *block);

    if (index == 0) {
      fusionCandidates = getFusionCandidates(*block->getParent(), filterFn);
      continue;
    }
    llvm::set_intersect(fusionCandidates,
                        getFusionCandidates(*block->getParent(), filterFn));
  }

  for (Operation* fusionCandidate : fusionCandidates) {
    // Ad-hoc DCE to trim the fusion candidate from dead users that could have
    // been added in the previous fusion cycles. Normally those ops would be
    // garbage collected after the pattern rewriter driver finished working,
    // but here it requires manual handling.
    eliminateTriviallyDeadUsers(rewriter, fusionCandidate);

    // Push tensor.dim ops 'above' the fusion candidate. This is normally done
    // by canonicalization passes, but running the whole canonicalization
    // pipeline here is too expensive.
    reifyDimOpsUsers(rewriter, fusionCandidate);

    // After the previous steps, there should be at most one user of the
    // fusion candidate per block. Otherwise this candidate should not be fused.
    // We always want to fuse linalg.fill.
    auto fusionCandidateUsers = llvm::to_vector(fusionCandidate->getUsers());
    if (!atMostOneUserPerBlock(fusionCandidateUsers, blocks)) {
      continue;
    }
    if (succeeded(
            fuseIntoUsers(rewriter, fusionCandidate, fusionCandidateUsers))) {
      return success();
    }
  }
  return failure();
}

FailureOr<TilingResult> createFusedOp(PatternRewriter& rewriter,
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
  FailureOr<TilingResult> tiledProducer = tileableOp.generateResultTileValue(
      rewriter, /*resultNumber=*/0, offsets, sizes);
  if (failed(tiledProducer)) {
    return rewriter.notifyMatchFailure(tileableOp,
                                       "failed to tile the producer");
  }

  return tiledProducer;
}

void fuseFillOpsIntoForallOp(PatternRewriter& rewriter,
                             scf::ForallOp parallelOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(parallelOp.getBody());
  for (OpOperand& output :
       parallelOp->getOpOperands().take_back(parallelOp.getNumResults())) {
    auto fillOp = output.get().getDefiningOp<linalg::FillOp>();
    if (!fillOp) continue;

    // Clone `linalg.fill` op inside the loop, update the uses of bbArg.
    BlockArgument regionOutputArg = parallelOp.getTiedBlockArgument(&output);
    auto clonedFill = cast<linalg::FillOp>(
        mlir::clone(rewriter, fillOp, fillOp.getResultTypes(),
                    {fillOp.value(), regionOutputArg}));

    output.set(fillOp.output());
    setLabel(clonedFill, kTransformedLabel);

    SmallVector<tensor::ExtractSliceOp> sliceOps;
    regionOutputArg.replaceUsesWithIf(
        clonedFill.getResult(0), [&](OpOperand& operand) {
          Operation* owner = operand.getOwner();
          if (auto sliceOp = dyn_cast_or_null<tensor::ExtractSliceOp>(owner))
            sliceOps.push_back(sliceOp);
          return owner != clonedFill &&
                 !isa<tensor::ParallelInsertSliceOp>(owner) &&
                 owner->getParentOfType<scf::ForallOp>() == parallelOp;
        });

    // Use standard fusion logic to swap extract_slice(fill) ->
    // fill(extract_slice).
    for (tensor::ExtractSliceOp sliceOp : sliceOps)
      (void)fuse(rewriter, sliceOp);
  }
}

// Finds the source of the operand. It could be a tensor.empty, a region arg or
// an op outside of the cluster.
Value getTiedSourceOp(PatternRewriter& rewriter, OpOperand* operand,
                      const FusionCluster& fusionCluster) {
  auto* definingOp = operand->get().getDefiningOp();
  if (!definingOp) return operand->get();

  // A tensor.empty used tied to fusion cluster result should not be fused, so
  // bufferization can properly handle allocations. If the same tensor.empty is
  // used in other ops for temporary result, it should be fused. Copied op is
  // not in the cluster, so it will not be fused.
  if (auto emptyOp = dyn_cast<tensor::EmptyOp>(definingOp)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(emptyOp);

    auto newEmptyOp = cast<tensor::EmptyOp>(rewriter.clone(*emptyOp));
    operand->set(newEmptyOp);
    return newEmptyOp;
  }

  // Source of the operand is outside of the cluster, so pass it as an argument.
  if (!llvm::is_contained(fusionCluster.operations, definingOp)) {
    return operand->get();
  }

  // Source of the operand is another DPS op from the cluster. Look higher in
  // the chain.
  if (auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(definingOp)) {
    OpOperand* tiedOperand =
        dstStyleOp.getTiedOpOperand(operand->get().dyn_cast<OpResult>());
    return getTiedSourceOp(rewriter, tiedOperand, fusionCluster);
  }

  return operand->get();
}

SmallVector<Value> getRootOpInitOperands(PatternRewriter& rewriter,
                                         const FusionCluster& fusionCluster) {
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(fusionCluster.root);
  if (!dstStyleOp) return {};

  SmallVector<Value> initOperands;

  for (auto* operand : dstStyleOp.getDpsInitOperands()) {
    initOperands.push_back(getTiedSourceOp(rewriter, operand, fusionCluster));
  }

  return initOperands;
}

}  // namespace

FailureOr<Operation*> fuse(PatternRewriter& rewriter,
                           tensor::ExtractSliceOp extractSliceOp) {
  Location loc = extractSliceOp.getLoc();
  FailureOr<TilingResult> fusedOr = createFusedOp(rewriter, extractSliceOp);
  if (failed(fusedOr)) return failure();  // Match failure already notified.

  // Insert cast if needed.
  Value fused = fusedOr->tiledOps.front()->getResult(0);
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

void fuseGreedily(PatternRewriter& rewriter, ArrayRef<Block*> blocks,
                  llvm::function_ref<bool(Operation*)> filterFn) {
  while (succeeded(fuseGreedilyOneOpIntoBlock(rewriter, blocks, filterFn)))
    ;
}

// Cluster producers and consumers around the root op.
FusionCluster getFusionCluster(
    Operation* op, llvm::function_ref<bool(Operation*)> producerFilterFn,
    llvm::function_ref<bool(Operation*)> consumerFilterFn) {
  // Find a chain of users and use the last one as a root of cluster.
  SetVector<Operation*> resultOps;

  Operation* rootOp = op;
  while (true) {
    auto users = llvm::to_vector(rootOp->getUsers());

    if (users.size() != 1) break;

    if (!consumerFilterFn(users[0])) break;
    resultOps.insert(rootOp);

    rootOp = users[0];
  }

  // Run DFS to find all ops that satisfy producerFilterFn.
  SmallVector<Operation*> remainingProducers;
  for (Value operand : op->getOperands())
    remainingProducers.push_back(operand.getDefiningOp());

  while (!remainingProducers.empty()) {
    Operation* curOp = remainingProducers.pop_back_val();
    if (!curOp || resultOps.contains(curOp)) continue;
    if (curOp == op || producerFilterFn(curOp)) {
      resultOps.insert(curOp);
      for (Value operand : curOp->getOperands())
        remainingProducers.push_back(operand.getDefiningOp());
    }
  }
  return {resultOps, rootOp, {}};
}

FailureOr<GMLSTTilingResult> tileUsingSCFForallOpAndFuseGreedily(
    PatternRewriter& rewriter, Operation* op, const scf::SCFTilingOptions& opts,
    llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  auto tilingResult =
      tileUsingSCFForallOp(rewriter, cast<TilingInterface>(op), opts);
  if (failed(tilingResult)) return failure();

  for (Operation* tiledOp : tilingResult->tiledOps)
    setLabel(tiledOp, kTransformedLabel);

  // If tiling created an `scf.forall` loop, we fuse.
  if (tilingResult->loop != nullptr) {
    rewriter.replaceOp(op, tilingResult->loop->getResults());
    // Fuse ops into the loop.
    fuseGreedily(rewriter, tilingResult->tiledOps.front()->getBlock(),
                 fuseFilterFn);
    fuseFillOpsIntoForallOp(rewriter, tilingResult->loop);
  }
  return tilingResult;
}

FailureOr<scf::SCFTilingResult> tileUsingSCFForOpAndFuseGreedily(
    PatternRewriter& rewriter, Operation* op, const scf::SCFTilingOptions& opts,
    llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, opts);
  if (failed(tilingResult)) return failure();
  rewriter.replaceOp(op, tilingResult->replacements);

  for (Operation* tiledOp : tilingResult->tiledOps)
    setLabel(tiledOp, kTransformedLabel);

  // If tiling created an `scf.for` loop nest, we fuse.
  if (!tilingResult->loops.empty()) {
    scf::ForOp innerLoop = tilingResult->loops.back();
    fuseGreedily(rewriter, innerLoop.getBody(), fuseFilterFn);
  }
  return tilingResult;
}

LogicalResult tilePeeledOpsToScalars(
    PatternRewriter& rewriter, const GmlStPeelingResult& peelingResult,
    llvm::function_ref<bool(Operation*)> fuseFilterFn) {
  for (scf::ForallOp peeledLoop : peelingResult.tailLoops) {
    SmallVector<Value> yieldedTensors =
        getYieldedValues(peeledLoop.getTerminator());

    assert(yieldedTensors.size() == 1 &&
           "expected to have a single result in scf.forall loop");
    auto definingOp = yieldedTensors.front().getDefiningOp<TilingInterface>();
    if (!definingOp) return failure();

    auto opts = getSCFTilingOptions(
        SmallVector<int64_t>(definingOp.getLoopIteratorTypes().size(), 1));
    if (failed(tileUsingSCFForallOpAndFuseGreedily(rewriter, definingOp, opts,
                                                   fuseFilterFn))) {
      return failure();
    }
  }
  return success();
}

FailureOr<gml_st::FusionOp> wrapFusionCluster(
    PatternRewriter& rewriter, const FusionCluster& fusionCluster) {
  auto loc = fusionCluster.root->getLoc();

  SmallVector<Value> initOperands =
      getRootOpInitOperands(rewriter, fusionCluster);

  // 1. Find operands and results of the cluster op.
  SetVector<Value> clusterOperands;
  SmallVector<Value> clusterResults;
  SmallVector<Value> constantOps;
  auto visitOpOperand = [&](OpOperand* operand) {
    auto* definingOp = operand->get().getDefiningOp();

    if (auto constantOp = dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
      constantOps.push_back(constantOp);
      return;
    }

    if (fusionCluster.operations.contains(definingOp)) return;
    if (llvm::is_contained(initOperands, operand->get())) return;

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

  clusterOperands.insert(initOperands.begin(), initOperands.end());

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

  // 4. Copy ops into the cluster region.
  // 4.1. Copy arith.constant ops.
  for (auto v : constantOps) {
    auto newOp = cast<arith::ConstantOp>(rewriter.clone(*v.getDefiningOp()));
    mapper.map(v, newOp);
  }

  // 4.2. Copy ops into the cluster region in topoligical order to avoid
  // swapping depending ops.
  SmallVector<Operation*> clusterOps(fusionCluster.operations.begin(),
                                     fusionCluster.operations.end());

  mlir::computeTopologicalSorting(clusterOps);
  for (Operation* op : clusterOps) {
    rewriter.clone(*op, mapper);
  }

  // 4.3 Create terminator gml_st.yield.
  SmallVector<Value> yieldOpOperands = llvm::to_vector(llvm::map_range(
      clusterResults, [&](Value v) { return mapper.lookupOrDefault(v); }));
  auto yieldOp = rewriter.create<gml_st::YieldOp>(loc, yieldOpOperands);

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

  SmallVector<Value> yieldOpOperands = llvm::to_vector(
      llvm::map_range(fusionOp.getTerminator().getOperands(),
                      [&](Value v) { return mapper.lookupOrDefault(v); }));

  rewriter.replaceOp(fusionOp, yieldOpOperands);

  return success();
}

}  // namespace mlir::gml_st
