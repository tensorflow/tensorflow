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

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/tensor_utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMELEMENTWISEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

// Indicates the the dimension is not mapped to dimensions of the root op.
constexpr int64_t kNotMappedToRootDims = -1;

using FusionFilterFn = llvm::function_ref<bool(Operation *)>;
using CandidatesMap = llvm::SmallMapVector<Value, SmallVector<int64_t>, 4>;

// Find the root of the fusion cluster.
Operation *findRootElementwiseOp(Operation *op, FusionFilterFn fusionFilterFn) {
  Operation *rootOp = op;
  Operation *curOp = nullptr;
  do {
    curOp = nullptr;
    for (OpOperand &use : rootOp->getUses()) {
      Operation *owner = use.getOwner();
      if (!fusionFilterFn(owner)) continue;
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner)) {
        if (llvm::is_contained(dpsOp.getDpsInitOperands(), &use)) continue;
      }
      curOp = owner;
      rootOp = curOp;
      break;
    }
  } while (curOp != nullptr);
  // If the root is a reshape, don't use it, use the defining op for the
  // argument instead.
  if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(rootOp))
    return rootOp->getOperand(0).getDefiningOp();
  return rootOp;
}

// Depending on the type of the defining op for the `result`, adds its arguments
// with the maps to the root result dimensions.
void addMappedTensorArgs(Value result, const SmallVector<int64_t> &map,
                         CandidatesMap &args) {
  Operation *defOp = result.getDefiningOp();
  if (!defOp) return;

  mlir::TypeSwitch<Operation *>(defOp)
      .Case<linalg::FillOp, linalg::MapOp, thlo::ReverseOp>([&](auto op) {
        for (OpOperand *operand :
             cast<DestinationStyleOpInterface>(op.getOperation())
                 .getDpsInputOperands()) {
          Value val = operand->get();
          if (!isa<RankedTensorType>(val.getType())) continue;
          args[val] = map;
        }
      })
      .Case<linalg::TransposeOp>([&](auto op) {
        auto transposeOp = cast<linalg::TransposeOp>(op);
        SmallVector<int64_t> composed(map.size(), 0);
        for (auto [index, id] : llvm::enumerate(transposeOp.getPermutation())) {
          composed[index] = map[id];
        }
        args[transposeOp.getInput()] = composed;
      })
      .Case<linalg::BroadcastOp>([&](auto op) {
        auto broadcastOp = cast<linalg::BroadcastOp>(op);
        SmallVector<int64_t> composed;
        SmallVector<int64_t> bcastDims = to_vector(broadcastOp.getDimensions());

        for (auto [index, id] : llvm::enumerate(map)) {
          if (llvm::is_contained(bcastDims, index)) continue;
          composed.push_back(id);
        }
        args[broadcastOp.getInput()] = composed;
      })
      .Case<tensor::CollapseShapeOp>([&](auto op) {
        auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
        auto srcType = collapseShapeOp.getSrcType();

        SmallVector<int64_t> preservedDims = getPreservedDimensions(
            srcType.getShape(), collapseShapeOp.getReassociationIndices());

        SmallVector<int64_t> composed(srcType.getRank(), kNotMappedToRootDims);
        for (auto [index, mapDim] : llvm::enumerate(map))
          composed[preservedDims[index]] = mapDim;
        args[collapseShapeOp.getSrc()] = composed;
      })
      .Case<tensor::ExpandShapeOp>([&](auto op) {
        auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
        auto dstType = expandShapeOp.getResultType();

        SmallVector<int64_t> preservedDims = getPreservedDimensions(
            dstType.getShape(), expandShapeOp.getReassociationIndices());

        SmallVector<int64_t> composed(expandShapeOp.getSrcType().getRank());
        for (auto [index, preservedDim] : llvm::enumerate(preservedDims))
          composed[index] = map[preservedDim];
        args[expandShapeOp.getSrc()] = composed;
      })
      .Default(
          [](Operation *) { llvm_unreachable("The op is not supported"); });
}

// Starts a graph traversal from the root trying to fuse all ops that satisfy
// `fusionFilterFn` and also have no users outside of this fusion cluster.
FusionCluster findElementwiseCluster(Operation *rootOp,
                                     FusionFilterFn fusionFilterFn) {
  Value rootResult = rootOp->getResult(0);

  SetVector<Operation *> resultOps;
  resultOps.insert(rootOp);
  CandidatesMap mappedArgs, candidates;

  // Add operands of root.
  int64_t rootRank = rootResult.getType().cast<RankedTensorType>().getRank();
  auto identityMap = llvm::to_vector(llvm::seq<int64_t>(0, rootRank));
  addMappedTensorArgs(rootResult, identityMap, candidates);

  while (!candidates.empty()) {
    bool fusionHappened = false;
    SmallVector<Value> argsToErase;
    for (auto [arg, map] : llvm::reverse(candidates)) {
      // If the arg is already coming outside of the cluster, i.e. it is a
      // function argument or a result of some op that is not included by the
      // fusionFilterFn, then we remove such arg.
      Operation *defOp = arg.getDefiningOp();
      if (mappedArgs.contains(arg) || !defOp || resultOps.contains(defOp) ||
          !fusionFilterFn(defOp)) {
        mappedArgs[arg] = map;
        argsToErase.push_back(arg);
        continue;
      }

      // If there are any users of this op outside of this op outside of fusion
      // cluster, then skip.
      if (llvm::any_of(arg.getUsers(), [&](Operation *user) {
            return !resultOps.contains(user);
          })) {
        continue;
      }

      resultOps.insert(defOp);
      addMappedTensorArgs(arg, map, candidates);
      fusionHappened = true;
      break;
    }
    for (Value argToErase : argsToErase) {
      candidates.erase(argToErase);
    }

    // If an op to fuse was not found, we add all current candidates  to the
    // result.
    if (!fusionHappened) {
      for (auto &candidate : candidates) {
        mappedArgs.insert(candidate);
      }
      break;
    }
  }
  FusionCluster fusionCluster;
  fusionCluster.root = rootOp;
  fusionCluster.operations = std::move(resultOps);
  llvm::append_range(fusionCluster.argDimsMapping, mappedArgs);
  return fusionCluster;
}

// Searches through the inner-most dimensions of the arguments of the fusion
// cluster to find the most beneficial dimension to tile. Default tile size is 1
// x ... x 1 x vector_size, which leads to vector.transfer_write to the init
// tensor.
// In case of broadcast, transpose and other maps with the non-identity mapping
// between op input and op result the innermost dimension of the input can be
// different from the one of result.
SmallVector<int64_t> optimizeTileSizes(const FusionCluster &fusionCluster,
                                       int64_t vectorSize) {
  auto rootTy =
      cast<RankedTensorType>(fusionCluster.root->getResultTypes().front());

  if (rootTy.getRank() == 0) return {};
  SmallVector<int64_t> tileSizes(rootTy.getRank(), 1);
  tileSizes.back() = vectorSize;

  int64_t rootInnermostDim = rootTy.getRank() - 1;
  int64_t innermostDimWithMostElements = rootInnermostDim;
  int64_t innermostDimMaxElements = std::numeric_limits<int64_t>::min();
  for (auto &[arg, map] : fusionCluster.argDimsMapping) {
    auto argInnermostDimIt = llvm::find_if(
        llvm::reverse(map),
        [](int64_t item) { return item != kNotMappedToRootDims; });
    if (argInnermostDimIt == map.rend()) continue;
    int64_t argInnermostDim = *argInnermostDimIt;
    if (argInnermostDim == rootInnermostDim) continue;

    int64_t numElements = rootTy.getDimSize(argInnermostDim);
    if (innermostDimMaxElements >= numElements &&
        !ShapedType::isDynamic(numElements))
      continue;
    innermostDimMaxElements = numElements;
    innermostDimWithMostElements = argInnermostDim;
  }
  tileSizes[innermostDimWithMostElements] = vectorSize;
  return tileSizes;
}

template <typename OpTy>
struct TileElementwisePattern : public OpRewritePattern<OpTy> {
  TileElementwisePattern(MLIRContext *context, int64_t vectorSize,
                         bool fuseDegenerateReshapes,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit),
        vectorSize(vectorSize),
        fuseDegenerateReshapes(fuseDegenerateReshapes) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (hasSingleElementOperandsAndResults(op)) return failure();
    if (hasLabel(op, kTransformedLabel)) return failure();

    if (isa<scf::ForallOp, scf::ForOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");
    }

    // Find the root from which to start tiling and fusion.
    auto fusionFilterFn = [&](Operation *op) {
      if (fuseDegenerateReshapes) {
        if (auto reshapeOp = dyn_cast<tensor::CollapseShapeOp>(op))
          return isDegenerateReshapeOp(reshapeOp);
        if (auto reshapeOp = dyn_cast<tensor::ExpandShapeOp>(op))
          return isDegenerateReshapeOp(reshapeOp);
      }
      // Add thlo.concatenate here.
      return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp,
                 linalg::TransposeOp, thlo::ReverseOp>(op);
    };
    Operation *fusionRoot = findRootElementwiseOp(op, fusionFilterFn);

    // Find the fusion cluster and its arguments.
    FusionCluster fusionCluster =
        findElementwiseCluster(fusionRoot, fusionFilterFn);

    // Find what dimensions to tile.
    SmallVector<int64_t> tileSizes =
        optimizeTileSizes(fusionCluster, vectorSize);

    // Tile and fuse.
    auto tiledLoop = tileUsingSCFForallOpAndFuseGreedily(
        rewriter, op, getSCFTilingOptions(tileSizes),
        [&](Operation *op) { return fusionCluster.operations.contains(op); });
    if (failed(tiledLoop)) return failure();

    // Peel.
    auto peelingResult = peelAllLoops(tiledLoop->loop, rewriter);
    setLabel(tiledLoop->loop, kPerfectlyTiledLoopLabel);

    // Tile ops in the peeled loop again, to size 1, so they can be
    // scalarized.
    return tilePeeledOpsToScalars(rewriter, peelingResult, fusionFilterFn);
  }

 private:
  int64_t vectorSize;
  bool fuseDegenerateReshapes;
};

struct TransformElementwiseForCpuPass
    : public impl::TransformElementwiseForCpuPassBase<
          TransformElementwiseForCpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // clang-format off
    patterns.add<
      TileElementwisePattern<linalg::BroadcastOp>,
      TileElementwisePattern<linalg::FillOp>,
      TileElementwisePattern<linalg::MapOp>,
      TileElementwisePattern<linalg::TransposeOp>,
      TileElementwisePattern<thlo::ReverseOp>
    >(ctx, vectorSize, fuseDegenerateReshapes);
    // clang-format on

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
    f.walk([](linalg::MapOp op) { removeLabel(op, kTransformedLabel); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformElementwiseForCpuPass(int64_t vectorSize,
                                     bool fuseDegenerateReshapes) {
  TransformElementwiseForCpuPassOptions opts;
  opts.vectorSize = vectorSize;
  opts.fuseDegenerateReshapes = fuseDegenerateReshapes;
  return std::make_unique<mlir::gml_st::TransformElementwiseForCpuPass>(opts);
}

}  // namespace mlir::gml_st
