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

#include <iterator>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONPLANNINGFORCPUPASS
#define GEN_PASS_DEF_INLINEFUSIONCLUSTERSPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kFusionPlanningLabel =
    "__fusion_planning_label__";

// Returns true is consumer and producer should be fused and tiled together.
bool allowedToFuse(Operation* consumerOp, Operation* producerOp) {
  if (isa<thlo::ScatterOp, thlo::SortOp>(producerOp)) return false;

  if (isa<linalg::FillOp>(producerOp)) {
    auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(consumerOp);
    if (!dstStyleOp) return false;

    if (llvm::any_of(dstStyleOp.getDpsInitOperands(), [&](OpOperand* operand) {
          return operand->get().getDefiningOp() == producerOp;
        }))
      return true;
  }

  if (isa<linalg::MapOp, thlo::ReverseOp>(consumerOp)) return true;
  if (isa<linalg::BroadcastOp>(consumerOp)) return false;

  if (isa<linalg::ReduceOp>(consumerOp))
    return isa<linalg::MapOp, linalg::BroadcastOp, thlo::ReverseOp>(producerOp);
  if (isa<linalg::MatmulOp>(consumerOp))
    return isa<linalg::BroadcastOp, thlo::ReverseOp>(producerOp);
  if (isa<linalg::FillOp>(consumerOp)) return isa<tensor::EmptyOp>(producerOp);
  return false;
}

// Runs graph search to find ops that can be fused together.
template <typename OpTy>
LogicalResult fusionPattern(OpTy op, PatternRewriter& rewriter) {
  // The op is already in a fusion cluster.
  if (isa<gml_st::FusionOp>(op.getOperation()->getParentOp())) return failure();

  // The op was already processed.
  if (hasLabel(op, kFusionPlanningLabel)) return failure();

  for (auto& use : op->getUses()) {
    auto* useOp = use.getOwner();
    // This op can be potentially fused into one of the consumens. Wait until
    // that other op is processed.
    if (useOp && allowedToFuse(useOp, op.getOperation())) return failure();
  }

  SetVector<Operation*> resultOps;
  SmallVector<Operation*> remainingProducers;
  resultOps.insert(op.getOperation());
  for (auto operand : op.getOperands())
    remainingProducers.push_back(operand.getDefiningOp());

  while (!remainingProducers.empty()) {
    Operation* curOp = remainingProducers.pop_back_val();
    if (!curOp) continue;

    if (llvm::is_contained(resultOps, curOp)) continue;

    if (!llvm::all_of(curOp->getUses(), [&](mlir::OpOperand& use) {
          auto* consumerOp = use.getOwner();
          // Check that curOp is allowed to fused with all consumers.
          if (!allowedToFuse(consumerOp, curOp)) return false;
          // Check that all consumers are already in the fusion cluster.
          if (!llvm::is_contained(resultOps, consumerOp)) return false;
          return true;
        }))
      continue;

    resultOps.insert(curOp);

    for (auto operand : curOp->getOperands())
      remainingProducers.push_back(operand.getDefiningOp());
  }

  FusionCluster fusionCluster;
  fusionCluster.root = op;
  fusionCluster.operations = resultOps;
  if (failed(wrapFusionCluster(rewriter, fusionCluster))) return failure();

  // Mark all ops as processed.
  for (auto* op : resultOps) setLabel(op, kFusionPlanningLabel);

  return success();
}

// Duplicate linalg.fill op with rank-0 tensors results that have multiple
// users. If linalg.fill is used inside and outside of a fusion cluster, it will
// not be fused and can break some other passes that expect linalg.reduce inits
// to be linalg.fill.
LogicalResult copyConstantLikeFillOp(linalg::FillOp fillOp,
                                     PatternRewriter& rewriter) {
  // Only modify ops that fill rank-0 tensors.
  if (fillOp.getRank(fillOp.getDpsInitOperand(0)) != 0) return failure();

  // Nothing to do, because the op has 0 or 1 users.
  if (std::distance(fillOp->user_begin(), fillOp->user_end()) <= 1)
    return failure();

  for (auto& use : fillOp->getUses()) {
    Operation* ownerOp = use.getOwner();

    auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(ownerOp);
    if (!dstStyleOp || !dstStyleOp.isDpsInit(&use)) continue;

    auto newFillOp = cast<linalg::FillOp>(rewriter.clone(*fillOp));
    use.set(newFillOp.getResult(0));
    return success();
  }
  return failure();
}

struct FusionPlanningForCpuPass
    : public impl::FusionPlanningForCpuPassBase<FusionPlanningForCpuPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* context = &getContext();

    // Cleanup passes to prepare ops for better clustering.
    {
      RewritePatternSet patterns(context);
      patterns.add(copyConstantLikeFillOp);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Move ops to gml_st.fusion clusters.
    {
      RewritePatternSet patterns(context);
      patterns.add(fusionPattern<linalg::MapOp>);
      patterns.add(fusionPattern<linalg::MatmulOp>);
      patterns.add(fusionPattern<linalg::ReduceOp>);
      patterns.add(fusionPattern<linalg::TransposeOp>);
      patterns.add(fusionPattern<thlo::ReverseOp>);
      patterns.add(fusionPattern<thlo::ScatterOp>);
      patterns.add(fusionPattern<thlo::SortOp>);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

struct InlineFusionClustersPass
    : public impl::InlineFusionClustersPassBase<InlineFusionClustersPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add(inlineFusionCluster);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFusionPlanningForCpuPass() {
  return std::make_unique<mlir::gml_st::FusionPlanningForCpuPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createInlineFusionClustersPass() {
  return std::make_unique<mlir::gml_st::InlineFusionClustersPass>();
}

}  // namespace mlir::gml_st
