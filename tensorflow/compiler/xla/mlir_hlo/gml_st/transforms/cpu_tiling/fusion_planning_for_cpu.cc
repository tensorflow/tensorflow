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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONPLANNINGFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

template <typename OpTy>
LogicalResult singleOpFusionClusterPattern(OpTy op, PatternRewriter& rewriter) {
  if (isa<gml_st::FusionOp>(op.getOperation()->getParentOp())) return failure();

  FusionCluster fusionCluster;
  fusionCluster.root = op;
  fusionCluster.operations.insert(op);

  if (failed(wrapFusionCluster(rewriter, fusionCluster))) return failure();

  return success();
}

LogicalResult matmulFusionPattern(linalg::MatmulOp op,
                                  PatternRewriter& rewriter) {
  if (isa<gml_st::FusionOp>(op.getOperation()->getParentOp())) return failure();

  auto fusionCluster = findMapFusionCluster(op);
  if (failed(wrapFusionCluster(rewriter, fusionCluster))) return failure();

  return success();
}

LogicalResult reduceFusionPattern(linalg::ReduceOp op,
                                  PatternRewriter& rewriter) {
  if (isa<gml_st::FusionOp>(op.getOperation()->getParentOp())) return failure();

  auto fusionCluster = findMapFusionCluster(op);
  if (failed(wrapFusionCluster(rewriter, fusionCluster))) return failure();

  return success();
}

LogicalResult mapFusionPattern(linalg::MapOp op, PatternRewriter& rewriter) {
  if (isa<gml_st::FusionOp>(op.getOperation()->getParentOp())) return failure();

  auto fusionCluster = findMapFusionCluster(op);
  if (failed(wrapFusionCluster(rewriter, fusionCluster))) return failure();

  return success();
}

struct FusionPlanningForCpuPass
    : public impl::FusionPlanningForCpuPassBase<FusionPlanningForCpuPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.add(singleOpFusionClusterPattern<thlo::ScatterOp>);
      patterns.add(singleOpFusionClusterPattern<thlo::SortOp>);
      patterns.add(singleOpFusionClusterPattern<thlo::ReverseOp>);
      patterns.add(singleOpFusionClusterPattern<linalg::TransposeOp>);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.add(matmulFusionPattern);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.add(reduceFusionPattern);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.add(mapFusionPattern);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFusionPlanningForCpu() {
  return std::make_unique<mlir::gml_st::FusionPlanningForCpuPass>();
}

}  // namespace mlir::gml_st
