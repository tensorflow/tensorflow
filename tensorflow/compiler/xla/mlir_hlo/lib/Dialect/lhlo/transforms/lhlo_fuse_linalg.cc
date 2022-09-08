/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for fusing linalg ops obtained after LHLO
// lowering.

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/lhlo/transforms/passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {

#define GEN_PASS_DEF_LHLOFUSELINALGPASS
#include "mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.h.inc"

namespace {

using linalg::LinalgOp;

class LhloFuseLinalgPass
    : public impl::LhloFuseLinalgPassBase<LhloFuseLinalgPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

 public:
  LhloFuseLinalgPass() = default;
  LhloFuseLinalgPass(const LhloFuseLinalgPass&) = default;
  LhloFuseLinalgPass(bool useParallelLoops,
                     llvm::ArrayRef<unsigned> tileSizes) {
    tile_sizes_ = tileSizes;
    use_parallel_loops_.setValue(useParallelLoops);
  }

  void runOnOperation() override {
    auto func = getOperation();

    // TODO(pifon): Remove assumption that the function has a single block.
    if (!llvm::hasSingleElement(func)) {
      emitError(func.getLoc(), "The function needs to have a single block.");
      signalPassFailure();
      return;
    }

    // The fusion in Linalg is currently possible only when the consumer op is
    // tiled. In order to greedily fuse the ops, we have to start from the tiled
    // root linalg ops, i.e. linalg ops that write to output buffers of the
    // function or are returned in case of escaping allocations.
    llvm::SmallDenseSet<Value> resultBuffers;
    for (auto funcArg : func.getArguments()) {
      resultBuffers.insert(funcArg);
    }
    for (auto& block : func) {
      auto returnOp =
          mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
      if (!returnOp) continue;
      for (auto operand : returnOp.getOperands()) {
        resultBuffers.insert(operand);
      }
    }
    // Resolve aliasing operations (like casts) on the result to identify
    // results. This only handles escaping results.
    // TODO(herhut): Use BufferizeAliasAnalysis for this.
    llvm::SmallVector<Value, 4> worklist(resultBuffers.begin(),
                                         resultBuffers.end());
    while (!worklist.empty()) {
      Value result = worklist.pop_back_val();
      auto* definingOp = result.getDefiningOp();
      if (!definingOp) {
        continue;
      }

      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(definingOp)) {
        auto alias = viewLike.getViewSource();
        if (resultBuffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(definingOp)) {
        auto alias = toTensor.getMemref();
        if (resultBuffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto toMemref = dyn_cast<bufferization::ToMemrefOp>(definingOp)) {
        auto alias = toMemref.getTensor();
        if (resultBuffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto tensorCast = dyn_cast<tensor::CastOp>(definingOp)) {
        auto alias = tensorCast.getSource();
        if (resultBuffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto regionInterface =
              dyn_cast<RegionBranchOpInterface>(definingOp)) {
        for (Region& region : regionInterface.getOperation()->getRegions()) {
          // Only consider regions that can return to the parent region.
          SmallVector<RegionSuccessor, 2> successorRegions;
          regionInterface.getSuccessorRegions(region.getRegionNumber(),
                                              successorRegions);
          if (llvm::none_of(successorRegions, [&](auto successorRegion) {
                return successorRegion.isParent();
              }))
            continue;

          // Iterate over all immediate terminators and record the values
          // corresponding to result_buffers of interest.
          for (Block& block : region) {
            if (block.empty()) continue;
            Operation& operation = block.back();
            if (!operation.hasTrait<OpTrait::ReturnLike>()) continue;
            auto idx = result.dyn_cast<OpResult>().getResultNumber();
            if (resultBuffers.insert(operation.getOperand(idx)).second) {
              worklist.push_back(operation.getOperand(idx));
            }
          }
        }
      }
    }

    MLIRContext* ctx = func.getContext();
    OpBuilder b(func);
    func.walk([&](linalg::GenericOp genericOp) {
      SmallVector<int64_t, 2> tileSizes(tile_sizes_.begin(), tile_sizes_.end());
      if (tileSizes.empty()) {
        tileSizes = SmallVector<int64_t, 2>(genericOp.getNumLoops(), 1);
      }
      auto op = cast<LinalgOp>(genericOp.getOperation());
      for (OpOperand* opOperand : op.getOutputBufferOperands()) {
        if (!resultBuffers.count(opOperand->get())) continue;
        if (tileGenericOp(op, tileSizes, &b)) {
          genericOp.erase();
          return;
        }
      }
    });
    auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();

    // Fuse producers of tiled linalg ops.
    llvm::SmallDenseSet<Operation*> eraseSet;
    SmallVector<LinalgOp, 8> linalgOps;
    func.walk([&](LinalgOp op) { linalgOps.push_back(op); });
    for (LinalgOp op : llvm::reverse(linalgOps)) {
      for (OpOperand* inputOperand : op.getInputOperands()) {
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        auto info = fuseProducerOfBuffer(b, *inputOperand, graph);
        if (failed(info)) continue;
        auto* originalOp = info->originalProducer.getOperation();
        eraseSet.insert(originalOp);
        auto* originalOpInLinalgOpsVector =
            std::find_if(linalgOps.begin(), linalgOps.end(),
                         [&](const Operation* op) { return op == originalOp; });
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
      }

      auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
        return signalPassFailure();
    }
    for (auto* e : eraseSet) e->erase();
  }

 private:
  bool tileGenericOp(LinalgOp op, ArrayRef<int64_t> tileSizes, OpBuilder* b) {
    auto loopType = use_parallel_loops_
                        ? linalg::LinalgTilingLoopType::ParallelLoops
                        : linalg::LinalgTilingLoopType::Loops;
    IRRewriter rewriter(*b);
    return succeeded(linalg::tileLinalgOp(
        rewriter, op,
        linalg::LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(
            loopType)));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLhloFuseLinalgPass(
    bool useParallelLoops, ArrayRef<unsigned> tileSizes) {
  return std::make_unique<LhloFuseLinalgPass>(useParallelLoops, tileSizes);
}

}  // namespace lmhlo
}  // namespace mlir
