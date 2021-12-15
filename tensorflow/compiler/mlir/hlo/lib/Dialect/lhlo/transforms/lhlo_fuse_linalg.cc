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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/lhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/lhlo/transforms/passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {
namespace {

using linalg::LinalgOp;

class LhloFuseLinalgPass : public LhloFuseLinalgPassBase<LhloFuseLinalgPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

 public:
  LhloFuseLinalgPass() = default;
  LhloFuseLinalgPass(const LhloFuseLinalgPass&) {}
  LhloFuseLinalgPass(bool use_parallel_loops,
                     llvm::ArrayRef<unsigned> tile_sizes) {
    tile_sizes_ = tile_sizes;
    use_parallel_loops_.setValue(use_parallel_loops);
  }

  void runOnFunction() override {
    auto func = getFunction();

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
    llvm::SmallDenseSet<Value> result_buffers;
    for (auto func_arg : func.getArguments()) {
      result_buffers.insert(func_arg);
    }
    for (auto& block : func) {
      auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(block.getTerminator());
      if (!returnOp) continue;
      for (auto operand : returnOp.getOperands()) {
        result_buffers.insert(operand);
      }
    }
    // Resolve aliasing operations (like casts) on the result to identify
    // results. This only handles escaping results.
    // TODO(herhut): Use BufferizeAliasAnalysis for this.
    llvm::SmallVector<Value, 4> worklist(result_buffers.begin(),
                                         result_buffers.end());
    while (!worklist.empty()) {
      Value result = worklist.pop_back_val();
      auto definingOp = result.getDefiningOp();
      if (!definingOp) {
        continue;
      }

      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(definingOp)) {
        auto alias = viewLike.getViewSource();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto to_tensor = dyn_cast<bufferization::ToTensorOp>(definingOp)) {
        auto alias = to_tensor.memref();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto to_memref = dyn_cast<bufferization::ToMemrefOp>(definingOp)) {
        auto alias = to_memref.tensor();
        if (result_buffers.insert(alias).second) {
          worklist.push_back(alias);
        }
        continue;
      }

      if (auto tensor_cast = dyn_cast<tensor::CastOp>(definingOp)) {
        auto alias = tensor_cast.source();
        if (result_buffers.insert(alias).second) {
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
            if (result_buffers.insert(operation.getOperand(idx)).second) {
              worklist.push_back(operation.getOperand(idx));
            }
          }
        }
      }
    }

    MLIRContext* ctx = func.getContext();
    OpBuilder b(func);
    func.walk([&](linalg::GenericOp generic_op) {
      SmallVector<int64_t, 2> tile_sizes(tile_sizes_.begin(),
                                         tile_sizes_.end());
      if (tile_sizes.empty()) {
        tile_sizes = SmallVector<int64_t, 2>(generic_op.getNumLoops(), 1);
      }
      auto op = cast<LinalgOp>(generic_op.getOperation());
      for (OpOperand* op_operand : op.getOutputBufferOperands()) {
        if (!result_buffers.count(op_operand->get())) continue;
        if (tileGenericOp(op, tile_sizes, &b)) {
          generic_op.erase();
          return;
        }
      }
    });
    auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Fuse producers of tiled linalg ops.
    llvm::SmallDenseSet<Operation*> erase_set;
    SmallVector<LinalgOp, 8> linalg_ops;
    func.walk([&](LinalgOp op) { linalg_ops.push_back(op); });
    for (LinalgOp op : llvm::reverse(linalg_ops)) {
      for (OpOperand* inputOperand : op.getInputOperands()) {
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalg_ops);
        auto info = fuseProducerOfBuffer(b, *inputOperand, graph);
        if (failed(info)) continue;
        auto originalOp = info->originalProducer.getOperation();
        erase_set.insert(originalOp);
        auto originalOpInLinalgOpsVector =
            std::find_if(linalg_ops.begin(), linalg_ops.end(),
                         [&](const Operation* op) { return op == originalOp; });
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
      }

      auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
    for (auto* e : erase_set) e->erase();
  }

 private:
  bool tileGenericOp(LinalgOp op, ArrayRef<int64_t> tile_sizes, OpBuilder* b) {
    auto loopType = use_parallel_loops_
                        ? linalg::LinalgTilingLoopType::ParallelLoops
                        : linalg::LinalgTilingLoopType::Loops;
    return succeeded(linalg::tileLinalgOp(*b, op,
                                          linalg::LinalgTilingOptions()
                                              .setTileSizes(tile_sizes)
                                              .setLoopType(loopType)));
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createLhloFuseLinalgPass(
    bool use_parallel_loops, ArrayRef<unsigned> tile_sizes) {
  return std::make_unique<LhloFuseLinalgPass>(use_parallel_loops, tile_sizes);
}

}  // namespace lmhlo
}  // namespace mlir
