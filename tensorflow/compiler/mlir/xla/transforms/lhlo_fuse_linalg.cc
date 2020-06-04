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

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace xla_lhlo {
namespace {

using linalg::LinalgOp;

class LhloFuseLinalg : public PassWrapper<LhloFuseLinalg, FunctionPass> {
 public:
  LhloFuseLinalg() = default;
  LhloFuseLinalg(const LhloFuseLinalg&) {}
  LhloFuseLinalg(bool use_parallel_loops, llvm::ArrayRef<unsigned> tile_sizes) {
    tile_sizes_ = tile_sizes;
    use_parallel_loops_.setValue(use_parallel_loops);
  }

  void runOnFunction() override {
    auto func = getFunction();

    // TODO(pifon): Remove assumption that the function has a single block.
    if (func.getBlocks().size() != 1) {
      emitError(func.getLoc(), "The function needs to have a single block.");
      signalPassFailure();
      return;
    }

    // The fusion in Linalg is currently possible only when the consumer op is
    // tiled. In order to greedily fuse the ops, we have to start from the tiled
    // root linalg ops, i.e. linalg ops that write to output buffers of the
    // function.
    llvm::SmallDenseSet<Value> func_args;
    for (auto func_arg : func.getArguments()) {
      func_args.insert(func_arg);
    }
    MLIRContext* ctx = func.getContext();
    OpBuilder b(func);
    OperationFolder folder(ctx);
    func.walk([&](linalg::GenericOp generic_op) {
      SmallVector<int64_t, 2> tile_sizes(tile_sizes_.begin(),
                                         tile_sizes_.end());
      if (tile_sizes.empty()) {
        tile_sizes = SmallVector<int64_t, 2>(generic_op.getNumLoops(), 1);
      }
      auto op = cast<LinalgOp>(generic_op.getOperation());
      for (const Value result : op.getOutputBuffers()) {
        if (!func_args.count(result)) continue;
        if (tileGenericOp(op, tile_sizes, &b)) {
          generic_op.erase();
          return;
        }
      }
    });
    auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    applyPatternsAndFoldGreedily(func, patterns);

    // Fuse producers of tiled linalg ops.
    llvm::SmallDenseSet<Operation*> erase_set;
    SmallVector<Operation*, 8> linalg_ops;
    func.walk([&](LinalgOp op) { linalg_ops.push_back(op); });
    for (auto* op : llvm::reverse(linalg_ops)) {
      for (unsigned id = 0, e = LinalgOp(op).getNumInputs(); id < e; ++id) {
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalg_ops);
        if (auto info = fuseProducerOf(b, op, id, graph, &folder)) {
          auto originalOp = info->originalProducer.getOperation();
          erase_set.insert(originalOp);
          auto originalOpInLinalgOpsVector = std::find_if(
              linalg_ops.begin(), linalg_ops.end(),
              [&](const Operation* op) { return op == originalOp; });
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
        }
      }

      auto patterns = linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      applyPatternsAndFoldGreedily(func, patterns);
    }
    for (auto* e : erase_set) e->erase();
  }

 private:
  bool tileGenericOp(LinalgOp op, ArrayRef<int64_t> tile_sizes, OpBuilder* b) {
    auto loopType = use_parallel_loops_
                        ? linalg::LinalgTilingLoopType::ParallelLoops
                        : linalg::LinalgTilingLoopType::Loops;
    auto tiled_generic_op = linalg::tileLinalgOp(*b, op,
                                                 linalg::LinalgTilingOptions()
                                                     .setTileSizes(tile_sizes)
                                                     .setLoopType(loopType));
    return tiled_generic_op.hasValue();
  }

  Option<bool> use_parallel_loops_{
      *this, "use-parallel-loops",
      llvm::cl::desc(
          "Tiles GenericOp consumer to parallel loops before linalg fusion"),
      llvm::cl::init(false)};

  ListOption<unsigned> tile_sizes_{
      *this, "tile-sizes",
      llvm::cl::desc(
          "Tile sizes by which to tile linalg generic before linalg fusion"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLhloFuseLinalg(
    bool use_parallel_loops, ArrayRef<unsigned> tile_sizes) {
  return absl::make_unique<LhloFuseLinalg>(use_parallel_loops, tile_sizes);
}

static PassRegistration<LhloFuseLinalg> legalize_pass(
    "lhlo-fuse-linalg",
    "Greedily fuse linalg ops obtained after LHLO lowering.");

}  // namespace xla_lhlo
}  // namespace mlir
