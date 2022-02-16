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
#include <iterator>
#include <memory>

#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::SmallVector;
using mlir::Attribute;
using mlir::Block;
using mlir::dyn_cast;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::success;
using mlir::Value;
using mlir::arith::ConstantIndexOp;
using mlir::linalg::CodegenStrategy;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgTilingOptions;

/// Returns true if the operation is a GenericOp implementing a transposition.
// TODO(diegocaballero): Move it to MLIR core?
bool IsTransposeGenericOp(Operation *op) {
  // Check that op is a generic op and has at least 2 dimensions.
  auto generic_op = mlir::dyn_cast<GenericOp>(op);
  if (!generic_op) return false;
  if (generic_op.getNumLoops() < 2) return false;

  // Check whether the body has only one operation (yield op). Transpose ops
  // fused with any other operations are not supported for now.
  Block *body = generic_op.getBody();
  if (body->empty() || body->begin() != std::prev(body->end())) return false;
  auto yield_op = dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yield_op || (yield_op.getNumOperands() != 1)) return false;

  // Check input and output.
  if ((generic_op.getNumInputs() != 1) || (generic_op.getNumOutputs() != 1))
    return false;

  // Check that input is yielded.
  if (generic_op.getTiedBlockArgument(generic_op.getInputOperand(0)) !=
      yield_op.getOperand(0))
    return false;

  // Check parallel iterators.
  auto iterator_types = generic_op.iterator_types();
  if (std::any_of(
          iterator_types.begin(), iterator_types.end(),
          [](Attribute attr) { return !mlir::isParallelIterator(attr); }))
    return false;

  // Check that the two indexing maps are a permutation.
  auto indexing_maps = generic_op.getIndexingMaps();
  if (indexing_maps.size() != 2) return false;
  return (indexing_maps[0].isIdentity() && indexing_maps[1].isPermutation()) ||
         (indexing_maps[0].isPermutation() && indexing_maps[1].isIdentity());
}

struct TileTransposePass : public TileTransposeBase<TileTransposePass> {
  void runOnOperation() override {
    auto get_tile_size = [&](OpBuilder b, Operation *op) {
      auto num_loops = llvm::cast<GenericOp>(op).getNumLoops();
      SmallVector<Value> tiles(num_loops,
                               b.create<ConstantIndexOp>(op->getLoc(), 1));
      if (tiles.size() >= 2) {
        tiles[tiles.size() - 1] = b.create<ConstantIndexOp>(op->getLoc(), 8);
        tiles[tiles.size() - 2] = b.create<ConstantIndexOp>(op->getLoc(), 8);
      }
      return tiles;
    };

    CodegenStrategy strategy;
    strategy.tile(
        GenericOp::getOperationName(),
        LinalgTilingOptions()
            .setTileSizeComputationFunction(get_tile_size)
            .setLoopType(mlir::linalg::LinalgTilingLoopType::TiledLoops),
        /*filter=*/[](Operation *op) {
          return success(IsTransposeGenericOp(op));
        });

    mlir::OpPassManager dynamic_pm("builtin.func");
    strategy.configurePassPipeline(dynamic_pm, &getContext());
    if (failed(runPipeline(dynamic_pm, getOperation())))
      return signalPassFailure();
  }
};

struct LowerTransposePass : public LowerTransposeBase<LowerTransposePass> {
  void runOnOperation() override {
    mlir::OpPassManager dynamic_pm("builtin.func");
    CodegenStrategy strategy;
    strategy.vectorLowering(
        mlir::linalg::LinalgVectorLoweringOptions()
            .enableVectorTransposeLowering()
            .enableAVX2Lowering()
            .setAVX2LoweringOptions(
                mlir::x86vector::avx2::LoweringOptions().setTransposeOptions(
                    mlir::x86vector::avx2::TransposeLoweringOptions()
                        .lower4x8xf32()
                        .lower8x8xf32())));

    strategy.configurePassPipeline(dynamic_pm, &getContext());
    if (failed(runPipeline(dynamic_pm, getOperation())))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateTileTransposePass() {
  return std::make_unique<TileTransposePass>();
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateLowerVectorTransposePass() {
  return std::make_unique<LowerTransposePass>();
}

}  // namespace tensorflow
