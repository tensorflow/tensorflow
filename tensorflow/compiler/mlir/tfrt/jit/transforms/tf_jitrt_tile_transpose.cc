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
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_TILETRANSPOSE
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::SmallVector;
using mlir::dyn_cast;
using mlir::failure;
using mlir::MLIRContext;
using mlir::Operation;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Value;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgTilingOptions;

static constexpr llvm::StringRef kTileTransposeAppliedLabel =
    "__tile_transpose_applied_label__";

/// Returns true if the operation is a GenericOp implementing a transposition.
// TODO(diegocaballero): Move it to MLIR core?
bool IsTransposeGenericOp(Operation *op) {
  // Check that op is a generic op and has at least 2 dimensions.
  auto generic_op = dyn_cast<GenericOp>(op);
  if (!generic_op) return false;
  if (generic_op.getNumLoops() < 2) return false;

  // Check whether the body has only one operation (yield op). Transpose ops
  // fused with any other operations are not supported for now.
  mlir::Block *body = generic_op.getBody();
  if (body->empty() || body->begin() != std::prev(body->end())) return false;
  auto yield_op = dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yield_op || (yield_op.getNumOperands() != 1)) return false;

  // Check input and output.
  if ((generic_op.getNumDpsInputs() != 1) || (generic_op.getNumDpsInits() != 1))
    return false;

  // Check that input is yielded.
  if (generic_op.getMatchingBlockArgument(generic_op.getDpsInputOperand(0)) !=
      yield_op.getOperand(0))
    return false;

  // Check parallel iterators.
  auto iterator_types = generic_op.getIteratorTypesArray();
  if (std::any_of(iterator_types.begin(), iterator_types.end(),
                  [](auto iterator_type) {
                    return !mlir::linalg::isParallelIterator(iterator_type);
                  }))
    return false;

  // Check that the two indexing maps are a permutation.
  auto indexing_maps = generic_op.getIndexingMapsArray();
  if (indexing_maps.size() != 2) return false;
  return (indexing_maps[0].isIdentity() && indexing_maps[1].isPermutation()) ||
         (indexing_maps[0].isPermutation() && indexing_maps[1].isIdentity());
}

struct TileTransposePattern : public mlir::OpRewritePattern<GenericOp> {
  TileTransposePattern(LinalgTilingOptions options, MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<GenericOp>(context, benefit), options(options) {}

  mlir::LogicalResult matchAndRewrite(
      GenericOp linalg_op, PatternRewriter &rewriter) const override {
    if (mlir::gml_st::hasLabel(linalg_op, kTileTransposeAppliedLabel))
      return failure();
    if (!IsTransposeGenericOp(linalg_op)) return failure();

    auto tiled_linalg_op =
        mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_linalg_op) || tiled_linalg_op.value().loops.empty())
      return failure();

    auto tiled_loop =
        mlir::dyn_cast<LoopOp>(*tiled_linalg_op.value().loops.front());
    if (!tiled_loop) return failure();

    tiled_loop->walk([&](GenericOp tiledOp) {
      mlir::gml_st::setLabel(tiledOp, kTileTransposeAppliedLabel);
    });

    rewriter.replaceOp(linalg_op, tiled_loop->getResults());
    return success();
  }

 private:
  LinalgTilingOptions options;
};

struct TileTransposePass : public impl::TileTransposeBase<TileTransposePass> {
  void runOnOperation() override {
    auto get_tile_size = [&](mlir::OpBuilder b, Operation *op) {
      auto generic_op = llvm::cast<GenericOp>(op);
      unsigned num_loops = generic_op.getNumLoops();
      assert(num_loops >= 2 && "Expect two or more dimension in transpose op");

      // Compute the tile sizes for the 2-D vectorization of the transpose. We
      // pick eight as default vectorization factor for both dimensions since
      // it's the most performant AVX2 pattern for now. We pick the contiguous
      // dimension of the input as first vector dimension and the contiguous
      // dimension of the output as second vector dimension. This will maximize
      // contiguous vector loads/stores and minimize insert/extract/gather/
      // scatter operations.
      SmallVector<Value> tiles(num_loops,
                               b.create<ConstantIndexOp>(op->getLoc(), 1));
      auto indexing_maps = generic_op.getIndexingMapsArray();
      unsigned last_dim = num_loops - 1;
      unsigned vec_factor0 = 8, vec_factor1 = 8;
      unsigned vec_dim0 = indexing_maps[0].getDimPosition(last_dim);
      unsigned vec_dim1 = indexing_maps[1].getDimPosition(last_dim);

      // If the contiguous dimensions of both input and output are not
      // transposed (i.e, they are the same), we vectorize only that dimension.
      // That transpose case doesn't require intra-register transposition but
      // just copying a set of contiguous sub-buffers from the input to the
      // output tensor. Vectorizing a second dimension would increase too much
      // the memory pressure for no reason.
      if (vec_dim0 == vec_dim1) {
        tiles[vec_dim0] = b.create<ConstantIndexOp>(op->getLoc(), vec_factor0);
      } else {
        tiles[vec_dim0] = b.create<ConstantIndexOp>(op->getLoc(), vec_factor0);
        tiles[vec_dim1] = b.create<ConstantIndexOp>(op->getLoc(), vec_factor1);
      }

      return tiles;
    };

    auto func = getOperation();
    auto tiling_options =
        LinalgTilingOptions().setTileSizeComputationFunction(get_tile_size);

    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<TileTransposePattern>(tiling_options, patterns.getContext());
    if (failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    func.walk([](GenericOp op) {
      mlir::gml_st::removeLabel(op, kTileTransposeAppliedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileTransposePass() {
  return std::make_unique<TileTransposePass>();
}

}  // namespace tensorflow
