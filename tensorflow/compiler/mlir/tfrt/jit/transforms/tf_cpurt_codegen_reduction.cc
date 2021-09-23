/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using mlir::failure;
using mlir::Identifier;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::Operation;
using mlir::Optional;
using mlir::PatternRewriter;
using mlir::success;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;
using mlir::linalg::PaddingValueComputationFunction;
using mlir::linalg::TiledLoopOp;
using mlir::tensor::ExtractSliceOp;
using mlir::tensor::InsertSliceOp;

// Tiles a GenericOp that models a reduction and then fuses its inputs and
// outputs. Currently, only the FillOp that initializes the output is fused into
// the TiledLoopOp.
struct TileAndFusePattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  TileAndFusePattern(const LinalgTilingOptions &options,
                     const LinalgTransformationFilter &filter,
                     mlir::MLIRContext *context,
                     mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    auto tiled_op = tileLinalgOp(rewriter, linalg_op, options);
    if (!tiled_op) return failure();

    auto tiled_loop_op = mlir::dyn_cast<TiledLoopOp>(tiled_op->loops.front());
    if (!tiled_loop_op) return failure();

    if (failed(FuseFillOp(rewriter, tiled_loop_op, tiled_op->op))) {
      return failure();
    }
    rewriter.replaceOp(linalg_op, tiled_loop_op->getResults());

    tiled_loop_op->walk([&](LinalgOp tiledOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    });
    return success();
  }

 private:
  // Replaces
  //
  // %0 = linalg.fill(%cst, %out)
  //
  // with
  //
  // %0 = linalg.fill(%cst, %out)
  // %1 = linalg.fill(%cst, %0)
  //
  // The idea is to still initialize the output of the reduction even if the
  // FillOp is fused into the TiledLoopOp. In that case %1 will be fused into
  // the loop body and %0 will remain outside of the loop.
  std::pair<FillOp, FillOp> ChainFillOp(PatternRewriter &rewriter,
                                        FillOp fill_op) const {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(fill_op);

    auto first = rewriter.clone(*fill_op);
    auto second = rewriter.replaceOpWithNewOp<FillOp>(fill_op, fill_op.value(),
                                                      first->getResult(0));
    return std::make_pair(mlir::cast<FillOp>(first), second);
  }

  // Fuses FillOp producer of the output argument of the TiledLoopOp and inserts
  // an operation that accumulates the partial result, i.e. reduced tile, and
  // the current value of the output tile.
  LogicalResult FuseFillOp(PatternRewriter &rewriter, TiledLoopOp tiled_loop,
                           LinalgOp tiled_op) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(tiled_op);

    auto fill_op = tiled_loop.outputs().front().getDefiningOp<FillOp>();
    if (!fill_op) return failure();

    auto fill_op_chain = ChainFillOp(rewriter, fill_op);

    Optional<mlir::linalg::FusionInfo> fusion_info =
        mlir::linalg::fuseProducerOfTensor(
            rewriter, fill_op_chain.second->getResult(0),
            *tiled_op.getOutputOperands().front());
    if (!fusion_info.hasValue()) return failure();

    auto fused_fill_op = mlir::cast<FillOp>(fusion_info->fusedProducer);

    mlir::Value partial_result = tiled_op->getResult(0);

    // Find insert_slice that inserts the result back to the output.
    auto insert =
        mlir::dyn_cast<InsertSliceOp>(*partial_result.getUsers().begin());
    if (!insert) return failure();

    // Create operation that accumulates the partial result into the output.
    auto num_parallel_loops = tiled_op.getNumParallelLoops();
    mlir::SmallVector<mlir::StringRef, 3> parallel_iter_types(
        num_parallel_loops, mlir::getParallelIteratorTypeName());
    auto id_map = rewriter.getMultiDimIdentityMap(num_parallel_loops);

    auto loc = tiled_op.getLoc();
    auto accumulator = rewriter.create<GenericOp>(
        loc, partial_result.getType(), llvm::makeArrayRef(partial_result),
        llvm::makeArrayRef(fused_fill_op.output()),
        llvm::makeArrayRef({id_map, id_map}), parallel_iter_types);

    auto reduce_tile = mlir::cast<GenericOp>(tiled_op);
    mlir::BlockAndValueMapping bvm;
    rewriter.cloneRegionBefore(reduce_tile.region(), accumulator.region(),
                               accumulator.region().end(), bvm);
    rewriter.updateRootInPlace(insert, [&]() {
      insert.sourceMutable().assign(accumulator.getResult(0));
    });
    return success();
  }

  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Rewrite linalg.fill(extract_slice) as linalg.fill(init_tensor). This rewrite
// is required for correctness, because otherwise after bufferization the fused
// output linalg.fill would still use the buffer for the reduction of the whole
// output instead of allocating a local buffer only for the reduced tile.
//
// A better way to perform this transformation is to have it in MLIR Core as a
// part of the fusion logic. To support this correctly, we would also modify
// logic for padding, so that we could pad fill(init_tensor). Currently, only
// fill(extract_slice) can be padded. All these changes will happen once we
// converge on the pipeline design.
struct FillOfExtractSlice : public mlir::OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp fill,
                                PatternRewriter &rewriter) const override {
    if (!fill.hasTensorSemantics()) return failure();

    auto fill_tensor_type = fill.getOutputTensorTypes().back();
    if (!fill_tensor_type.hasStaticShape()) return failure();

    if (auto extract = fill.output().getDefiningOp<ExtractSliceOp>()) {
      llvm::SmallVector<int64_t, 4> static_sizes = llvm::to_vector<4>(
          llvm::map_range(extract.static_sizes().cast<mlir::ArrayAttr>(),
                          [](mlir::Attribute a) -> int64_t {
                            return a.cast<mlir::IntegerAttr>().getInt();
                          }));
      auto init = rewriter.create<mlir::linalg::InitTensorOp>(
          fill.getLoc(), extract.getDynamicSizes(), static_sizes,
          fill_tensor_type.getElementType());
      rewriter.replaceOpWithNewOp<FillOp>(fill, fill.value(), init);
      return success();
    }
    return failure();
  }
};

// Match 2D row reduction. This is a starting point, we will relax this
// condition further down the road, when we add support for more reduction
// types.
bool is2DRowReduction(mlir::Operation *op) {
  auto reduction = mlir::dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumOutputs() != 1 || reduction.getNumLoops() != 2)
    return false;
  auto iter_types = reduction.iterator_types();
  return mlir::isParallelIterator(iter_types[0]) &&
         mlir::isReductionIterator(iter_types[1]);
}

struct CodegenReductionPass
    : public CodegenReductionBase<CodegenReductionPass> {
  void runOnFunction() override {
    mlir::linalg::LinalgTilingOptions tiling_options;
    tiling_options.setTileSizes({4, 4});
    tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::TiledLoops);

    auto func = getFunction();
    auto context = func.getContext();

    auto patterns =
        mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
    auto filter = LinalgTransformationFilter(
                      llvm::None, {Identifier::get("tiled", context)})
                      .addFilter([](Operation *op) {
                        return success(is2DRowReduction(op));
                      });
    patterns.insert<FillOfExtractSlice>(context);
    patterns.insert<TileAndFusePattern>(tiling_options, filter,
                                        patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](mlir::linalg::LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass() {
  return std::make_unique<CodegenReductionPass>();
}

}  // namespace tensorflow
