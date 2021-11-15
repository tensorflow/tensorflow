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

#include <utility>

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using mlir::BlockAndValueMapping;
using mlir::BlockArgument;
using mlir::cast;
using mlir::dyn_cast;
using mlir::failure;
using mlir::Identifier;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::Operation;
using mlir::OpInterfaceRewritePattern;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Value;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::InitTensorOp;
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
struct TileReductionAndFuseOutput : public OpInterfaceRewritePattern<LinalgOp> {
  TileReductionAndFuseOutput(const LinalgTilingOptions &options,
                             const LinalgTransformationFilter &filter,
                             MLIRContext *context,
                             mlir::PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    if (linalg_op.getNumOutputs() != 1) return failure();

    auto tiled_op = tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_op)) return failure();

    auto tiled_loop_op = dyn_cast<TiledLoopOp>(tiled_op->loops.front());
    if (!tiled_loop_op) return failure();

    rewriter.replaceOp(linalg_op, tiled_loop_op.getResult(0));

    return RewriteTiledReduction(rewriter, tiled_loop_op, tiled_op->op);
  }

 private:
  // Add a new output argument to the `tiled_loop`. It will be produced by
  // `init_tensor` op with the same shape as the first output argument.
  //
  // Rewrite
  //
  //   %init = linalg.init_tensor
  //   %fill = linalg.fill(%cst, %init)
  //   linalg.tiled_loop outs(%fill)
  //
  // into
  //
  //   %init = linalg.init_tensor
  //** %init_clone = linalg.init_tensor
  //   %fill = linalg.fill(%cst, %init)
  //** linalg.tiled_loop outs(%fill, %init_clone)
  BlockArgument CloneAndAppendInitTensorToTiledLoop(
      PatternRewriter &rewriter, FillOp fill, TiledLoopOp tiled_loop) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(fill);

    auto init = fill.output().getDefiningOp<InitTensorOp>();
    auto init_clone = cast<InitTensorOp>(rewriter.clone(*init));
    mlir::OpOperand *init_clone_output_operand;
    rewriter.updateRootInPlace(tiled_loop, [&]() {
      init_clone_output_operand =
          &tiled_loop.appendOutputOperand(rewriter, init_clone.result());
    });
    return tiled_loop.getTiedBlockArgument(*init_clone_output_operand);
  }

  // Fuse `fill` operation into the `tiled_loop`, rewire the `linalg.generic` to
  // use it as the output for the reduced tile. Also create an additional
  // `insert_slice` that updates the new output.
  //
  // Rewrite
  //
  // %init = linalg.init_tensor
  // %init_clone = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_clone) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //   %reduce = linalg.generic outs (%extract_output_slice)
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_clone = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_clone) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //** %slice_of_cloned_output = tensor.extract_slice %init
  //** %reduce = linalg.generic outs (%slice_of_cloned_input)
  //** %update_cloned_output = tensor.insert_slice %reduce into %init_clone
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_cloned_output
  // }
  void FuseFill(PatternRewriter &rewriter, LinalgOp tiled_op, FillOp fill,
                BlockArgument loop_output_bb_arg,
                BlockArgument cloned_output_bb_arg,
                ExtractSliceOp extract_output_slice,
                InsertSliceOp insert_output_slice) const {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tiled_op);

    BlockAndValueMapping bvm;
    bvm.map(loop_output_bb_arg, cloned_output_bb_arg);
    Value slice_of_cloned_output =
        cast<ExtractSliceOp>(rewriter.clone(*extract_output_slice, bvm));

    auto fused_fill = rewriter.create<FillOp>(tiled_op.getLoc(), fill.value(),
                                              slice_of_cloned_output);
    rewriter.updateRootInPlace(tiled_op, [&]() {
      tiled_op.getOutputOperand(0)->set(fused_fill.result());
    });

    bvm.map(insert_output_slice.dest(), cloned_output_bb_arg);
    rewriter.setInsertionPointAfter(tiled_op);
    Value cloned_insert =
        cast<InsertSliceOp>(rewriter.clone(*insert_output_slice, bvm));
    auto yield = tiled_op.getOperation()->getBlock()->getTerminator();
    rewriter.updateRootInPlace(
        yield, [&]() { yield->insertOperands(1, cloned_insert); });
  }

  // Add an operation that combines the partial result with the output.
  //
  // Rewrite
  //
  // %init = linalg.init_tensor
  // %init_clone = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_clone) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_cloned_output = tensor.extract_slice %init
  //   %reduce = linalg.generic outs (%slice_of_cloned_input)
  //   %update_cloned_output = tensor.insert_slice %reduce into %init_clone
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_cloned_output
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_clone = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_clone) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_cloned_output = tensor.extract_slice %init
  //   %reduce = linalg.generic outs (%slice_of_cloned_input)
  //   %update_cloned_output = tensor.insert_slice %reduce into %init_clone
  //
  //** %combine = linalg.generic ins (%reduce) outs (%extract_output_slice)
  //** %insert_output_slice = tensor.insert_slice %combine into %fill
  //
  //   linalg.yield %insert_output_slice, %update_cloned_output
  // }
  void CombineReducedTileWithOutput(PatternRewriter &rewriter,
                                    LinalgOp tiled_op, Value partial_result,
                                    ExtractSliceOp extract_output_slice,
                                    InsertSliceOp insert_output_slice) const {
    rewriter.setInsertionPointAfter(tiled_op);
    auto num_parallel_loops = tiled_op.getNumParallelLoops();
    mlir::SmallVector<mlir::StringRef, 3> parallel_iter_types(
        num_parallel_loops, mlir::getParallelIteratorTypeName());
    auto id_map = rewriter.getMultiDimIdentityMap(num_parallel_loops);

    auto accumulator = rewriter.create<GenericOp>(
        tiled_op.getLoc(), partial_result.getType(),
        llvm::makeArrayRef(partial_result),
        llvm::makeArrayRef(extract_output_slice.result()),
        llvm::makeArrayRef({id_map, id_map}), parallel_iter_types);

    auto reduce_tile = mlir::cast<GenericOp>(tiled_op);
    BlockAndValueMapping bvm;
    rewriter.cloneRegionBefore(reduce_tile.region(), accumulator.region(),
                               accumulator.region().end(), bvm);
    rewriter.updateRootInPlace(insert_output_slice, [&]() {
      insert_output_slice.sourceMutable().assign(accumulator.getResult(0));
    });
  }

  // Unfortunaly, there is no way to modify the results of the loop inplace. So
  // we have to replace it with a clone.
  TiledLoopOp CreateLoopWithUpdatedResults(PatternRewriter &rewriter,
                                           TiledLoopOp tiled_loop) const {
    auto loc = tiled_loop.getLoc();
    rewriter.setInsertionPoint(tiled_loop);
    auto new_loop = rewriter.create<TiledLoopOp>(
        loc, mlir::TypeRange(tiled_loop.outputs()), tiled_loop.getOperands(),
        tiled_loop->getAttrs());
    rewriter.inlineRegionBefore(tiled_loop.region(), new_loop.region(),
                                new_loop.region().begin());

    rewriter.replaceOp(tiled_loop, new_loop.getResult(0));
    return new_loop;
  }

  // Fuses FillOp producer of the output argument of the TiledLoopOp and inserts
  // an operation that accumulates the partial result, i.e. reduced tile, and
  // the current value of the output tile.
  LogicalResult RewriteTiledReduction(PatternRewriter &rewriter,
                                      TiledLoopOp tiled_loop,
                                      LinalgOp tiled_op) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(tiled_op);

    // Find tiled loop output operand and the corresponding block argument.
    mlir::OpOperand *loop_output_operand =
        tiled_loop.findOutputOperand(tiled_loop.outputs().front());
    BlockArgument loop_output_bb_arg =
        tiled_loop.getTiedBlockArgument(*loop_output_operand);

    // Find `linalg.fill` producer of the output.
    auto fill = loop_output_operand->get().getDefiningOp<FillOp>();
    if (!fill) return failure();

    // Find extract_slice/insert_slice pair used to RMW output.
    auto extract_output_slice =
        tiled_op.getOutputOperand(0)->get().getDefiningOp<ExtractSliceOp>();
    if (!extract_output_slice) return failure();

    Value tiled_op_result = tiled_op->getResult(0);
    auto insert_output_slice =
        dyn_cast<InsertSliceOp>(*tiled_op_result.getUsers().begin());
    if (!insert_output_slice) return failure();

    // Fuse the output.
    BlockArgument cloned_output_bb_arg =
        CloneAndAppendInitTensorToTiledLoop(rewriter, fill, tiled_loop);
    FuseFill(rewriter, tiled_op, fill, loop_output_bb_arg, cloned_output_bb_arg,
             extract_output_slice, insert_output_slice);
    CombineReducedTileWithOutput(rewriter, tiled_op, tiled_op_result,
                                 extract_output_slice, insert_output_slice);

    // Update the results.
    TiledLoopOp updated_loop =
        CreateLoopWithUpdatedResults(rewriter, tiled_loop);
    updated_loop->walk([&](LinalgOp tOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tOp);
    });
    return success();
  }

  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Match 2D row reduction. This is a starting point, we will relax this
// condition further down the road, when we add support for more reduction
// types.
bool is2DRowOrColumnReduction(Operation *op) {
  auto reduction = dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumOutputs() != 1 || reduction.getNumLoops() != 2)
    return false;
  return reduction.getNumReductionLoops() == 1;
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
    mlir::memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    auto filter = LinalgTransformationFilter(
                      llvm::None, {Identifier::get("tiled", context)})
                      .addFilter([](Operation *op) {
                        return success(is2DRowOrColumnReduction(op));
                      });
    patterns.insert<TileReductionAndFuseOutput>(tiling_options, filter,
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
