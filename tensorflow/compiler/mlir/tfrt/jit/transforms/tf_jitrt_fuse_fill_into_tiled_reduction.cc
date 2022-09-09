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

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_FUSEFILLINTOTILEDREDUCTION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::makeArrayRef;
using mlir::BlockAndValueMapping;
using mlir::BlockArgument;
using mlir::dyn_cast;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpFoldResult;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::gml_st::LoopOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::InitTensorOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::YieldOp;
using mlir::tensor::ExtractSliceOp;
using mlir::tensor::InsertSliceOp;

SmallVector<OpFoldResult> GetParallelDimStep(LoopOp tiled_loop) {
  assert(tiled_loop.getNumLoops() == 2 && "Expected a 2D loop");
  Value step = tiled_loop.isParallelDimension(0) ? tiled_loop.step().front()
                                                 : tiled_loop.step().back();
  if (auto constant = step.getDefiningOp<mlir::arith::ConstantOp>()) {
    return {constant.getValue()};
  }
  return {step};
}

// Fuses `linalg.fill` into a loop with a tiled reduction.
// Currently, only 2D case is supported. Fusion into a tiled 1D reduction is
// also possible.
struct FuseFillIntoTiledReductionPattern : public OpRewritePattern<GenericOp> {
  explicit FuseFillIntoTiledReductionPattern(MLIRContext *context,
                                             mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (linalg_op.getNumOutputs() != 1) return failure();
    if (linalg_op.getNumLoops() != 2) return failure();

    // Get immediate parent.
    auto tiled_loop_op =
        dyn_cast<LoopOp>(linalg_op->getParentRegion()->getParentOp());
    if (!tiled_loop_op) return failure();
    if (tiled_loop_op.getNumLoops() != 2) return failure();

    return RewriteTiledReduction(rewriter, tiled_loop_op, linalg_op);
  }

 private:
  // Add a new output argument to the `tiled_loop`. It will be produced by
  // `init_tensor` op with the same shape of the tiled output argument.
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
  //** %init_tile = linalg.init_tensor [%stride]
  //   %fill = linalg.fill(%cst, %init)
  //** linalg.tiled_loop outs(%fill, %init_tile)
  BlockArgument CloneAndAppendInitTensorToTiledLoop(PatternRewriter &rewriter,
                                                    FillOp fill,
                                                    LoopOp tiled_loop) const {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(fill);

    auto init = fill.output().getDefiningOp<InitTensorOp>();

    Value init_clone = rewriter.create<InitTensorOp>(
        init.getLoc(), GetParallelDimStep(tiled_loop),
        init.getType().cast<mlir::RankedTensorType>().getElementType());
    mlir::OpOperand *init_clone_output_operand;
    rewriter.updateRootInPlace(tiled_loop, [&]() {
      init_clone_output_operand =
          &tiled_loop.appendOutputOperand(rewriter, init_clone);
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
  // %init_tile = linalg.init_tensor [%stride]
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //   %reduce = linalg.generic outs (%extract_output_slice)
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //** %slice_of_output_tile = tensor.extract_slice %init
  //** %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //** %reduce = linalg.generic outs (%fill_of_output_tile)
  //** %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  void FuseFill(PatternRewriter &rewriter, LinalgOp tiled_op, FillOp fill,
                BlockArgument loop_output_bb_arg,
                BlockArgument output_tile_bb_arg,
                ExtractSliceOp extract_output_slice,
                InsertSliceOp insert_output_slice) const {
    Location loc = tiled_op.getLoc();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tiled_op);

    SmallVector<OpFoldResult> offset{rewriter.getIndexAttr(0)};
    Value slice_of_output_tile = rewriter.create<ExtractSliceOp>(
        loc, output_tile_bb_arg, offset, extract_output_slice.getMixedSizes(),
        extract_output_slice.getMixedStrides());

    auto fused_fill =
        rewriter.create<FillOp>(loc, fill.value(), slice_of_output_tile);
    rewriter.updateRootInPlace(tiled_op, [&]() {
      tiled_op.getOutputOperand(0)->set(fused_fill.result());
    });

    rewriter.setInsertionPointAfter(tiled_op);
    Value cloned_insert = rewriter.create<mlir::tensor::InsertSliceOp>(
        loc, fused_fill.getResult(0), output_tile_bb_arg, offset,
        extract_output_slice.getMixedSizes(),
        extract_output_slice.getMixedStrides());

    auto yield = tiled_op.getOperation()->getBlock()->getTerminator();
    rewriter.updateRootInPlace(
        yield, [&]() { yield->insertOperands(1, cloned_insert); });
  }

  // Add an operation that combines the partial result with the output.
  //
  // Rewrite
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_output_tile = tensor.extract_slice %init
  //   %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //   %reduce = linalg.generic outs (%fill_of_output_tile)
  //   %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //   %insert_output_slice = tensor.insert_slice %reduce into %fill
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  //
  // into
  //
  // %init = linalg.init_tensor
  // %init_tile = linalg.init_tensor
  // %fill = linalg.fill(%cst, %init)
  // linalg.tiled_loop outs(%fill, %init_tile) {
  //   %extract_output_slice = tensor.extract_slice %fill
  //
  //   %slice_of_output_tile = tensor.extract_slice %init
  //   %fill_of_output_tile = linalg.fill(%cst, %slice_of_output_tile)
  //   %reduce = linalg.generic outs (%fill_of_output_tile)
  //   %update_output_tile = tensor.insert_slice %reduce into %init_tile
  //
  //** %combine = linalg.generic ins (%reduce) outs (%extract_output_slice)
  //** %insert_output_slice = tensor.insert_slice %combine into %fill
  //
  //   linalg.yield %insert_output_slice, %update_output_tile
  // }
  LogicalResult CombineReducedTileWithOutput(
      PatternRewriter &rewriter, LinalgOp tiled_op, Value partial_result,
      ExtractSliceOp extract_output_slice,
      InsertSliceOp insert_output_slice) const {
    rewriter.setInsertionPointAfter(tiled_op);
    auto num_parallel_loops = tiled_op.getNumParallelLoops();
    SmallVector<mlir::StringRef, 3> parallel_iter_types(
        num_parallel_loops, mlir::getParallelIteratorTypeName());
    auto id_map = rewriter.getMultiDimIdentityMap(num_parallel_loops);

    auto combiner_or = DetectCombiner(tiled_op);
    if (failed(combiner_or)) return failure();
    Operation *combiner = combiner_or.getValue();

    auto accumulator = rewriter.create<GenericOp>(
        tiled_op.getLoc(), partial_result.getType(),
        makeArrayRef(partial_result), makeArrayRef((Value)extract_output_slice),
        makeArrayRef({id_map, id_map}), parallel_iter_types,
        [&](OpBuilder &b, Location nested_loc, ValueRange args) {
          BlockAndValueMapping bvm;
          bvm.map(combiner->getOperands(), args);
          Value result_val = b.clone(*combiner, bvm)->getResult(0);
          b.create<YieldOp>(nested_loc, result_val);
        });

    rewriter.updateRootInPlace(insert_output_slice, [&]() {
      insert_output_slice.getSourceMutable().assign(accumulator.getResult(0));
    });
    return success();
  }

  // Unfortunaly, there is no way to modify the results of the loop inplace. So
  // we have to replace it with a clone.
  LoopOp CreateLoopWithUpdatedResults(PatternRewriter &rewriter,
                                      LoopOp tiled_loop) const {
    auto loc = tiled_loop.getLoc();
    rewriter.setInsertionPoint(tiled_loop);
    auto new_loop = rewriter.create<LoopOp>(
        loc, mlir::TypeRange(tiled_loop.outputs()), tiled_loop.getOperands(),
        tiled_loop->getAttrs());
    rewriter.inlineRegionBefore(tiled_loop.region(), new_loop.region(),
                                new_loop.region().begin());

    rewriter.replaceOp(tiled_loop, new_loop.getResult(0));
    return new_loop;
  }

  // Fuses FillOp producer of the output argument of the LoopOp and inserts
  // an operation that accumulates the partial result, i.e. reduced tile, and
  // the current value of the output tile.
  LogicalResult RewriteTiledReduction(PatternRewriter &rewriter,
                                      LoopOp tiled_loop,
                                      LinalgOp tiled_op) const {
    OpBuilder::InsertionGuard guard(rewriter);
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
    BlockArgument output_tile_bb_arg =
        CloneAndAppendInitTensorToTiledLoop(rewriter, fill, tiled_loop);
    FuseFill(rewriter, tiled_op, fill, loop_output_bb_arg, output_tile_bb_arg,
             extract_output_slice, insert_output_slice);
    // We have already modified the loop above, so we need to update the
    // results.
    CreateLoopWithUpdatedResults(rewriter, tiled_loop);
    return CombineReducedTileWithOutput(rewriter, tiled_op, tiled_op_result,
                                        extract_output_slice,
                                        insert_output_slice);
  }
};

struct FuseFillIntoTiledReductionPass
    : public impl::FuseFillIntoTiledReductionBase<
          FuseFillIntoTiledReductionPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<FuseFillIntoTiledReductionPattern>(context);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseFillIntoTiledReductionPass() {
  return std::make_unique<FuseFillIntoTiledReductionPass>();
}

}  // namespace tensorflow
