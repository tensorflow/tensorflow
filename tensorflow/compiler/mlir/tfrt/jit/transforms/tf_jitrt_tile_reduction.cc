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
#include <vector>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_TILEREDUCTION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::makeArrayRef;
using mlir::BlockAndValueMapping;
using mlir::dyn_cast;
using mlir::failure;
using mlir::FailureOr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::IteratorTypeAttr;
using mlir::gml_st::LoopOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractSliceOp;
using mlir::utils::IteratorType;

static constexpr llvm::StringRef kTileReductionAppliedLabel =
    "__tile_reduction_applied_label__";

// Match 1D or 2D reduction.
bool isCanonicalizedReduction(Operation *op) {
  auto reduction = mlir::dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumDpsInits() != 1) return false;
  if (reduction.getNumLoops() > 2) return false;
  return reduction.getNumReductionLoops() == 1;
}

// Tiles a GenericOp that models a 2D row or column reduction.
struct RowOrColumnReductionTilingPattern : public OpRewritePattern<GenericOp> {
  RowOrColumnReductionTilingPattern(const LinalgTilingOptions &options,
                                    MLIRContext *context,
                                    mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (mlir::gml_st::hasLabel(linalg_op, kTileReductionAppliedLabel))
      return failure();
    if (!isCanonicalizedReduction(linalg_op)) return failure();

    if (linalg_op.getNumDpsInits() != 1) return failure();
    if (linalg_op.getNumLoops() != 2) return failure();

    auto tiled_op = mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_op)) return failure();

    tiled_op->loops.front()->walk([&](LinalgOp tOp) {
      mlir::gml_st::setLabel(tOp, kTileReductionAppliedLabel);
    });

    rewriter.replaceOp(linalg_op, tiled_op->tensorResults);
    return success();
  }

 private:
  LinalgTilingOptions options;
};

// Rewrites a 1D reduction for vectorization. Matches `linalg.generic` that
// combines elements of tensor<?xELEM_TYPE> into tensor<ELEM_TYPE> and then
// creates a perfectly-tilable loop to reduce tensor<?xELEM_TYPE> ->
// tensor<VECTOR_SIZExELEM_TYPE> and an additional `linalg.generic` that reduces
// tensor<VECTOR_SIZExELEM_TYPE> to tensor<ELEM_TYPE>.
//
// Example:
//
// %sum = linalg.generic {
//   indexing_maps = [affine_map<(d0) -> (d0)>,
//                    affine_map<(d0) -> ()>],
//   iterator_types = ["reduction"]}
//   ins(%input : tensor<?xf32>)
//   outs(%fill : tensor<f32>) {
// ^bb0(%in: f32, %out: f32):
//   %add = arith.addf %in, %out : f32
//   linalg.yield %add : f32
// } -> tensor<f32>
//
// will be rewritten as
//
// %vector_result = gml_st.loop (%i)
//     = (%c0) to (%TILABLE_UB) step (%vector_size)
//     ins (%input_ = %input: tensor<?xf32>)
//     outs (%tmp_result_ = %tmp_result: tensor<VECTOR_SIZExf32>)
//     iterators["reduction"] {
//   %tile = tensor.extract_slice %arg2[%i] [%TILE_SIZE] [1]
//     : tensor<?xf32> to tensor<TILE_SIZExf32>
//   %tile_reshape = tensor.expand_shape %tile [[0, 1]]
//     : tensor<VECTOR_SIZExf32> into tensor<1xVECTOR_SIZExf32>
//   %combine = linalg.generic ins(%tile_reshape : tensor<1xVECTOR_SIZExf32>)
//     outs(%tmp_result_ : tensor<VECTOR_SIZExf32>) -> tensor<VECTOR_SIZExf32>
//   linalg.yield %combine : tensor<VECTOR_SIZExf32>
// }
// %horizontal_reduce = linalg.generic
//   ins(%vector_result : tensor<VECTOR_SIZExf32>)
//   outs(%fill : tensor<f32>) -> tensor<f32> // combiner only
// %result = gml_st.loop (%i)
//     = (%TILABLE_UB) to (%INPUT_SIZE) step (%vector_size)
//     ins (%input_ = %input: tensor<?xf32>)
//     outs (%tmp_result_ = %horizontal_reduce: tensor<f32>)
//     iterators["reduction"] {
//   linalg.generic // reduces the tail
// }
//
// This is necessary to push horizontal reduction to the later stage.
struct OneDimReductionTilingPattern : public OpRewritePattern<GenericOp> {
  OneDimReductionTilingPattern(int64_t vector_size, int64_t tile_size,
                               mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        vector_size(vector_size),
        tile_size(tile_size) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (mlir::gml_st::hasLabel(linalg_op, kTileReductionAppliedLabel))
      return failure();
    if (!isCanonicalizedReduction(linalg_op)) return failure();

    // Check if all inputs have a 1D identity map.
    if (linalg_op.getNumLoops() != 1) return failure();
    auto indexing_maps = linalg_op.getIndexingMapsArray();
    for (auto affine_map : makeArrayRef(indexing_maps).drop_back()) {
      if (!affine_map.isIdentity()) return failure();
    }

    Location loc = linalg_op.getLoc();
    Value input = linalg_op.getDpsInputOperand(0)->get();
    // All inputs have the same size because of identity maps for indexing.
    SmallVector<Value> inputs = linalg_op.getInputs();
    Value input_size = rewriter.create<mlir::tensor::DimOp>(loc, input, 0);

    auto fill_op = linalg_op.getOutputs().front().getDefiningOp<FillOp>();
    auto empty_op = fill_op.output().getDefiningOp<mlir::tensor::EmptyOp>();

    auto neutral_value = fill_op.value();
    auto element_type = empty_op.getType().getElementType();

    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value tile_size_value = rewriter.create<ConstantIndexOp>(loc, tile_size);
    Value new_empty =
        rewriter.create<mlir::tensor::EmptyOp>(loc, vector_size, element_type);
    Value new_fill =
        rewriter.create<FillOp>(loc, fill_op.value(), new_empty).result();

    llvm::Optional<Value> tilable_bound_or =
        getTilableBound(rewriter, loc, zero, input_size, tile_size_value);
    Value tilable_bound =
        tilable_bound_or.has_value() ? *tilable_bound_or : input_size;

    GenericOp tiled_reduction;
    auto perfectly_tiled_loop = rewriter.create<LoopOp>(
        loc, makeArrayRef(zero), makeArrayRef(tilable_bound),
        makeArrayRef(tile_size_value), inputs, makeArrayRef(new_fill),
        rewriter.getArrayAttr({IteratorTypeAttr::get(rewriter.getContext(),
                                                     IteratorType::reduction)}),
        [&](OpBuilder &b, Location nested_loc, ValueRange ivs,
            ValueRange inputs, ValueRange outputs) {
          SmallVector<Value, 2> reshaped_tiled_inputs =
              TileAndReshapeInputTensors(b, nested_loc, ivs, inputs,
                                         neutral_value, input_size,
                                         tile_size_value);
          // Create `linalg.generic` to combine
          // `tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE> input with
          // the `tensor<VECTOR_SIZExELEM_TYPE>` output.
          SmallVector<IteratorType, 2> iter_types{IteratorType::reduction,
                                                  IteratorType::parallel};
          SmallVector<mlir::AffineMap, 2> indexing_maps(
              inputs.size(), rewriter.getMultiDimIdentityMap(2));
          indexing_maps.push_back(
              mlir::AffineMap::get(2, 0, b.getAffineDimExpr(1)));
          tiled_reduction = b.create<GenericOp>(
              nested_loc, outputs[0].getType(), reshaped_tiled_inputs,
              makeArrayRef({outputs[0]}), indexing_maps, iter_types,
              /*bodyBuild=*/nullptr);
          mlir::Region &region = tiled_reduction.getRegion();
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.cloneRegionBefore(linalg_op.getRegion(), region,
                                     region.end());
          b.create<mlir::gml_st::YieldOp>(nested_loc,
                                          tiled_reduction.getResult(0));
        });
    // Create `linalg.generic` to reduce
    // tensor<VECTOR_SIZExELEM_TYPE>->tensor<ELEM_TYPE>.
    auto horizontal_reduction_or = ReduceVectorIntoOutput(
        rewriter, linalg_op, perfectly_tiled_loop.getResult(0));
    if (failed(horizontal_reduction_or)) return failure();
    auto horizontal_reduction = horizontal_reduction_or.value();
    Value result = horizontal_reduction->getResult(0);

    // If the loop was not perfectly tiled, then we have to combine
    // `horizontal_reduction` with the elements in the `tail`.
    if (tilable_bound_or.has_value()) {
      auto final_reduction = rewriter.create<LoopOp>(
          loc, tilable_bound, input_size, tile_size_value, inputs,
          makeArrayRef(result),
          rewriter.getArrayAttr({IteratorTypeAttr::get(
              rewriter.getContext(), IteratorType::reduction)}),
          [&](OpBuilder &b, Location nested_loc, ValueRange ivs,
              ValueRange inputs, ValueRange outputs) {
            BlockAndValueMapping bvm;
            mlir::AffineExpr sym0, sym1;
            bindSymbols(b.getContext(), sym0, sym1);
            auto diff_map = mlir::AffineMap::get(0, 2, {sym1 - sym0});

            Value one = b.create<ConstantIndexOp>(nested_loc, 1);
            auto size = b.createOrFold<mlir::AffineApplyOp>(
                nested_loc, diff_map, ValueRange{tilable_bound, input_size});
            std::vector<Value> sliced_inputs;
            sliced_inputs.reserve(inputs.size());
            for (Value input : inputs) {
              sliced_inputs.push_back(
                  b.create<ExtractSliceOp>(nested_loc, input, ivs, size, one));
            }
            bvm.map(linalg_op.getInputs(), sliced_inputs);
            bvm.map(linalg_op.getOutputs(), outputs);
            auto new_linalg_op = b.clone(*linalg_op.getOperation(), bvm);
            mlir::gml_st::setLabel(new_linalg_op, kTileReductionAppliedLabel);
            b.create<mlir::gml_st::YieldOp>(nested_loc,
                                            new_linalg_op->getResult(0));
          });
      result = final_reduction.getResult(0);
    }
    rewriter.replaceOp(linalg_op, result);

    perfectly_tiled_loop->walk([&](GenericOp op) {
      mlir::gml_st::setLabel(op, kTileReductionAppliedLabel);
    });
    mlir::gml_st::setLabel(horizontal_reduction, kTileReductionAppliedLabel);
    return success();
  }

 private:
  // Computes an upper bound that can be perfectly tiled. Return llvm::None, if
  // the loop is already perfectly tiled.
  mlir::Optional<Value> getTilableBound(OpBuilder &b, Location loc, Value lb,
                                        Value ub, Value step) const {
    auto lb_int = getConstantIntValue(lb);
    auto ub_int = getConstantIntValue(ub);
    auto step_int = getConstantIntValue(step);

    // No specialization necessary if step already divides upper bound evenly.
    if (lb_int && ub_int && step_int && (*ub_int - *lb_int) % *step_int == 0)
      return llvm::None;
    // No specialization necessary if step size is 1.
    if (mlir::isConstantIntValue(step, 1)) return llvm::None;
    mlir::AffineExpr sym0, sym1, sym2;
    bindSymbols(b.getContext(), sym0, sym1, sym2);

    // New upper bound: %ub - (%ub - %lb) mod %step
    auto mod_map = mlir::AffineMap::get(0, 3, {sym1 - ((sym1 - sym0) % sym2)});
    return {b.createOrFold<mlir::AffineApplyOp>(loc, mod_map,
                                                ValueRange{lb, ub, step})};
  }

  // Tiles, pads and reshapes every input argument of type tensor<?xELEM_TYPE>
  // into tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
  SmallVector<Value, 2> TileAndReshapeInputTensors(
      OpBuilder &b, Location nested_loc, ValueRange ivs, ValueRange inputs,
      Value neutral_value, Value input_size, Value tile_size_value) const {
    SmallVector<Value, 2> reshaped_tiled_inputs;

    SmallVector<mlir::ReassociationIndices> indices = {{0, 1}};
    auto identity_1d_map = b.getMultiDimIdentityMap(1);
    auto iv = ivs.front();

    mlir::OpFoldResult tile_size_fold = tile_size_value;
    mlir::OpFoldResult input_size_fold = input_size;
    auto tile_sizes = mlir::linalg::computeTileSizes(
        b, nested_loc, tile_size_fold, input_size_fold);
    for (auto input : inputs) {
      // Extract slice of input.
      Value slice = mlir::linalg::makeTiledShape(
          b, nested_loc, input, tile_size_fold, identity_1d_map,
          mlir::OpFoldResult(iv), input_size_fold, tile_sizes,
          /*omitPartialTileCheck=*/true);
      auto element_type = slice.getType().cast<ShapedType>().getElementType();

      // Reshape input tile to
      // tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
      Value expand_shape = b.create<ExpandShapeOp>(
          nested_loc,
          RankedTensorType::get({tile_size / vector_size, vector_size},
                                element_type),
          slice, indices);
      reshaped_tiled_inputs.push_back(expand_shape);
    }
    return reshaped_tiled_inputs;
  }

  // Creates `linalg.generic` to reduce
  // tensor<VECTOR_SIZExELEM_TYPE>->tensor<ELEM_TYPE>. To perform that we match
  // the combiner in the original "untiled" linalg_op.
  FailureOr<GenericOp> ReduceVectorIntoOutput(PatternRewriter &rewriter,
                                              LinalgOp linalg_op,
                                              Value partial_result) const {
    SmallVector<IteratorType, 3> reduction_iter_type(1,
                                                     IteratorType::reduction);
    auto map = mlir::AffineMap::get(1, 0, llvm::None, rewriter.getContext());

    auto combiner_or = DetectCombiner(linalg_op);
    if (failed(combiner_or)) return failure();
    Operation *combiner = combiner_or.value();

    auto accumulator = rewriter.create<GenericOp>(
        linalg_op.getLoc(), linalg_op->getResultTypes(),
        makeArrayRef(partial_result),
        makeArrayRef(linalg_op.getDpsInitOperand(0)->get()),
        makeArrayRef({rewriter.getMultiDimIdentityMap(1), map}),
        reduction_iter_type,
        [&](OpBuilder &b, Location nested_loc, ValueRange args) {
          BlockAndValueMapping bvm;
          bvm.map(combiner->getOperands(), args);
          Value result_val = b.clone(*combiner, bvm)->getResult(0);
          b.create<mlir::linalg::YieldOp>(nested_loc, result_val);
        });
    return accumulator;
  }

 private:
  int64_t vector_size;
  int64_t tile_size;
};

struct TileReductionPass : public impl::TileReductionBase<TileReductionPass> {
  TileReductionPass() = default;
  TileReductionPass(int64_t vector_size, int64_t reduction_1d_tile,
                    llvm::ArrayRef<int64_t> reduction_2d_tiles) {
    reduction_vector_size = vector_size;
    reduction_1d_tile_size = reduction_1d_tile;
    reduction_2d_tile_sizes = reduction_2d_tiles;
  }
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    assert(reduction_1d_tile_size % reduction_vector_size == 0 &&
           "Tile size for 1D reduction should be a multiple of vector size");
    auto patterns =
        mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.add<OneDimReductionTilingPattern>(
        reduction_vector_size, reduction_1d_tile_size, patterns.getContext());

    assert(reduction_2d_tile_sizes.size() == 2 &&
           "Tiling sizes for 2D reductions should have two elements");
    patterns.add<RowOrColumnReductionTilingPattern>(
        LinalgTilingOptions{}.setTileSizes(reduction_2d_tile_sizes),
        patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](LinalgOp op) {
      mlir::gml_st::removeLabel(op, kTileReductionAppliedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileReductionPass() {
  return std::make_unique<TileReductionPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileReductionPass(int64_t reduction_vector_size,
                        int64_t reduction_1d_tile_size,
                        llvm::ArrayRef<int64_t> reduction_2d_tile_sizes) {
  return std::make_unique<TileReductionPass>(
      reduction_vector_size, reduction_1d_tile_size, reduction_2d_tile_sizes);
}

}  // namespace tensorflow
