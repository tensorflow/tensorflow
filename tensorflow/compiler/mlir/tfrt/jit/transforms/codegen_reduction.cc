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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h.inc"

using llvm::makeArrayRef;
using mlir::BlockAndValueMapping;
using mlir::dyn_cast;
using mlir::failure;
using mlir::FailureOr;
using mlir::Identifier;
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
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::InitTensorOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingLoopType;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;
using mlir::linalg::PaddingValueComputationFunction;
using mlir::linalg::PadTensorOp;
using mlir::linalg::TiledLoopOp;
using mlir::linalg::YieldOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractSliceOp;
using mlir::tensor::InsertSliceOp;

// Tiles a GenericOp that models a 2D row or column reduction.
struct RowOrColumnReductionTilingPattern : public OpRewritePattern<GenericOp> {
  RowOrColumnReductionTilingPattern(const LinalgTilingOptions &options,
                                    const LinalgTransformationFilter &filter,
                                    MLIRContext *context,
                                    mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    if (linalg_op.getNumOutputs() != 1) return failure();
    if (linalg_op.getNumLoops() != 2) return failure();

    auto tiled_op = tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_op)) return failure();

    tiled_op->loops.front()->walk([&](LinalgOp tOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tOp);
    });

    rewriter.replaceOp(linalg_op, tiled_op->tensorResults);
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Rewrites a 1D reduction for vectorization. Matches `linalg.generic` that
// combines elements of tensor<?xELEM_TYPE> into tensor<ELEM_TYPE> and then
// creates a loop to reduce tensor<?xELEM_TYPE> -> tensor<VECTOR_SIZExELEM_TYPE>
// and an additional `linalg.generic` that reduces tensor<VECTOR_SIZExELEM_TYPE>
// to tensor<ELEM_TYPE>.
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
// %vector_result = linalg.tiled_loop (%i)
//     = (%c0) to (%INPUT_SIZE) step (%vector_size)
//     ins (%input_ = %input: tensor<?xf32>)
//     outs (%tmp_result_ = %tmp_result: tensor<VECTOR_SIZExf32>)
//     iterators["reduction"] {
//   %tile = tensor.extract_slice %arg2[%i] [%TILE_SIZE] [1]
//     : tensor<?xf32> to tensor<?xf32>
//   %tile_pad = linalg.pad_tensor %tile
//     : tensor<?xf32> to tensor<VECTOR_SIZExf32>
//   %tile_reshape = tensor.expand_shape %tile_pad [[0, 1]]
//     : tensor<VECTOR_SIZExf32> into tensor<1xVECTOR_SIZExf32>
//   %combine = linalg.generic ins(%tile_reshape : tensor<1xVECTOR_SIZExf32>)
//     outs(%tmp_result_ : tensor<VECTOR_SIZExf32>) -> tensor<VECTOR_SIZExf32>
//   linalg.yield %combine : tensor<VECTOR_SIZExf32>
//   }
// %result = linalg.generic ins(%vector_result : tensor<VECTOR_SIZExf32>)
//   outs(%fill : tensor<f32>) -> tensor<f32>
//
// This is necessary to push horizontal reduction to the later stage.
struct OneDimReductionTilingPattern : public OpRewritePattern<GenericOp> {
  OneDimReductionTilingPattern(int64_t vector_size,
                               const LinalgTransformationFilter &filter,
                               mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        filter(filter),
        vector_size(vector_size) {}

  LogicalResult matchAndRewrite(GenericOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();
    if (linalg_op.getNumOutputs() != 1) return failure();

    // Check if all inputs have a 1D identity map.
    if (linalg_op.getNumLoops() != 1) return failure();
    auto indexing_maps = linalg_op.getIndexingMaps();
    for (auto affine_map : makeArrayRef(indexing_maps).drop_back()) {
      if (!affine_map.isIdentity()) return failure();
    }

    Location loc = linalg_op.getLoc();
    Value input = linalg_op.getInputOperand(0)->get();
    // All inputs have the same size because of identity maps for indexing.
    SmallVector<Value> inputs = linalg_op.inputs();
    Value input_size = rewriter.create<mlir::tensor::DimOp>(loc, input, 0);

    auto fill_op = linalg_op.outputs().front().getDefiningOp<FillOp>();
    auto init_op = fill_op.output().getDefiningOp<InitTensorOp>();

    auto neutral_value = fill_op.value();
    auto element_type = init_op.getType().getElementType();

    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value vector_size_value =
        rewriter.create<ConstantIndexOp>(loc, vector_size);
    Value new_init = rewriter.create<InitTensorOp>(loc, ValueRange{},
                                                   vector_size, element_type);
    Value new_fill =
        rewriter.create<FillOp>(loc, fill_op.value(), new_init).result();

    GenericOp tiled_reduction;
    auto tiled_loop_op = rewriter.create<TiledLoopOp>(
        loc, makeArrayRef(zero), makeArrayRef(input_size),
        makeArrayRef(vector_size_value), inputs, makeArrayRef(new_fill),
        rewriter.getStrArrayAttr(mlir::getReductionIteratorTypeName()),
        [&](OpBuilder &b, Location nested_loc, ValueRange ivs,
            ValueRange inputs, ValueRange outputs) {
          SmallVector<Value, 2> reshaped_tiled_inputs =
              TileAndReshapeInputTensors(b, nested_loc, ivs, inputs,
                                         neutral_value, input_size,
                                         vector_size_value);
          // Create `linalg.generic` to combine
          // `tensor<1xVECTOR_SIZExELEM_TYPE>1 input with the
          // `tensor<VECTOR_SIZExELEM_TYPE>` output.
          SmallVector<mlir::StringRef, 2> iter_types{
              mlir::getReductionIteratorTypeName(),
              mlir::getParallelIteratorTypeName()};
          SmallVector<mlir::AffineMap, 2> indexing_maps(
              inputs.size(), rewriter.getMultiDimIdentityMap(2));
          indexing_maps.push_back(
              mlir::AffineMap::get(2, 0, b.getAffineDimExpr(1)));
          tiled_reduction = b.create<GenericOp>(
              nested_loc, outputs[0].getType(), reshaped_tiled_inputs,
              makeArrayRef({outputs[0]}), indexing_maps, iter_types,
              /*bodyBuild=*/nullptr);
          mlir::Region &region = tiled_reduction.region();
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.cloneRegionBefore(linalg_op.region(), region, region.end());
          b.create<YieldOp>(nested_loc, tiled_reduction.getResult(0));
        });
    // Create `linalg.generic` to reduce
    // tensor<VECTOR_SIZExELEM_TYPE>->tensor<ELEM_TYPE>.
    auto final_reduction_or =
        ReduceVectorIntoOutput(rewriter, linalg_op, tiled_loop_op.getResult(0));
    if (failed(final_reduction_or)) return failure();
    auto final_reduction = final_reduction_or.getValue();
    rewriter.replaceOp(linalg_op, final_reduction->getResults());

    tiled_loop_op->walk([&](GenericOp op) {
      filter.replaceLinalgTransformationFilter(rewriter, op);
      filter.replaceLinalgTransformationFilter(rewriter, final_reduction);
    });
    return success();
  }

  // Tiles, pads and reshapes every input argument of type tensor<?xELEM_TYPE>
  // into tensor<1xVECTOR_SIZExELEM_TYPE>.
  SmallVector<Value, 2> TileAndReshapeInputTensors(
      OpBuilder &b, Location nested_loc, ValueRange ivs, ValueRange inputs,
      Value neutral_value, Value input_size, Value vector_size_value) const {
    SmallVector<Value, 2> reshaped_tiled_inputs;

    SmallVector<mlir::ReassociationIndices> indices = {{0, 1}};
    auto identity_1d_map = b.getMultiDimIdentityMap(1);
    auto iv = ivs.front();

    auto tile_sizes = mlir::linalg::computeTileSizes(
        b, nested_loc, ivs, vector_size_value, input_size);
    for (auto input : inputs) {
      // Extract slice of input.
      Value slice = mlir::linalg::makeTiledShape(
          b, nested_loc, input, vector_size_value, identity_1d_map, iv,
          input_size, tile_sizes);
      auto element_type = slice.getType().cast<ShapedType>().getElementType();

      // Pad input tile.
      Value pad = PadTensorOp::createPadHighOp(
          RankedTensorType::get({vector_size}, element_type), slice,
          neutral_value, false, nested_loc, b);

      // Reshape input tile to tensor<1xVECTOR_SIZExELEM_TYPE>.
      Value expand_shape = b.create<ExpandShapeOp>(
          nested_loc, RankedTensorType::get({1, vector_size}, element_type),
          pad, indices);
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
    SmallVector<mlir::StringRef, 3> reduction_iter_type(
        1, mlir::getReductionIteratorTypeName());
    auto map = mlir::AffineMap::get(1, 0, llvm::None, rewriter.getContext());

    auto combiner_or = DetectCombiner(linalg_op);
    if (failed(combiner_or)) return failure();
    Operation *combiner = combiner_or.getValue();

    auto accumulator = rewriter.create<GenericOp>(
        linalg_op.getLoc(), linalg_op->getResultTypes(),
        makeArrayRef(partial_result),
        makeArrayRef(linalg_op.getOutputOperand(0)->get()),
        makeArrayRef({rewriter.getMultiDimIdentityMap(1), map}),
        reduction_iter_type,
        [&](OpBuilder &b, Location nested_loc, ValueRange args) {
          BlockAndValueMapping bvm;
          bvm.map(combiner->getOperands(), args);
          Value result_val = b.clone(*combiner, bvm)->getResult(0);
          b.create<YieldOp>(nested_loc, result_val);
        });
    return accumulator;
  }

 private:
  LinalgTransformationFilter filter;
  int64_t vector_size;
};

// Match 1D or 2D reduction.
bool isCanonicalizedReduction(Operation *op) {
  auto reduction = mlir::dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumLoops() > 2) return false;
  return reduction.getNumReductionLoops() == 1;
}

struct CodegenReductionPass
    : public CodegenReductionBase<CodegenReductionPass> {
  CodegenReductionPass() = default;
  CodegenReductionPass(int64_t reduction_1d_tile,
                       llvm::ArrayRef<int64_t> reduction_2d_tiles) {
    reduction_1d_tile_size = reduction_1d_tile;
    reduction_2d_tile_sizes = reduction_2d_tiles;
  }
  void runOnFunction() override {
    auto func = getFunction();
    auto context = func.getContext();

    auto filter = LinalgTransformationFilter(
                      llvm::None, {Identifier::get("tiled", context)})
                      .addFilter([](Operation *op) {
                        return success(isCanonicalizedReduction(op));
                      });
    auto patterns =
        mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
    patterns.insert<OneDimReductionTilingPattern>(
        reduction_1d_tile_size, filter, patterns.getContext());

    assert(reduction_2d_tile_sizes.size() == 2 &&
           "Tiling sizes for 2D reductions should have two elements");
    patterns.insert<RowOrColumnReductionTilingPattern>(
        LinalgTilingOptions{}
            .setTileSizes(reduction_2d_tile_sizes)
            .setLoopType(LinalgTilingLoopType::TiledLoops),
        filter, patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass() {
  return std::make_unique<CodegenReductionPass>();
}

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass(
    int64_t reduction_1d_tile_size,
    llvm::ArrayRef<int64_t> reduction_2d_tile_sizes) {
  return std::make_unique<CodegenReductionPass>(reduction_1d_tile_size,
                                                reduction_2d_tile_sizes);
}

}  // namespace tensorflow
