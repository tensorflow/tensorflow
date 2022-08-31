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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/rewriters.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

Value createPoint(OpBuilder &b, Location loc, Value superset, ValueRange ivs) {
  ArrayAttr allDynamicOffsetsAttr = b.getI64ArrayAttr(
      SmallVector<int64_t>(ivs.size(), ShapedType::kDynamicStrideOrOffset));
  return b.create<PointOp>(loc, superset, ivs, allDynamicOffsetsAttr);
}

Value createTile(OpBuilder &b, Location loc, Value superset, ValueRange ivs,
                 ValueRange upperBounds, ValueRange steps,
                 ArrayRef<int64_t> tileSizes) {
  // Compute the actual size of the tile.
  ArrayRef<int64_t> supersetShape =
      superset.getType().cast<TileType>().getShape();
  uint64_t rank = supersetShape.size();
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  staticSizes.reserve(rank);
  for (auto i : llvm::seq<int64_t>(0, rank)) {
    // If the dimension is perfectly tiled, use the statically known tile size.
    if (tileSizes[i] == 1 || (supersetShape[i] != ShapedType::kDynamicSize &&
                              supersetShape[i] % tileSizes[i] == 0)) {
      staticSizes.push_back(tileSizes[i]);
      continue;
    }

    // Otherwise, compute the tile size dynamically.
    auto remainderInDim = b.create<arith::SubIOp>(loc, upperBounds[i], ivs[i]);
    auto tileSizeInDim =
        b.create<arith::MinSIOp>(loc, steps[i], remainderInDim);
    staticSizes.push_back(ShapedType::kDynamicSize);
    dynamicSizes.push_back(tileSizeInDim);
  }

  auto tileTy = b.getType<TileType>(staticSizes);
  auto allDynamicOffsetsAttr = b.getI64ArrayAttr(
      SmallVector<int64_t>(rank, ShapedType::kDynamicStrideOrOffset));
  auto staticSizesAttr = b.getI64ArrayAttr(staticSizes);
  auto unitStridesAttr = b.getI64ArrayAttr(SmallVector<int64_t>(rank, 1));
  return b.create<TileOp>(loc, tileTy, superset, ivs, dynamicSizes,
                          ValueRange{}, allDynamicOffsetsAttr, staticSizesAttr,
                          unitStridesAttr);
}

Value createTileOrPoint(OpBuilder &b, Location loc, Value space, ValueRange ivs,
                        ValueRange upperBounds, ValueRange steps,
                        ArrayRef<int64_t> tileSizes) {
  if (llvm::all_of(tileSizes, [](int64_t d) { return d == 1; })) {
    return createPoint(b, loc, space, ivs);
  }
  return createTile(b, loc, space, ivs, upperBounds, steps, tileSizes);
}

Value createNestedPloopTilingRecursively(
    OpBuilder &b, Location loc, Value init, Value source,
    ArrayRef<SmallVector<int64_t>> nestedTileSizes) {
  assert(!nestedTileSizes.empty() && "expect tile sizes");

  // Create root space.
  auto sourceTy = source.getType().cast<RankedTensorType>();
  SmallVector<Value> sourceDynamicDims =
      tensor::createDynamicDimValues(b, loc, source);
  auto sourceSpaceTy = b.getType<TileType>(sourceTy.getShape());
  Value sourceSpace = b.create<SpaceOp>(loc, sourceSpaceTy, sourceDynamicDims,
                                        b.getI64ArrayAttr(sourceTy.getShape()));

  // Create loop bounds.
  SmallVector<Value> lowerBounds(sourceTy.getRank(),
                                 b.create<arith::ConstantIndexOp>(loc, 0));
  SmallVector<Value> upperBounds = tensor::createDimValues(b, loc, source);
  SmallVector<Value> steps = llvm::to_vector(
      llvm::map_range(nestedTileSizes.front(), [&](int64_t s) -> Value {
        return b.create<arith::ConstantIndexOp>(loc, s);
      }));

  // Create ploop.
  auto ploop = b.create<ParallelOp>(
      loc, sourceTy, lowerBounds, upperBounds, steps,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        Value subset = createTileOrPoint(b, loc, sourceSpace, ivs, upperBounds,
                                         steps, nestedTileSizes.front());
        Value innerResult = b.create<MaterializeOp>(loc, source, subset);

        // Recur if there are more tile sizes, and it's not a point yet.
        nestedTileSizes = nestedTileSizes.drop_front();
        if (!nestedTileSizes.empty() && subset.getType().isa<TileType>()) {
          auto materializedInitSubset =
              b.create<MaterializeOp>(loc, init, subset);
          innerResult = createNestedPloopTilingRecursively(
              b, loc, materializedInitSubset, innerResult, nestedTileSizes);
        }

        b.create<SetYieldOp>(loc, ValueRange{innerResult}, ValueRange{init},
                             ValueRange{subset});
      });
  return ploop.getResults().front();
}

Value createNestedPloopTiling(OpBuilder &b, Location loc, Value source,
                              ArrayRef<SmallVector<int64_t>> &nestedTileSizes) {
  // Create init tensor.
  auto sourceTy = source.getType().cast<RankedTensorType>();
  SmallVector<Value> sourceDynamicDims =
      tensor::createDynamicDimValues(b, loc, source);
  auto init = b.create<linalg::InitTensorOp>(
      loc, sourceDynamicDims, sourceTy.getShape(), sourceTy.getElementType());

  return createNestedPloopTilingRecursively(b, loc, init, source,
                                            nestedTileSizes);
}

LogicalResult tileUniqueFunctionResult(
    func::FuncOp f, ArrayRef<SmallVector<int64_t>> nestedTileSizes) {
  assert(!nestedTileSizes.empty() && "expect tile sizes");

  // Apply to functions that return a single ranked tensor.
  FunctionType funcTy = f.getFunctionType();
  if (funcTy.getNumResults() != 1) return failure();
  auto resultTy = funcTy.getResults().front().dyn_cast<RankedTensorType>();
  if (!resultTy) return failure();

  // Only apply to single-block functions.
  llvm::iplist<Block> &allBlocks = f.getBody().getBlocks();
  if (allBlocks.size() != 1) return failure();
  Block &block = allBlocks.front();

  // Find return op and the unique source value to be tiled.
  auto returnOp = llvm::dyn_cast<func::ReturnOp>(block.getTerminator());
  Value source = returnOp.getOperands().front();
  auto sourceTy = source.getType().cast<RankedTensorType>();

  // All nested tiles must be of the same rank as the source value.
  int64_t rank = sourceTy.getRank();
  if (llvm::any_of(nestedTileSizes, [&](auto it) {
        return static_cast<int64_t>(it.size()) != rank;
      })) {
    return failure();
  }

  // Create tiled implementation right before the return op.
  OpBuilder b(f.getContext());
  b.setInsertionPoint(returnOp);
  Value tiledSource =
      createNestedPloopTiling(b, source.getLoc(), source, nestedTileSizes);

  // Return the tiled value.
  b.create<func::ReturnOp>(returnOp.getLoc(), tiledSource);
  returnOp.erase();
  return success();
}

// Parse comma-separated integeres as tile sizes:
//   <tile-sizes> ::== `[` <int> ( `,` <int> )* `]`
llvm::Optional<SmallVector<int64_t>> parseTileSizes(
    std::istringstream &istream) {
  SmallVector<int64_t> tileSizes;

  // Parse opening bracket `[`.
  if (istream.peek() != '[') return llvm::None;
  istream.get();

  // Parse leading extent.
  int64_t value;
  istream >> value;
  tileSizes.push_back(value);

  // Parse trailing extents.
  while (istream.peek() == ',') {
    istream.get();
    istream >> value;
    tileSizes.push_back(value);
  }

  // Parse closing bracket `]`.
  if (istream.peek() != ']') return llvm::None;
  istream.get();

  return tileSizes;
}

// Parse comma-sepatated nested tile sizes:
//   <nested-tile-sizes> ::== <tile-sizes> ( `,` <tile-sizes> )*
llvm::Optional<SmallVector<SmallVector<int64_t>>> parseNestedTileSizes(
    std::istringstream &istream) {
  SmallVector<SmallVector<int64_t>> nestedTileSizes;

  // Parse leading tile sizes.
  llvm::Optional<SmallVector<int64_t>> tileSizes = parseTileSizes(istream);
  if (!tileSizes) return llvm::None;
  nestedTileSizes.push_back(*tileSizes);

  // Parse trailing tile sizes.
  while (istream.peek() == ',') {
    istream.get();
    tileSizes = parseTileSizes(istream);
    if (!tileSizes) return llvm::None;
    nestedTileSizes.push_back(*tileSizes);
  }

  // Ensure to fully parse the argument.
  if (!istream.eof()) return llvm::None;

  return nestedTileSizes;
}

llvm::Optional<SmallVector<SmallVector<int64_t>>> parseNestedTileSizes(
    const std::string &str) {
  std::istringstream istream(str);
  return parseNestedTileSizes(istream);
}

struct DeprecatedTilingPass
    : public DeprecatedTilingPassBase<DeprecatedTilingPass> {
  DeprecatedTilingPass() : DeprecatedTilingPassBase<DeprecatedTilingPass>() {
    tileSizesOpt.setCallback(
        [&](const std::string &str) { tileSizes = parseNestedTileSizes(str); });
  }
  explicit DeprecatedTilingPass(const std::string &tileSizesStr)
      : DeprecatedTilingPassBase<DeprecatedTilingPass>() {
    tileSizes = parseNestedTileSizes(tileSizesStr);
  }
  explicit DeprecatedTilingPass(
      const SmallVector<SmallVector<int64_t>> &tileSizes)
      : DeprecatedTilingPassBase<DeprecatedTilingPass>(),
        tileSizes(tileSizes) {}

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<linalg::LinalgDialect, tensor::TensorDialect, GmlStDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();

    // If tile sizes were provided in string form, e.g. in lit tests, we might
    // fail to parse them.
    if (!tileSizes) {
      f.emitError()
          << "Unknown tiling sizes (not provided or failed to parse them from '"
          << tileSizesOpt << "'";
      return signalPassFailure();
    }

    // Assert our expectation to tile functions with unique ranked tensor
    // results. This is important for the e2e tests to make sure that we
    // actually test a tiled implementation.
    FunctionType funcTy = f.getFunctionType();
    bool isTilingTarget = funcTy.getNumResults() == 1 &&
                          funcTy.getResults().front().isa<RankedTensorType>();
    if (isTilingTarget) {
      if (failed(tileUniqueFunctionResult(f, *tileSizes))) {
        return signalPassFailure();
      }
    }
  }

  llvm::Optional<SmallVector<SmallVector<int64_t>>> tileSizes;
};

struct TilingResult {
  TilingInterface tiledOp;
  Operation *loop;
};

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the tile processed within the inner most loop.
Operation *generateTileLoopNest(OpBuilder &builder, Location loc,
                                ArrayRef<Range> loopRanges,
                                ArrayRef<Value> tileSizeVals,
                                ArrayRef<Value> dstOperands, bool distribute,
                                SmallVector<OpFoldResult> &offsets,
                                SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable
  // of the tiled loop.
  AffineExpr s0, s1, d0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());

  SmallVector<Value> lbs, ubs, steps;
  SmallVector<unsigned> nonemptyRangeIndices;
  for (auto &loopRange : llvm::enumerate(loopRanges)) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    offsets.push_back(offset);
    sizes.push_back(size);
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) continue;
    lbs.push_back(offset);
    ubs.push_back(size);
    steps.push_back(tileSizeVals[loopRange.index()]);
    nonemptyRangeIndices.push_back(loopRange.index());
  }

  auto buildBody = [&](OpBuilder &nestedBuilder, Location bodyLoc,
                       ValueRange ivs) {
    for (const auto &en : llvm::enumerate(ivs)) {
      Value iv = en.value();
      size_t index = en.index();
      Value boundedTileSize = nestedBuilder.create<AffineMinOp>(
          bodyLoc, minMap, ValueRange{iv, steps[index], ubs[index]});
      sizes[nonemptyRangeIndices[index]] = boundedTileSize;
      offsets[nonemptyRangeIndices[index]] = iv;
    }
  };
  Operation *loop =
      distribute
          ? builder
                .create<gml_st::ParallelOp>(
                    loc, TypeRange(ValueRange{dstOperands}), lbs, ubs, steps,
                    [&](OpBuilder &nestedBuilder, Location bodyLoc,
                        ValueRange ivs) {
                      buildBody(nestedBuilder, bodyLoc, ivs);
                    })
                .getOperation()
          : builder
                .create<gml_st::ForOp>(
                    loc, TypeRange(ValueRange{dstOperands}), lbs, ubs, steps,
                    dstOperands,
                    [&](OpBuilder &nestedBuilder, Location bodyLoc,
                        ValueRange ivs, ValueRange /*inits*/) {
                      buildBody(nestedBuilder, bodyLoc, ivs);
                    })
                .getOperation();
  return loop;
}

/// Pattern to tile an op that implements the `TilingInterface` using
/// `gml_st.for` for iterating over the tiles.
struct TilingPattern : public OpInterfaceRewritePattern<TilingInterface> {
  TilingPattern(MLIRContext *context, OpFilterFn filterFn,
                TilingOptions options, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        filterFn(filterFn),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn || failed(filterFn(op)) || hasTransformationAttr(op))
      return failure();

    if (!options.tileSizeComputationFn) {
      return rewriter.notifyMatchFailure(
          op, "missing tile size computation function");
    }

    // Implement adding accumulator to the gml_st.parallel terminator.
    if (options.distribute &&
        llvm::any_of(op.getLoopIteratorTypes(), [](StringRef type) {
          return type == getReductionIteratorTypeName();
        }))
      return failure();

    // 1. Get the range of the loops that are represented by the operation.
    SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
    size_t numLoops = iterationDomain.size();
    if (numLoops == 0)
      return rewriter.notifyMatchFailure(op, "missing iteration domain");

    // 2. Materialize the tile sizes. Enforce the convention that "tiling by
    // zero" skips tiling a particular dimension. This convention is
    // significantly simpler to handle instead of adjusting affine maps to
    // account for missing dimensions.
    SmallVector<Value> tileSizeVector =
        options.tileSizeComputationFn(rewriter, op);
    if (tileSizeVector.size() < iterationDomain.size()) {
      auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
    }

    // 3. Materialize an empty loop nest that iterates over the tiles.
    auto dstOperands = op.getDestinationOperands(rewriter);
    SmallVector<OpFoldResult> offsets, sizes;
    TilingResult tilingResult;
    tilingResult.loop = generateTileLoopNest(
        rewriter, op.getLoc(), iterationDomain, tileSizeVector, dstOperands,
        options.distribute, offsets, sizes);
    Block *loopBody = &tilingResult.loop->getRegion(0).front();
    Operation *terminator = loopBody->getTerminator();
    rewriter.setInsertionPoint(terminator);

    // 4. Insert the tiled implementation within the loop.
    tilingResult.tiledOp =
        op.getTiledImplementation(rewriter, dstOperands, offsets, sizes, true);

    // 5. Add `gml_st.set_yield` terminator.
    SmallVector<Value> dstSubsets;
    for (Value dst : tilingResult.tiledOp.getDestinationOperands(rewriter))
      dstSubsets.push_back(dst.getDefiningOp<MaterializeOp>().set());
    rewriter.replaceOpWithNewOp<SetYieldOp>(terminator,
                                            tilingResult.tiledOp->getResults(),
                                            dstOperands, dstSubsets);

    // 6. Replace the uses of `outputs` with the output block arguments.
    if (!options.distribute) {
      auto forLoop = cast<gml_st::ForOp>(tilingResult.loop);
      for (auto [dst, regionArg] :
           llvm::zip(dstOperands, forLoop.getRegionOutputArgs())) {
        dst.replaceUsesWithIf(regionArg, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == loopBody;
        });
      }
    }
    rewriter.replaceOp(op, tilingResult.loop->getResults());
    setTransformationAttr(rewriter, tilingResult.tiledOp);
    return success();
  }

 private:
  OpFilterFn filterFn;
  TilingOptions options;
};

struct TilingPass : public TilingPassBase<TilingPass> {
  TilingPass() = default;
  TilingPass(StringRef label, bool distributeFlag,
             llvm::ArrayRef<int64_t> sizes) {
    tilingTarget = label.str();
    distribute = distributeFlag;
    tileSizes = sizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    TilingOptions opts;
    opts.distribute = distribute;
    SmallVector<int64_t> ts(tileSizes.begin(), tileSizes.end());
    opts.tileSizeComputationFn = [ts](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());
      return llvm::to_vector<4>(llvm::map_range(ts, [&](int64_t s) {
        Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
        return v;
      }));
    };

    auto filterFn = [&](Operation *op) {
      return success(hasMatchingLabel(op, tilingTarget));
    };
    RewritePatternSet patterns(ctx);
    populateTilingPatterns(ctx, filterFn, opts, &patterns);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();

    // Clean up by removing temporary attributes.
    f.walk([](Operation *op) { removeTransformationAttr(op); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass() {
  return std::make_unique<DeprecatedTilingPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass(
    const SmallVector<SmallVector<int64_t>> &tileSizes) {
  return std::make_unique<DeprecatedTilingPass>(tileSizes);
}

std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass(
    const std::string &tileSizes) {
  return std::make_unique<DeprecatedTilingPass>(tileSizes);
}

void populateTilingPatterns(MLIRContext *context, OpFilterFn filterFn,
                            const TilingOptions &opts,
                            RewritePatternSet *patterns) {
  patterns->add<TilingPattern>(context, filterFn, opts);
}

std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    StringRef tilingTarget, bool distribute, ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingPass>(tilingTarget, distribute, tileSizes);
}

}  // namespace gml_st
}  // namespace mlir
