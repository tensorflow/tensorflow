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
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMATMULFORCPUPASS
#define GEN_PASS_DEF_SIMPLIFYDEADCOPYPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kMatmulTransformedLabel =
    "__matmul_transformed_label__";

// Helper to pick the tile shapes to use as the 2 inner dimensions of the
// 4D shapes appearing in a Mmt4D.
class Mmt4DTileParams {
 public:
  Mmt4DTileParams(ArrayRef<int> m0k0n0, const llvm::StringRef comment)
      : m0(m0k0n0[0]), k0(m0k0n0[1]), n0(m0k0n0[2]), comment(comment) {}
  std::array<int64_t, 2> lhs() const { return {m0, k0}; }
  std::array<int64_t, 2> rhs() const { return {k0, n0}; }
  std::array<int64_t, 2> acc() const { return {m0, n0}; }
  std::array<int64_t, 2> rhsTranspose() const { return {n0, k0}; }
  const std::string &getComment() const { return comment; }

 private:
  const int64_t m0;
  const int64_t k0;
  const int64_t n0;
  const std::string comment;
};

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = v.getType().cast<ShapedType>();
  if (!type.isDynamicDim(dim)) {
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType /*t*/) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType /*t*/) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

// Returns dimensions of |shapedTypeValue|, handling both static and dynamic
// shapes.
SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc,
                                  Value shapedTypeValue) {
  return llvm::to_vector(llvm::map_range(
      llvm::seq<int64_t>(
          0, shapedTypeValue.getType().cast<ShapedType>().getRank()),
      [&](int64_t dim) { return getDim(builder, loc, shapedTypeValue, dim); }));
}

std::optional<Value> getPaddingValue(Value &source) {
  auto padOp = source.getDefiningOp<tensor::PadOp>();
  if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad())
    return std::nullopt;

  Value constantPaddingValue = padOp.getConstantPaddingValue();
  if (!constantPaddingValue) return std::nullopt;

  source = padOp.getSource();
  return constantPaddingValue;
}

// Returns a tiled and packed value of |source|, the data layout is described by
// |innerDimsPos|, |innerTileSizes| and |outerDimsPerm|.
Value pack(Location loc, PatternRewriter &rewriter, Value source,
           ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> innerTileSizes,
           ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(innerTileSizes));
  auto empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, source, innerTileSizesOfr, innerDimsPos, outerDimsPerm);
  std::optional<Value> paddingValue = getPaddingValue(source);
  return rewriter.create<tensor::PackOp>(loc, source, empty, innerDimsPos,
                                         innerTileSizesOfr, paddingValue,
                                         outerDimsPerm);
}

// Returns an unpacked value of |source|, the data layout is described by
// |innerDimsPos|, |innerTileSizes| and |outerDimsPerm|. |resultShapeValue| is
// used to create the destination tensor for the resulting unpacked value.
Value unpack(Location loc, PatternRewriter &rewriter, Value source,
             Value resultShapeValue, ArrayRef<int64_t> innerDimsPos,
             ArrayRef<int64_t> innerTileSizes,
             ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> resultDims =
      getDims(rewriter, loc, resultShapeValue);
  auto empty = rewriter.create<tensor::EmptyOp>(
      loc, resultDims,
      source.getType().cast<RankedTensorType>().getElementType());

  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(innerTileSizes));

  return rewriter.create<tensor::UnPackOp>(loc, source, empty, innerDimsPos,
                                           innerTileSizesOfr, outerDimsPerm);
}

// Returns true if an input of the given |inputShape| needs padding to
// ensure that its shape will be a multiple of |tileShape|. That's always true
// in the dynamic shape case.
bool needsPadding(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> tileShape) {
  assert(inputShape.size() == tileShape.size());
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (inputShape[i] == ShapedType::kDynamic) {
      return true;
    }
    if (inputShape[i] % tileShape[i] != 0) {
      return true;
    }
  }
  return false;
}

// Pads |input| on the bottom and on the right to the next multiple of
// |tileShape|.
Value pad(Location loc, PatternRewriter &rewriter, Value input,
          ArrayRef<int64_t> tileShape) {
  SmallVector<OpFoldResult, 2> lowPadding, highPadding;
  SmallVector<int64_t, 2> resultTypeShape;
  auto inputType = input.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (!needsPadding(inputShape, tileShape)) {
    return input;
  }
  int64_t rank = inputType.getRank();
  for (int64_t i = 0; i < rank; ++i) {
    // No 'low' padding i.e. no padding at the top and on the left.
    lowPadding.push_back(rewriter.getIndexAttr(0));
    // 'High' padding i.e. padding at the bottom and on the right, and the
    // result type shape, will be dynamic in any dimension if and only if the
    // input shape is.
    if (inputShape[i] == ShapedType::kDynamic) {
      resultTypeShape.push_back(ShapedType::kDynamic);
      // There only remains to compute the 'high' padding Value.
      auto add = [&](Value a, Value b) {
        return rewriter.create<arith::AddIOp>(loc, a, b);
      };
      auto sub = [&](Value a, Value b) {
        return rewriter.create<arith::SubIOp>(loc, a, b);
      };
      auto rem = [&](Value a, Value b) {
        return rewriter.create<arith::RemSIOp>(loc, a, b);
      };
      // Compare to the plainer distanceToNextMultipleOf in the static
      // dimension case below.
      auto distanceToNextMultipleOf = [&](Value a, Value b) {
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value bMinusOne = sub(b, one);
        return sub(bMinusOne, rem(add(a, bMinusOne), b));
      };
      Value inputDim = rewriter.create<tensor::DimOp>(loc, input, i);
      Value tileDim =
          rewriter.create<arith::ConstantIndexOp>(loc, tileShape[i]);
      Value padding = distanceToNextMultipleOf(inputDim, tileDim);
      highPadding.push_back(padding);
    } else {
      auto distanceToNextMultipleOf = [=](int64_t a, int64_t b) {
        int64_t bMinusOne = b - 1;
        return bMinusOne - ((a + bMinusOne) % b);
      };
      int64_t inputDim = inputShape[i];
      int64_t tileDim = tileShape[i];
      int64_t padding = distanceToNextMultipleOf(inputDim, tileDim);
      resultTypeShape.push_back(inputDim + padding);
      highPadding.push_back(rewriter.getIndexAttr(padding));
    }
  }
  Type elementType = inputType.getElementType();
  RankedTensorType resultType =
      RankedTensorType::get(resultTypeShape, elementType);
  Value padValue;
  if (auto complexTy = elementType.dyn_cast<ComplexType>()) {
    auto zero = rewriter.getZeroAttr(complexTy.getElementType());
    padValue = rewriter.create<complex::ConstantOp>(
        loc, elementType, rewriter.getArrayAttr({zero, zero}));
  } else {
    auto zero = rewriter.getZeroAttr(elementType);
    padValue = rewriter.create<arith::ConstantOp>(loc, elementType, zero);
  }
  return rewriter.create<tensor::PadOp>(loc, resultType, input, lowPadding,
                                        highPadding, padValue);
}

// Returns a top-left slice from |input| shaped like |likeWhat|.
Value extractSliceLike(Location loc, PatternRewriter &rewriter, Value input,
                       Value likeWhat) {
  SmallVector<OpFoldResult, 2> offsets, dims, strides;
  auto resultType = likeWhat.getType().cast<RankedTensorType>();
  int64_t rank = resultType.getRank();
  auto resultShape = likeWhat.getType().cast<ShapedType>().getShape();
  for (int i = 0; i < rank; ++i) {
    offsets.push_back(rewriter.getIndexAttr(0));
    strides.push_back(rewriter.getIndexAttr(1));
    if (resultShape[i] == ShapedType::kDynamic) {
      dims.emplace_back(rewriter.create<tensor::DimOp>(loc, likeWhat, i));
    } else {
      dims.push_back(rewriter.getIndexAttr(resultShape[i]));
    }
  }
  return rewriter.create<tensor::ExtractSliceOp>(loc, resultType, input,
                                                 offsets, dims, strides);
}

bool haveEqualShapeDim(Value x, Value y, int i) {
  return x.getType().cast<ShapedType>().getDimSize(i) ==
         y.getType().cast<ShapedType>().getDimSize(i);
}

// Pattern to convert linalg.matmul to an equivalent subgraph using
// linalg.mmt4d. Currently, m0, n0 and k0 (packing parameters, aka layout tiling
// parameters) are compile-time constants.
struct MatmulToMmt4dPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  explicit MatmulToMmt4dPattern(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    Value lhs = matmulOp.getDpsInputOperand(0)->get();
    Value rhs = matmulOp.getDpsInputOperand(1)->get();
    Value acc = matmulOp.getDpsInitOperand(0)->get();

    // This transformation supports any mixing of static and dynamic dimensions,
    // with one exception: the dynamic-ness of each dimension of the accumulator
    // must match the dynamic-ness of the corresponding lhs/rhs dimension.
    // This limitation is not inherent to this transformation's code, it's just
    // here to avoid a current linalg folding limitation: at the moment,
    // removing this gives the following error in e2e matmul tests,
    //   "error: failed to legalize operation 'tensor.cast' that was explicitly
    //   marked illegal"
    // apparently due to some missing folding of tensor.cast op into reshapes.
    if (!haveEqualShapeDim(lhs, acc, 0) || !haveEqualShapeDim(rhs, acc, 1)) {
      return failure();
    }

    ShapedType lhsType = lhs.getType().cast<ShapedType>();
    ShapedType rhsType = rhs.getType().cast<ShapedType>();
    int64_t shapeM = lhsType.getShape()[0];
    int64_t shapeN = rhsType.getShape()[1];
    auto chooseMatMulOrMatVec = [=](ArrayRef<int> m0k0n0,
                                    ArrayRef<int> m0k0n0ForMatVec,
                                    ArrayRef<int> m0k0n0ForWhenRhsHas2Columns,
                                    std::string comment) {
      assert(m0k0n0ForMatVec[2] == 1 && "not a matrix*vector shape");
      assert(m0k0n0ForWhenRhsHas2Columns[2] == 2 &&
             "N=2 is expected when RHS has 2 columns");

      SmallVector<int> params;
      if (shapeN == 1 || shapeM == 1) {
        params.assign(m0k0n0ForMatVec.begin(), m0k0n0ForMatVec.end());
      } else if (shapeN == 2 || shapeM == 2) {
        params.assign(m0k0n0ForWhenRhsHas2Columns.begin(),
                      m0k0n0ForWhenRhsHas2Columns.end());
      } else {
        return Mmt4DTileParams(m0k0n0, comment);
      }

      if (shapeN == 1 || shapeN == 2) {
        comment += ", matrix * narrow matrix, where the narrow matrix has " +
                   std::to_string(shapeN) + " column(s)";
      } else {
        // The vector*matrix case is intentionally derived from the
        // matrix*vector case by swapping M and N dims so that in kernel
        // codegen we can reuse matrix*vector kernels by swapping LHS and RHS.
        std::swap(params[0], params[2]);
        comment += ", narrow matrix * matrix, where the narrow matrix has " +
                   std::to_string(shapeM) + " column(s)";
      }
      return Mmt4DTileParams(params, comment);
    };

    const auto &tileParams = chooseMatMulOrMatVec(
        {8, 1, 8}, {8, 1, 1}, {8, 1, 2}, "f32*f32->f32, generic");

    Value paddedLhs = pad(loc, rewriter, lhs, tileParams.lhs());
    Value paddedRhs = pad(loc, rewriter, rhs, tileParams.rhs());
    Value paddedAcc = pad(loc, rewriter, acc, tileParams.acc());

    Value packed4DLhs =
        pack(loc, rewriter, paddedLhs, {0, 1}, tileParams.lhs(), {});
    Value packed4DRhs = pack(loc, rewriter, paddedRhs, {1, 0},
                             tileParams.rhsTranspose(), {1, 0});
    Value packed4DAcc =
        pack(loc, rewriter, paddedAcc, {0, 1}, tileParams.acc(), {});

    auto mmt4d = rewriter.create<linalg::Mmt4DOp>(
        loc, packed4DAcc.getType(), ValueRange{packed4DLhs, packed4DRhs},
        ValueRange{packed4DAcc});
    mmt4d->setAttr(StringAttr::get(getContext(), "comment"),
                   StringAttr::get(getContext(), tileParams.getComment()));

    Value paddedResult = unpack(loc, rewriter, mmt4d.getResult(0), paddedAcc,
                                {0, 1}, tileParams.acc(), {});

    Value result = extractSliceLike(loc, rewriter, paddedResult, acc);
    rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});

    return success();
  }
};

FailureOr<TilingResult> tileMatmul(PatternRewriter &rewriter, Operation *op,
                                   ArrayRef<int64_t> tileSizes) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  return tileUsingGmlSt(opts, rewriter, cast<TilingInterface>(op));
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
void splitParallelAndReductionTiles(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &parallelSizes,
                                    SmallVectorImpl<int64_t> &reductionSizes) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::parallel) {
      reductionSizes[index] = 0;
    } else {
      parallelSizes[index] = 0;
    }
  }
}

FailureOr<Operation *> tileUsingSCFForAndReplace(
    PatternRewriter &rewriter, Operation *op,
    const scf::SCFTilingOptions &tilingOptions) {
  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, tilingOptions);
  if (failed(tilingResult) || tilingResult->loops.empty()) return failure();
  rewriter.replaceOp(op, tilingResult->replacements);
  return tilingResult->tiledOps.front();
}

/// Pattern to tile `linalg.mmt4d`.
struct Mmt4DTransformPattern : public OpRewritePattern<linalg::Mmt4DOp> {
  using OpRewritePattern<linalg::Mmt4DOp>::OpRewritePattern;

  explicit Mmt4DTransformPattern(MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Mmt4DOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::Mmt4DOp mmt4dOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(mmt4dOp, kMatmulTransformedLabel)) {
      return rewriter.notifyMatchFailure(mmt4dOp,
                                         "has already been transformed.");
    }

    // Tile tensor.pack ops.
    auto packTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [&](OpBuilder b, Operation *op) {
              auto numLoops =
                  cast<mlir::TilingInterface>(op).getLoopIteratorTypes().size();
              SmallVector<Value> tiles(
                  numLoops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
              return tiles;
            });

    auto *lhsOp = mmt4dOp.getInputs()[0].getDefiningOp();
    if (failed(tileUsingSCFForAndReplace(rewriter, lhsOp, packTilingOptions)))
      return failure();

    auto *rhsOp = mmt4dOp.getInputs()[1].getDefiningOp();
    if (failed(tileUsingSCFForAndReplace(rewriter, rhsOp, packTilingOptions)))
      return failure();

    auto *accOp = mmt4dOp.getOutputs()[0].getDefiningOp();
    if (failed(tileUsingSCFForAndReplace(rewriter, accOp, packTilingOptions)))
      return failure();

    // Tile tensor.unpack op.
    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              Location loc = op->getLoc();
              auto unpackOp = cast<tensor::UnPackOp>(op);
              auto numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<Value> tileSizes;
              for (size_t i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(getValueOrCreateConstantIndexOp(
                      builder, loc, dimAndTileMapping[i]));
                } else {
                  tileSizes.push_back(
                      getDimValue(builder, loc, unpackOp.getDest(), i));
                }
              }
              return tileSizes;
            });

    auto *unpackOp = *mmt4dOp->user_begin();
    if (failed(
            tileUsingSCFForAndReplace(rewriter, unpackOp, unpackTilingOptions)))
      return failure();

    // Compute the tile sizes. Note that at this stage we only do layout tiling.
    // Later we might also want to do traversal tiling (only on M and N dims).
    auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
      auto lhsShape =
          mmt4dOp.getInputs()[0].getType().cast<ShapedType>().getShape();
      auto rhsShape =
          mmt4dOp.getInputs()[1].getType().cast<ShapedType>().getShape();
      int64_t m0 = lhsShape[2];
      int64_t n0 = rhsShape[2];
      int64_t k0 = lhsShape[3];
      return {1, 1, 1, m0, n0, k0};
    };

    SmallVector<int64_t> parallelTileSizes = getL1TileSizes();
    SmallVector<int64_t> reductionTileSizes;

    // Search the number of outer parallel loops to separate them from possible
    // inner reduction dimensions.
    auto iterTypes = mmt4dOp.getIteratorTypesArray();
    // Make sure to only look at the leading loops for tiling---we will scan
    // this array to find the first non-parallel loop later and use that for
    // indexing into the tile sizes.
    if (iterTypes.size() > parallelTileSizes.size()) {
      iterTypes.resize(parallelTileSizes.size());
    }

    splitParallelAndReductionTiles(mmt4dOp.getOperation(), parallelTileSizes,
                                   reductionTileSizes);

    // Tile the parallel loops.
    auto tiledOp = tileUsingSCFForAndReplace(
        rewriter, mmt4dOp.getOperation(),
        scf::SCFTilingOptions().setTileSizes(parallelTileSizes));
    if (failed(tiledOp)) return failure();
    mmt4dOp = cast<linalg::Mmt4DOp>(*tiledOp);

    // Tile the reduction loops.
    tiledOp = tileUsingSCFForAndReplace(
        rewriter, mmt4dOp.getOperation(),
        scf::SCFTilingOptions().setTileSizes(reductionTileSizes));
    if (failed(tiledOp)) return failure();
    mmt4dOp = cast<linalg::Mmt4DOp>(*tiledOp);

    setLabel(mmt4dOp, kMatmulTransformedLabel);
    return success();
  }
};

/// Pattern to tile `linalg.matmul`, fuse `linalg.fill` into generated
/// `gml_st.parallel`, and peel the generated loops.
struct MatmulTransformPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  MatmulTransformPattern(MLIRContext *context,
                         MatmulTileSizeComputationFn tileSizeFn,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        tileSizeFn(std::move(tileSizeFn)) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(matmulOp, kMatmulTransformedLabel))
      return rewriter.notifyMatchFailure(matmulOp,
                                         "has already been transformed.");
    if (isa<scf::ForallOp, scf::ForOp>(matmulOp->getParentOp()))
      return rewriter.notifyMatchFailure(
          matmulOp, "has already been tiled by another pass.");

    auto cluster = findMapFusionCluster(matmulOp);
    auto fusionCluster = cluster.operations;
    auto *tilingRoot = cluster.root;

    auto lhsTy = matmulOp.getOperandTypes()[0].cast<ShapedType>();
    auto resultTy = matmulOp.getResultTypes()[0].cast<ShapedType>();

    auto tileSize = tileSizeFn(
        {resultTy.getDimSize(0), resultTy.getDimSize(1), lhsTy.getDimSize(1)});

    // Tiling of linalg.map requires two dimensions, linalg.matmul requires
    // three.
    SmallVector<int64_t> parallelDimsTileSizes{tileSize.m, tileSize.n};
    if (isa<linalg::MatmulOp>(tilingRoot)) parallelDimsTileSizes.push_back(0);

    // First level tiling: parallel dimensions.
    auto tilingParallelDimsResult =
        tileMatmul(rewriter, tilingRoot, parallelDimsTileSizes);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(tilingRoot,
                         tilingParallelDimsResult->loop->getResults());
      tilingRoot = tilingParallelDimsResult->tiledOps.front();

      // Fuse ops into the loop.
      fuseGreedily(rewriter, *tilingRoot->getBlock(),
                   [&](Operation *op) { return fusionCluster.contains(op); });
      (void)fuseFillOpsIntoForallOp(rewriter, tilingParallelDimsResult->loop);
    }

    // Second level tiling: reduction dimension for matmuls.
    SmallVector<scf::SCFTilingResult> tilingReductionDimsResults;
    for (auto op :
         llvm::to_vector(tilingRoot->getBlock()->getOps<linalg::MatmulOp>())) {
      auto result = tileMatmulReductionDims(rewriter, op, tileSize);
      if (failed(result)) return failure();
      tilingReductionDimsResults.push_back(*result);
    }

    // Peel parallel loops.
    //
    // We only want to peel (1) the parallel loop then (2) our kernel.
    auto peelingResult = peelAllLoops(tilingParallelDimsResult->loop, rewriter);

    // Peel reduction loop inside the main parallel loop, label the main loop as
    // "perfectly tiled" one, to enable vectorization after canonicalization.
    for (auto &res : tilingReductionDimsResults) {
      if (res.loops.size() == 1) {
        auto peelingResult = peelSCFForOp(rewriter, res.loops.front());
        setLabel(peelingResult.mainLoop, kPerfectlyTiledLoopLabel);
      }
    }
    return success();
  }

 private:
  FailureOr<scf::SCFTilingResult> tileMatmulReductionDims(
      PatternRewriter &rewriter, linalg::MatmulOp matmulOp,
      const MatmulSizes &tileSize) const {
    SmallVector<int64_t> reductionDimsTileSizes{0, 0, tileSize.k};
    scf::SCFTilingOptions opts;
    opts.setTileSizes(reductionDimsTileSizes);
    auto tilingReductionDimsResult =
        scf::tileUsingSCFForOp(rewriter, matmulOp.getOperation(), opts);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (!tilingReductionDimsResult->loops.empty()) {
      rewriter.replaceOp(matmulOp, tilingReductionDimsResult->replacements);
      matmulOp =
          cast<linalg::MatmulOp>(tilingReductionDimsResult->tiledOps.front());
    }

    setLabel(matmulOp, kMatmulTransformedLabel);
    return tilingReductionDimsResult;
  }

  MatmulTileSizeComputationFn tileSizeFn;
};

struct TransformMatmulForCpuPass
    : public impl::TransformMatmulForCpuPassBase<TransformMatmulForCpuPass> {
  TransformMatmulForCpuPass() = default;

  explicit TransformMatmulForCpuPass(MatmulTileSizeComputationFn tileSizeFn,
                                     bool lowerToMmt4DOp)
      : tileSizeFn(tileSizeFn ? std::move(tileSizeFn)
                              : [](MatmulSizes) -> MatmulSizes {
          return {4, 4, 4};
        }) {
    lowerToMmt4D = lowerToMmt4DOp;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Just do tiling and fusion on linalg.matmul.
    if (!lowerToMmt4D) {
      RewritePatternSet patterns(ctx);
      patterns.add<MatmulTransformPattern>(ctx, tileSizeFn);
      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
      // Ensure we drop the marker in the end.
      f.walk([](linalg::MatmulOp op) {
        removeLabel(op, kMatmulTransformedLabel);
      });
      return;
    }

    // Lower linalg.matmul to linalg.mmt4d (packed matmul).
    {
      // Convert linalg.matmul to linalg.mmt4d.
      RewritePatternSet patterns(ctx);
      patterns.add<MatmulToMmt4dPattern>(ctx);

      // Canonicalization.
      tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
      tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
      linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
      // Ensure we drop the marker in the end.
      f.walk([](Operation *op) {
        if (isa<linalg::MatmulOp>(op) || isa<linalg::Mmt4DOp>(op))
          removeLabel(op, kMatmulTransformedLabel);
      });
    }
    // Tiling pack, unpack and mmt4d ops.
    {
      RewritePatternSet patterns(ctx);
      // We tile towards SIMD codegen, so the tile sizes depend on the target
      // architecture (vector instruction sizes, etc.). Luckily, this
      // information is already captured in linalg.mmt4d during linalg.matmul ->
      // linalg.mmt4d lowering phase. It is hardcoded for AVX on x86 for now.
      patterns.add<Mmt4DTransformPattern>(ctx);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
      // Ensure we drop the marker in the end.
      f.walk(
          [](linalg::Mmt4DOp op) { removeLabel(op, kMatmulTransformedLabel); });
    }
    // Expanding pack and unpack ops to other primitive tensor/linalg ops and
    // canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern>(ctx);
      patterns.add<linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }

 private:
  MatmulTileSizeComputationFn tileSizeFn;
};

/// Remove memref::CopyOp whose target (can be either a memref::SubViewOp or
/// memref::AllocOp) has no other users.
struct SimplifyDeadCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto valueIt = op.getTarget();
    Operation *onlyNonStoreLikeUser = op;
    for (auto subviewOp = valueIt.getDefiningOp<memref::SubViewOp>(); subviewOp;
         onlyNonStoreLikeUser = subviewOp, valueIt = subviewOp.getSource(),
              subviewOp = valueIt.getDefiningOp<memref::SubViewOp>()) {
      // TODO(vuson) simplify if other uses are also memref.copy writing to
      // subview
      //    %alloc_4 = memref.alloc()
      //    %subview_5 = memref.subview %alloc_4
      //    %subview_6 = memref.subview %alloc_4
      //    memref.copy %arg0, %subview_6
      //    memref.copy %arg1, %subview_5
      if (!subviewOp->hasOneUse()) return failure();
    }

    auto hasOnlyStoreLikeUsers = [&](Value alloc) {
      return !llvm::any_of(alloc.getUsers(), [&](Operation *op) {
        if (op == onlyNonStoreLikeUser) return false;
        // TODO(vuson) remove this exception when MemoryEffectOpInterface gets
        // corrected for linalg::FillOp. Right now it has MemoryEffects::Read
        // while the only thing it ever reads is metadata such as dynamic sizes.
        if (isa<linalg::FillOp>(op)) return false;
        if (auto effect = dyn_cast<MemoryEffectOpInterface>(op)) {
          return effect.getEffectOnValue<MemoryEffects::Read>(alloc)
                     .has_value() ||
                 !effect.getEffectOnValue<MemoryEffects::Write>(alloc)
                      .has_value();
        }
        return true;
      });
    };
    if (!valueIt.getDefiningOp<memref::AllocOp>() ||
        !hasOnlyStoreLikeUsers(valueIt))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct SimplifyDeadCopyPass
    : public impl::SimplifyDeadCopyPassBase<SimplifyDeadCopyPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<SimplifyDeadCopyPattern>(ctx);
    memref::AllocOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass(MatmulTileSizeComputationFn tileSizeFn,
                                bool lowerToMmt4DOp) {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>(
      std::move(tileSizeFn), lowerToMmt4DOp);
}

std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyDeadCopyPass() {
  return std::make_unique<SimplifyDeadCopyPass>();
}

}  // namespace mlir::gml_st
