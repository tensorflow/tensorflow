/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_PACKMATMULPASS
#include "gml_st/transforms/passes.h.inc"

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
      tensor::createDimValues(rewriter, loc, resultShapeValue);
  auto empty = rewriter.create<tensor::EmptyOp>(
      loc, resultDims,
      source.getType().cast<RankedTensorType>().getElementType());

  SmallVector<OpFoldResult> innerTileSizesOfr =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(innerTileSizes));

  return rewriter.create<tensor::UnPackOp>(loc, source, empty, innerDimsPos,
                                           innerTileSizesOfr, outerDimsPerm);
}

bool haveEqualShapeDim(Value x, Value y, int i) {
  return x.getType().cast<ShapedType>().getDimSize(i) ==
         y.getType().cast<ShapedType>().getDimSize(i);
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

// Pattern to convert linalg.matmul to an equivalent subgraph using
// linalg.mmt4d. Currently, m0, n0 and k0 (packing parameters, aka layout tiling
// parameters) are compile-time constants.
LogicalResult packMatmul(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) {
  Location loc = matmulOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

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
  auto chooseMatMulOrMatVec =
      [=](ArrayRef<int> m0k0n0, ArrayRef<int> m0k0n0ForMatVec,
          ArrayRef<int> m0k0n0ForWhenRhsHas2Columns, std::string comment) {
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

  const auto &tileParams = chooseMatMulOrMatVec({8, 1, 8}, {8, 1, 1}, {8, 1, 2},
                                                "f32*f32->f32, generic");

  Value paddedLhs = pad(loc, rewriter, lhs, tileParams.lhs());
  Value paddedRhs = pad(loc, rewriter, rhs, tileParams.rhs());
  Value paddedAcc = pad(loc, rewriter, acc, tileParams.acc());

  Value packed4DLhs =
      pack(loc, rewriter, paddedLhs, {0, 1}, tileParams.lhs(), {});
  Value packed4DRhs =
      pack(loc, rewriter, paddedRhs, {1, 0}, tileParams.rhsTranspose(), {1, 0});
  Value packed4DAcc =
      pack(loc, rewriter, paddedAcc, {0, 1}, tileParams.acc(), {});

  auto mmt4d = rewriter.create<linalg::Mmt4DOp>(
      loc, packed4DAcc.getType(), ValueRange{packed4DLhs, packed4DRhs},
      ValueRange{packed4DAcc});
  mmt4d->setAttr(StringAttr::get(ctx, "comment"),
                 StringAttr::get(ctx, tileParams.getComment()));

  Value paddedResult = unpack(loc, rewriter, mmt4d.getResult(0), paddedAcc,
                              {0, 1}, tileParams.acc(), {});

  Value result = extractSliceLike(loc, rewriter, paddedResult, acc);
  rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});

  return success();
}

struct PackMatmulPass : public impl::PackMatmulPassBase<PackMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add(packMatmul);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createPackMatmulPass() {
  return std::make_unique<PackMatmulPass>();
}

}  // namespace mlir::gml_st
