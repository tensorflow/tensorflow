/* Copyright 2019 The OpenXLA Authors.

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

// This file implements logic for lowering MHLO general dot to a regular dot.

#include <sys/types.h>

#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEGENERALDOTPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

Value transposeReshape(Value arg, Location loc,
                       llvm::ArrayRef<int64_t> leftDims,
                       llvm::ArrayRef<int64_t> rightDims,
                       llvm::ArrayRef<int64_t> argShape,
                       PatternRewriter &rewriter) {
  auto elementType = getElementTypeOrSelf(arg.getType());

  int64_t leftSize = 1;
  for (auto dim : leftDims) {
    leftSize = (ShapedType::isDynamic(argShape[dim]) || leftSize < 0)
                   ? ShapedType::kDynamic
                   : leftSize * argShape[dim];
  }

  int64_t rightSize = 1;
  for (auto dim : rightDims) {
    rightSize = (ShapedType::isDynamic(argShape[dim]) || rightSize < 0)
                    ? ShapedType::kDynamic
                    : rightSize * argShape[dim];
  }

  // Generate the transpose permutation attribute.
  llvm::SmallVector<int64_t, 5> transposePermutation(leftDims.begin(),
                                                     leftDims.end());
  transposePermutation.append(rightDims.begin(), rightDims.end());

  TensorType transposePermutationType =
      RankedTensorType::get({static_cast<int64_t>(transposePermutation.size())},
                            rewriter.getIntegerType(64));

  auto transposePermutationAttr =
      DenseIntElementsAttr::get(transposePermutationType,
                                llvm::ArrayRef(transposePermutation))
          .cast<DenseIntElementsAttr>();

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 5> transposedShape;
  for (auto val : transposePermutation) {
    transposedShape.push_back(argShape[val]);
  }

  // If there are only a single pair of contracting dimensions and the output
  // rank is two we can skip a needless reshape.
  bool noReshape = transposedShape.size() == 2 && leftDims.size() == 1 &&
                   rightDims.size() == 1;

  // Construct transpose. If no reshape is needed, we are done.
  auto transposeType = RankedTensorType::get(transposedShape, elementType);
  Value transposeResult = rewriter.create<TransposeOp>(
      loc, transposeType, arg, transposePermutationAttr);
  if (noReshape) return transposeResult;

  // Return the final result.
  auto reshapedType = RankedTensorType::get({leftSize, rightSize}, elementType);

  if (reshapedType.hasStaticShape()) {
    return rewriter.create<ReshapeOp>(loc, reshapedType, transposeResult);
  }

  SmallVector<Value> reshapeDims;
  auto multiplyDynamicDims = [&](llvm::ArrayRef<int64_t> dims) -> Value {
    Value dynamicSize = rewriter.create<GetDimensionSizeOp>(
        loc, arg, rewriter.getI64IntegerAttr(dims.front()));
    Value dynamicSizeReshaped = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynamicSize);
    for (auto idx : dims.drop_front()) {
      Value dim = rewriter.create<GetDimensionSizeOp>(
          loc, arg, rewriter.getI64IntegerAttr(idx));
      Value dimReshaped = rewriter.create<ReshapeOp>(
          loc, RankedTensorType::get({1}, rewriter.getI32Type()), dim);
      dynamicSizeReshaped =
          rewriter.create<MulOp>(loc, dynamicSizeReshaped, dimReshaped);
    }
    return dynamicSizeReshaped;
  };

  if (leftSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(leftDims));
  } else {
    reshapeDims.push_back(
        rewriter.create<ConstantOp>(loc, rewriter.getI32TensorAttr(leftSize)));
  }

  if (rightSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(rightDims));
  } else {
    reshapeDims.push_back(
        rewriter.create<ConstantOp>(loc, rewriter.getI32TensorAttr(rightSize)));
  }

  Value reshapeDimsTensor = rewriter.create<ConcatenateOp>(
      loc, RankedTensorType::get({2}, rewriter.getI32Type()), reshapeDims,
      rewriter.getI64IntegerAttr(0));
  return rewriter.create<DynamicReshapeOp>(loc, reshapedType, transposeResult,
                                           reshapeDimsTensor);
}

Value processDotArg(Value arg, Location loc, ArrayRef<int64_t> contractDimsAttr,
                    bool outerDimsFirst, PatternRewriter &rewriter) {
  auto shape = arg.getType().cast<ShapedType>().getShape();

  llvm::SmallVector<bool, 5> isOuterDim;
  isOuterDim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contractDims;
  for (auto dim : contractDimsAttr) {
    contractDims.push_back(dim);
    isOuterDim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outerDims;
  for (const auto &it : llvm::enumerate(isOuterDim)) {
    if (it.value()) {
      outerDims.push_back(it.index());
    }
  }

  if (outerDimsFirst) {
    return transposeReshape(arg, loc, outerDims, contractDims, shape, rewriter);
  }

  return transposeReshape(arg, loc, contractDims, outerDims, shape, rewriter);
}

struct GeneralDotConvert : public OpRewritePattern<DotGeneralOp> {
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.

  explicit GeneralDotConvert(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dotNumbers = op.getDotDimensionNumbers();
    if (!dotNumbers.getLhsBatchingDimensions().empty() ||
        !dotNumbers.getRhsBatchingDimensions().empty()) {
      return failure();
    }

    ArrayAttr precisionConfig;
    auto opPrecisionConfig = op.getPrecisionConfig();
    if (opPrecisionConfig.has_value()) precisionConfig = *opPrecisionConfig;

    auto resultTy = op.getType().cast<ShapedType>();

    auto lhsContractingDims = dotNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dotNumbers.getRhsContractingDimensions();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    RankedTensorType lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return failure();

    // The MHLO dot operator directly supports a vector dot product
    // (two vectors reduce into a scalar) as well as a matrix vector
    // product (a matrix and vector reduce into a vector) without any
    // need for reshaping. We handle those special cases first, before
    // entering the general logic that reduces into a matrix.
    if (lhsTy.hasStaticShape() && rhsTy.hasStaticShape() &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
      if (lhsTy.getRank() == 1 && rhsTy.getRank() == 1) {
        // Vector-vector, reduces into scalar.
        assert(lhsContractingDims[0] == 0 && rhsContractingDims[0] == 0);
        ShapedType newTy = RankedTensorType::get({}, resultTy.getElementType());
        rewriter.replaceOpWithNewOp<DotOp>(op, newTy, lhs, rhs,
                                           precisionConfig);
        return success();
      }
      if (lhsTy.getRank() == 2 && rhsTy.getRank() == 1 &&
          lhsContractingDims[0] == 1) {
        // Matrix-vector, reduces into vector.
        assert(rhsContractingDims[0] == 0);
        ShapedType newTy = RankedTensorType::get({lhsTy.getShape()[0]},
                                                 resultTy.getElementType());
        rewriter.replaceOpWithNewOp<DotOp>(op, newTy, lhs, rhs,
                                           precisionConfig);
        return success();
      }
      if (lhsTy.getRank() == 2 && rhsTy.getRank() == 2 &&
          lhsContractingDims[0] == 1 && rhsContractingDims[0] == 0) {
        // Matrix-matrix, reduces into matrix. Note that for dense cases, this
        // rewriting rule simply provides a shortcut for what is to follow
        // (modulo optimizing the trivial transpose/reshape operations). For
        // sparse cases, however, this rewriting preserves the output sparsity
        // that was explicitly given for the general dot operation.
        Value newDotOp =
            rewriter.create<DotOp>(loc, resultTy, lhs, rhs, precisionConfig);
        if (auto enc = sparse_tensor::getSparseTensorEncoding(resultTy)) {
          newDotOp.setType(RankedTensorType::get(
              resultTy.getShape(), resultTy.getElementType(), enc));
        }
        rewriter.replaceOp(op, newDotOp);
        return success();
      }
    }

    // For any sparse situation, don't use any of the following rules, since
    // transposing and reshaping is not without cost. Instead, rely on the
    // default linalg lowering that follows later in the pipeline.
    if (sparse_tensor::hasAnySparseOperandOrResult(op)) return failure();

    // Compute the, possibly, transposed-reshaped operands.
    lhs = llvm::cast<mlir::TypedValue<mlir::RankedTensorType>>(processDotArg(
        lhs, loc, lhsContractingDims, /*outerDimsFirst=*/true, rewriter));
    rhs = llvm::cast<mlir::TypedValue<mlir::RankedTensorType>>(processDotArg(
        rhs, loc, rhsContractingDims, /*outerDimsFirst=*/false, rewriter));

    // Accept only static shaped types.
    auto lhsShapeType = lhs.getType().dyn_cast_or_null<ShapedType>();
    auto rhsShapeType = rhs.getType().dyn_cast_or_null<ShapedType>();
    if (!lhsShapeType || !rhsShapeType) return failure();

    // Generate new dot operator on expanded types.
    ShapedType newTy = RankedTensorType::get(
        {lhsShapeType.getShape()[0], rhsShapeType.getShape()[1]},
        resultTy.getElementType());
    Value newDotOp =
        rewriter.create<DotOp>(loc, newTy, lhs, rhs, precisionConfig);
    if (static_cast<int64_t>(lhsContractingDims.size()) ==
            lhsTy.getRank() - 1 &&
        static_cast<int64_t>(rhsContractingDims.size()) ==
            rhsTy.getRank() - 1) {
      rewriter.replaceOp(op, newDotOp);
      return success();
    }

    // We can avoid all the computation below if we know the static shape.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, resultTy, newDotOp);
      return success();
    }

    llvm::SmallVector<int64_t> staticDims;
    llvm::SmallVector<Value> dynDims;

    auto getDynamicDims = [&](Value arg,
                              llvm::ArrayRef<int64_t> contractingDims) {
      RankedTensorType ty = arg.getType().cast<RankedTensorType>();
      int index = 0;
      for (auto contractingDim : contractingDims) {
        for (; index < contractingDim; index++) {
          staticDims.push_back(ty.getDimSize(index));
          Value dynDim = rewriter.create<GetDimensionSizeOp>(
              loc, arg, rewriter.getI64IntegerAttr(index));
          Value dynDimReshaped = rewriter.create<ReshapeOp>(
              loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynDim);
          dynDims.push_back(dynDimReshaped);
        }
        index++;
      }

      for (; index < ty.getRank(); index++) {
        staticDims.push_back(ty.getDimSize(index));
        Value dynDim = rewriter.create<GetDimensionSizeOp>(
            loc, arg, rewriter.getI64IntegerAttr(index));
        Value dynDimReshaped = rewriter.create<ReshapeOp>(
            loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynDim);
        dynDims.push_back(dynDimReshaped);
      }
    };

    getDynamicDims(op.getLhs(), lhsContractingDims);
    getDynamicDims(op.getRhs(), rhsContractingDims);

    Value reshapeDimsTensor = rewriter.create<ConcatenateOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(dynDims.size())},
                              rewriter.getI32Type()),
        dynDims, rewriter.getI64IntegerAttr(0));

    Value result = rewriter.create<DynamicReshapeOp>(
        loc, RankedTensorType::get(staticDims, resultTy.getElementType()),
        newDotOp, reshapeDimsTensor);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LegalizeGeneralDotPass
    : public impl::LegalizeGeneralDotPassBase<LegalizeGeneralDotPass> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGeneralDotOpLoweringPatterns(&patterns, &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

void mlir::mhlo::populateGeneralDotOpLoweringPatterns(
    RewritePatternSet *patterns, MLIRContext *ctx) {
  patterns->add<GeneralDotConvert>(ctx);
}

std::unique_ptr<::mlir::Pass> mlir::mhlo::createLegalizeGeneralDotPass() {
  return std::make_unique<LegalizeGeneralDotPass>();
}
