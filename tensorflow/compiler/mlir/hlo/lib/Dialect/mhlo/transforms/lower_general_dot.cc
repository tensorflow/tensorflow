/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
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
                   ? ShapedType::kDynamicSize
                   : leftSize * argShape[dim];
  }

  int64_t rightSize = 1;
  for (auto dim : rightDims) {
    rightSize = (ShapedType::isDynamic(argShape[dim]) || rightSize < 0)
                    ? ShapedType::kDynamicSize
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
                                llvm::makeArrayRef(transposePermutation))
          .cast<DenseIntElementsAttr>();

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 5> transposedShape;
  for (auto val : transposePermutation) {
    transposedShape.push_back(argShape[val]);
  }
  auto transposeType = RankedTensorType::get(transposedShape, elementType);
  Value transposeResult = rewriter.create<TransposeOp>(
      loc, transposeType, arg, transposePermutationAttr);

  // If there are only a single pair of contracting dimensions and the output
  // rank is two we can skip a needless reshape.
  if (transposeType.getRank() == 2 && leftDims.size() == 1 &&
      rightDims.size() == 1)
    return transposeResult;

  // Return the final result.
  auto reshapedType = RankedTensorType::get({leftSize, rightSize}, elementType);

  if (reshapedType.hasStaticShape()) {
    return rewriter.create<mhlo::ReshapeOp>(loc, reshapedType, transposeResult);
  }

  SmallVector<Value> reshapeDims;
  auto multiplyDynamicDims = [&](llvm::ArrayRef<int64_t> dims) -> Value {
    Value dynamicSize = rewriter.create<mhlo::GetDimensionSizeOp>(
        loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
        rewriter.getI64IntegerAttr(dims.front()));

    for (auto idx : dims.drop_front()) {
      Value dim = rewriter.create<mhlo::GetDimensionSizeOp>(
          loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
          rewriter.getI64IntegerAttr(idx));
      dynamicSize = rewriter.create<mhlo::MulOp>(loc, dynamicSize, dim);
    }
    return dynamicSize;
  };

  if (leftSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(leftDims));
  } else {
    reshapeDims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(leftSize)));
  }

  if (rightSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(rightDims));
  } else {
    reshapeDims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(rightSize)));
  }

  Value reshapeDimsTensor = rewriter.create<mhlo::ConcatenateOp>(
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

    auto dotNumbers = op.dot_dimension_numbers();
    if (!dotNumbers.getLhsBatchingDimensions().empty() ||
        !dotNumbers.getRhsBatchingDimensions().empty()) {
      return failure();
    }

    auto lhsContractingDims = dotNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dotNumbers.getRhsContractingDimensions();

    auto lhs = op.lhs();
    auto rhs = op.rhs();

    RankedTensorType lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsTy || !rhsTy) return failure();

    lhs = processDotArg(op.lhs(), op.getLoc(),
                        dotNumbers.getLhsContractingDimensions(),
                        /*outer_dims_first=*/true, rewriter);

    rhs = processDotArg(op.rhs(), op.getLoc(),
                        dotNumbers.getRhsContractingDimensions(),
                        /*outer_dims_first=*/false, rewriter);

    // Accept only static shaped types.
    auto lhsShapeType = lhs.getType().dyn_cast_or_null<ShapedType>();
    auto rhsShapeType = rhs.getType().dyn_cast_or_null<ShapedType>();
    if (!lhsShapeType || !rhsShapeType) return failure();

    ArrayAttr precisionConfig;
    if (op.precision_config()) precisionConfig = *op.precision_config();
    Value newDotOp =
        rewriter.create<DotOp>(op.getLoc(), lhs, rhs, precisionConfig);
    if (lhsContractingDims.size() == (lhsTy.getRank() - 1) &&
        rhsContractingDims.size() == (rhsTy.getRank() - 1)) {
      rewriter.replaceOp(op, newDotOp);
      return success();
    }

    ShapedType resultTy = op.getType().cast<ShapedType>();

    // We can avoid all the computation below if we know the static shape.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, resultTy, newDotOp);
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
          dynDims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
              loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
              rewriter.getI64IntegerAttr(index)));
        }
        index++;
      }

      for (; index < ty.getRank(); index++) {
        staticDims.push_back(ty.getDimSize(index));
        dynDims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
            loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
            rewriter.getI64IntegerAttr(index)));
      }
    };

    getDynamicDims(op.lhs(), lhsContractingDims);
    getDynamicDims(op.rhs(), rhsContractingDims);

    Value reshapeDimsTensor = rewriter.create<mhlo::ConcatenateOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(dynDims.size())},
                              rewriter.getI32Type()),
        dynDims, rewriter.getI64IntegerAttr(0));

    Value result = rewriter.create<DynamicReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(staticDims, resultTy.getElementType()), newDotOp,
        reshapeDimsTensor);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LegalizeGeneralDotPass
    : public LegalizeGeneralDotPassBase<LegalizeGeneralDotPass> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    PopulateGeneralDotOpLoweringPatterns(&patterns, &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

void mlir::mhlo::PopulateGeneralDotOpLoweringPatterns(
    RewritePatternSet *patterns, MLIRContext *ctx) {
  patterns->add<GeneralDotConvert>(ctx);
}

std::unique_ptr<::mlir::Pass> mlir::mhlo::createLegalizeGeneralDotPass() {
  return std::make_unique<LegalizeGeneralDotPass>();
}
