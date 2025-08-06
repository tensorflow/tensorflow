/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_CONVERTSHAPETOSTABLEHLOWITHCONSTRAINTSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {
using ::mlir::stablehlo::AndOp;
using ::mlir::stablehlo::CompareOp;
using ::mlir::stablehlo::ComparisonDirection;
using ::mlir::stablehlo::ConcatenateOp;
using ::mlir::stablehlo::ConstantOp;
using ::mlir::stablehlo::CustomCallOp;
using ::mlir::stablehlo::OrOp;
using ::mlir::stablehlo::ReshapeOp;
using ::mlir::stablehlo::SliceOp;

// Cast from index-based shape representation used in the Shape dialect to the
// i32-based representation used in HLO:
//   * index => tensor<i32>.
//   * tensor<Nxindex> => tensor<Nxi32>.
//   * All i32-based types from above => themselves.
// There is no convenient op that can express this, so we're using
// unrealized_conversion_cast (with the idea that all these casts will
// annihilate at the end of the pass).
Value castToI32(PatternRewriter& rewriter, Location loc, Value value) {
  Type resultType;
  if (value.getType().isIndex())
    resultType = RankedTensorType::get({}, rewriter.getI32Type());
  if (auto valueType = mlir::dyn_cast<ShapedType>(value.getType())) {
    if (!valueType.hasStaticShape()) return {};
    if (valueType.getElementType().isInteger(32)) return value;
    if (valueType.getElementType().isIndex())
      resultType =
          RankedTensorType::get(valueType.getShape(), rewriter.getI32Type());
  }
  if (!resultType) return {};
  auto cast =
      rewriter.create<UnrealizedConversionCastOp>(loc, resultType, value);
  return cast.getResult(0);
}

// Pads input tensor<N x i32> by X ones from the left. The number X is
// determined by input pad. Result is tensor<(X+N) x i32>, where the first X
// elements are ones.
Value padFromLeft(PatternRewriter& rewriter, Location loc, Value input,
                  int64_t pad) {
  Value padI32 = rewriter.create<ConstantOp>(
      loc, DenseIntElementsAttr::get<int32_t>(
               RankedTensorType::get({pad}, rewriter.getI32Type()), 1));
  return rewriter.create<ConcatenateOp>(loc, ValueRange{padI32, input},
                                        /*dimension=*/0);
}

void insertShapeAssertionCustomCall(OpBuilder builder, Location loc,
                                    Value assert) {
  auto customCall =
      builder.create<CustomCallOp>(loc, TypeRange{}, ValueRange{assert});
  customCall.setCallTargetName("shape_assertion");
  customCall.setHasSideEffect(true);
  customCall->setAttr("error_message",
                      builder.getStringAttr("Shape assertion failed"));
}

struct ConvertCstrBroadcastableOp
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter& rewriter) const override {
    // As defined, op inputs must be 1D tensor or !shape.shape.
    // We only support inputs of two 1D tensors.
    if (op.getShapes().size() != 2) return failure();
    auto shape1 = castToI32(rewriter, op.getLoc(), op.getShapes().front());
    auto shape2 = castToI32(rewriter, op.getLoc(), op.getShapes().back());
    if (!shape1 || !shape2) return failure();
    auto tensorType1 = mlir::dyn_cast<RankedTensorType>(shape1.getType());
    auto tensorType2 = mlir::dyn_cast<RankedTensorType>(shape2.getType());
    if (!tensorType1 || !tensorType2) return failure();

    // If the two operand shapes are of different sizes, the smaller one is
    // padded with 1's from the left.
    int32_t rank =
        std::max(tensorType1.getDimSize(0), tensorType2.getDimSize(0));
    if (tensorType1.getDimSize(0) < tensorType2.getDimSize(0)) {
      shape1 =
          padFromLeft(rewriter, op.getLoc(), shape1,
                      tensorType2.getDimSize(0) - tensorType1.getDimSize(0));
    } else if (tensorType1.getDimSize(0) > tensorType2.getDimSize(0)) {
      shape2 =
          padFromLeft(rewriter, op.getLoc(), shape2,
                      tensorType1.getDimSize(0) - tensorType2.getDimSize(0));
    }

    // Compute if each dim is broadcastable. A dim is broadcastable iff
    // dimSize1 == dimSize2 or dimSize1 == 1 or dimSize2 == 1
    auto allOne = rewriter.create<ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get<int32_t>(
                         RankedTensorType::get({rank}, rewriter.getI32Type()),
                         static_cast<int32_t>(1)));
    Value dimSize1Is1 = rewriter.create<CompareOp>(op.getLoc(), shape1, allOne,
                                                   ComparisonDirection::EQ);
    Value dimSize2Is1 = rewriter.create<CompareOp>(op.getLoc(), shape2, allOne,
                                                   ComparisonDirection::EQ);
    Value eitherDimSizeIs1 =
        rewriter.create<OrOp>(op.getLoc(), dimSize1Is1, dimSize2Is1);
    Value dimSizeEq = rewriter.create<CompareOp>(op.getLoc(), shape1, shape2,
                                                 ComparisonDirection::EQ);
    Value dimBroadcastable =
        rewriter.create<OrOp>(op.getLoc(), eitherDimSizeIs1, dimSizeEq);

    // Iterate over each dim to check that all dims are broadcastable.
    auto boolType = RankedTensorType::get({1}, rewriter.getI1Type());
    Value allBroadcastable = rewriter.create<ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get<bool>(boolType, true));
    for (auto i = 0; i < rank; ++i) {
      Value broadcastable = rewriter.create<SliceOp>(
          op.getLoc(), dimBroadcastable, rewriter.getDenseI64ArrayAttr(i),
          rewriter.getDenseI64ArrayAttr(i + 1),
          rewriter.getDenseI64ArrayAttr(1));
      allBroadcastable =
          rewriter.create<AndOp>(op.getLoc(), allBroadcastable, broadcastable);
    }
    Value allBroadcastableScalar = rewriter.create<ReshapeOp>(
        op.getLoc(), RankedTensorType::get({}, rewriter.getI1Type()),
        allBroadcastable);

    // Add CustomCallOp and replace Cstr op with const witness, which is useful
    // for canonicalizer to remove the shape.assuming region.
    insertShapeAssertionCustomCall(rewriter, op->getLoc(),
                                   allBroadcastableScalar);
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op.getOperation(), true);
    return success();
  }
};

bool hasIndexStyle(Value value) {
  if (value.getType().isIndex()) return true;
  auto type = mlir::dyn_cast<ShapedType>(value.getType());
  return type && type.getElementType().isIndex();
}

struct ConvertShapeToStablehloWithConstraintsPass
    : public impl::ConvertShapeToStablehloWithConstraintsPassBase<
          ConvertShapeToStablehloWithConstraintsPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<shape::ShapeDialect>();
    target.addIllegalDialect<tensor::TensorDialect>();
    target.addIllegalOp<arith::IndexCastOp>();
    target.addIllegalOp<arith::MulIOp>();
    target.addDynamicallyLegalDialect<::mlir::stablehlo::StablehloDialect>(
        [](Operation* op) {
          return !llvm::any_of(op->getOperands(), hasIndexStyle);
        });
    target.addLegalOp<tensor::CastOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<shape::ConstWitnessOp, shape::AssumingOp,
                      shape::AssumingYieldOp>();

    RewritePatternSet patterns(&getContext());
    ::mlir::stablehlo::populateShapeToStablehloPatterns(&getContext(),
                                                        &patterns);

    patterns.add<ConvertCstrBroadcastableOp>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::quant::stablehlo
