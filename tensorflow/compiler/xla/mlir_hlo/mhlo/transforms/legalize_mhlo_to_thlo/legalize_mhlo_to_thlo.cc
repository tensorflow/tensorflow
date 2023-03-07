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
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/legalize_to_linalg_utils.h"
#include "mhlo/utils/mhlo_scatter_gather_utils.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEMHLOTOTHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

Value castToIndex(OpBuilder& b, Location loc, TensorType originalType,
                  Value value) {
  Type elementTy = originalType.getElementType();
  if (elementTy.isIndex()) return value;

  Type ty = RankedTensorType::get(originalType.getShape(), b.getIndexType());
  return elementTy.isUnsignedInteger()
             ? b.create<arith::IndexCastUIOp>(loc, ty, value).getResult()
             : b.create<arith::IndexCastOp>(loc, ty, value).getResult();
}

struct ConcatenateOpPattern : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    const int64_t concatDim = op.getDimension();
    const Location loc = op.getLoc();
    const Value anyOperand = adaptor.getVal().front();

    auto resultTy = typeConverter->convertType(op.getResult().getType())
                        .cast<RankedTensorType>();
    const ArrayRef<int64_t> resultShape = resultTy.getShape();
    const int64_t rank = resultTy.getRank();

    // Determine empty tensor size.
    SmallVector<int64_t> staticInitSizes(resultShape.begin(),
                                         resultShape.end());
    SmallVector<Value> dynamicInitSizes;
    for (int64_t i = 0; i < rank; ++i) {
      // No need to materialize anything for static dimensions.
      if (staticInitSizes[i] != ShapedType::kDynamic) {
        continue;
      }

      // For all dimensions other than the concatenation dimension, we can copy
      // the size from any operand.
      if (i != static_cast<int64_t>(concatDim)) {
        dynamicInitSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, anyOperand, i));
        continue;
      }

      // For the concatenation dimensions, sum up the sizes of all operands in
      // that dimension.
      int64_t staticSum = 0;
      Value dynamicSum;
      for (const Value operand : adaptor.getVal()) {
        auto operandTy = operand.getType().cast<RankedTensorType>();
        if (operandTy.getDimSize(concatDim) == ShapedType::kDynamic) {
          const Value dynamicSummand =
              rewriter.create<tensor::DimOp>(loc, operand, concatDim);
          if (dynamicSum) {
            dynamicSum =
                rewriter.create<arith::AddIOp>(loc, dynamicSum, dynamicSummand);
          } else {
            dynamicSum = dynamicSummand;
          }
        } else {
          staticSum += operandTy.getDimSize(concatDim);
        }
      }
      assert(dynamicSum && "expect at least one dynamic summand in this case");
      if (staticSum != 0) {
        dynamicSum = rewriter.create<arith::AddIOp>(
            loc, dynamicSum,
            rewriter.create<arith::ConstantIndexOp>(loc, staticSum));
      }
      dynamicInitSizes.push_back(dynamicSum);
    }

    // Create empty tensor and the new concat op.
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, staticInitSizes, resultTy.getElementType(), dynamicInitSizes);
    rewriter.replaceOpWithNewOp<thlo::ConcatenateOp>(
        op, resultTy, adaptor.getVal(), emptyTensor,
        rewriter.getIndexAttr(concatDim));
    return success();
  }
};

struct DynamicBroadcastInDimOpPattern
    : public OpConversionPattern<mhlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern<mhlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    Value outputDimensions = adaptor.getOutputDimensions();
    auto operandTy = adaptor.getOperand().getType().cast<RankedTensorType>();
    auto resultTy =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    // Only  apply to broadcasts that cannot be lowered to linalg, i.e. those
    // for which we do not know their expansion behavior at compile time.
    int64_t countKnownExpansionBehavior = 0;
    if (auto expandingDims = op.getKnownExpandingDimensions()) {
      countKnownExpansionBehavior += expandingDims->size();
    }
    if (auto nonexpandingDims = op.getKnownNonexpandingDimensions()) {
      countKnownExpansionBehavior += nonexpandingDims->size();
    }
    if (operandTy.getRank() == countKnownExpansionBehavior) return failure();

    // Create empty tensor as none of the operands are reusable/updatable.
    SmallVector<Value> dynamicDims;
    SmallVector<int64_t> staticShapeInfo;
    for (int i = 0; i < resultTy.getRank(); i++) {
      dynamicDims.push_back(rewriter.create<tensor::ExtractOp>(
          loc, outputDimensions,
          ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, i)}));
      staticShapeInfo.push_back(ShapedType::kDynamic);
    }
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, staticShapeInfo, resultTy.getElementType(), dynamicDims);

    auto broadcastDims = rewriter.getDenseI64ArrayAttr(
        llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>()));

    DenseI64ArrayAttr knownExpandingDims;
    if (op.getKnownExpandingDimensions().has_value()) {
      knownExpandingDims = rewriter.getDenseI64ArrayAttr(llvm::to_vector(
          op.getKnownExpandingDimensionsAttr().getValues<int64_t>()));
    }
    DenseI64ArrayAttr knownNonexpandingDims;
    if (op.getKnownNonexpandingDimensions().has_value()) {
      knownNonexpandingDims = rewriter.getDenseI64ArrayAttr(llvm::to_vector(
          op.getKnownNonexpandingDimensionsAttr().getValues<int64_t>()));
    }

    rewriter.replaceOpWithNewOp<thlo::DynamicBroadcastInDimOp>(
        op, resultTy, adaptor.getOperand(), emptyTensor, broadcastDims,
        knownExpandingDims, knownNonexpandingDims);
    return success();
  }
};

// Rewrites simple gather patterns (as checked below).
struct GatherPattern : public OpConversionPattern<mhlo::GatherOp> {
  using OpConversionPattern<mhlo::GatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!isCanonicalGather(op)) return failure();
    auto startIndicesType =
        adaptor.getStartIndices().getType().dyn_cast<RankedTensorType>();
    auto operandType =
        adaptor.getOperand().getType().dyn_cast<RankedTensorType>();

    if (!startIndicesType || !operandType) return failure();

    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(resultType.getRank());
    if (resultType.getDimSize(0) != ShapedType::kDynamic) {
      sizes.push_back(rewriter.getI64IntegerAttr(resultType.getDimSize(0)));
    } else {
      sizes.push_back(
          rewriter
              .create<tensor::DimOp>(op.getLoc(), adaptor.getStartIndices(), 0)
              .getResult());
    }
    llvm::copy(op.getSliceSizes().getValues<IntegerAttr>(),
               std::back_inserter(sizes));

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), sizes, resultType.getElementType());
    rewriter.replaceOpWithNewOp<thlo::GatherOp>(
        op, resultType, adaptor.getOperand(),
        castToIndex(rewriter, op.getLoc(), op.getStartIndices().getType(),
                    adaptor.getStartIndices()),
        emptyTensor);
    return success();
  }
};

bool isInBodyOfThloOp(Operation* op) {
  auto* parentOp = op->getParentRegion()->getParentOp();
  return isa<thlo::ScatterOp>(*parentOp) || isa<thlo::SortOp>(*parentOp);
}

// Rewrites a mhlo::ReturnOp inside a thlo::ReductionOp to thlo::YieldOp.
struct ThloRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!isInBodyOfThloOp(op)) return failure();
    SmallVector<Value, 4> operands(adaptor.getOperands());
    auto loc = op.getLoc();
    for (size_t i = 0; i < operands.size(); ++i) {
      if (operands[i].getType().isa<ShapedType>()) {
        operands[i] = rewriter.create<tensor::ExtractOp>(loc, operands[i]);
      }
    }
    rewriter.replaceOpWithNewOp<thlo::YieldOp>(op, operands);
    return success();
  }
};

struct ScatterPattern : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only canonicalized single-result scatter ops are supported.
    if (!isCanonicalScatter(op) || op.getNumResults() != 1) return failure();

    auto opType =
        typeConverter->convertType(op.getType(0)).dyn_cast<RankedTensorType>();
    if (!opType) return failure();

    Location loc = op.getLoc();
    auto thloScatter = rewriter.create<thlo::ScatterOp>(
        loc, opType,
        castToIndex(rewriter, loc, op.getScatterIndices().getType(),
                    adaptor.getScatterIndices()),
        adaptor.getUpdates().front(), adaptor.getInputs().front());

    Region& region = thloScatter.getUpdateComputation();
    rewriter.inlineRegionBefore(op.getRegion(), region, region.end());

    // Convert the signature of the body by inserting
    // tensor.from_elements/tensor.extract.
    TypeConverter::SignatureConversion signatureConverter(2);
    for (const auto& [idx, val] : llvm::enumerate(
             thloScatter.getUpdateComputation().getArgumentTypes())) {
      signatureConverter.addInputs(
          1 - idx, typeConverter->convertType(
                       val.cast<RankedTensorType>().getElementType()));
    }
    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());

    rewriter.replaceOp(op, thloScatter.getResults());
    return success();
  }
};

struct SortPattern : public OpConversionPattern<mhlo::SortOp> {
  using OpConversionPattern<mhlo::SortOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SortOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();

    SmallVector<Value> outputs;
    SmallVector<RankedTensorType> operandTypes;
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    for (auto [operand, resultType] :
         llvm::zip(adaptor.getInputs(), resultTypes)) {
      RankedTensorType operandType =
          operand.getType().dyn_cast<RankedTensorType>();
      if (!operandType)
        return rewriter.notifyMatchFailure(op, "expects known-rank operands");
      operandTypes.push_back(operandType);
      auto tensorResultType = resultType.cast<RankedTensorType>();

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, tensorResultType.getShape(), tensorResultType.getElementType());

      outputs.push_back(emptyTensor);
    }

    int64_t dimension = op.getDimension();
    // TODO(bchetioui): MHLO accepts dimensions in the range [-rank, rank),
    // while THLO accepts only dimensions in the range [0, rank). Ideally, they
    // should agree on the range of acceptable arguments, but while it is not
    // the case, this is a (reliable) workaround.
    if (dimension < 0) dimension = dimension + operandTypes.front().getRank();
    bool isStable = op.getIsStable();

    auto thloSort = rewriter.create<thlo::SortOp>(
        loc, resultTypes, adaptor.getInputs(), outputs,
        rewriter.getIndexAttr(dimension), rewriter.getBoolAttr(isStable));

    Region& region = thloSort.getComparator();
    rewriter.inlineRegionBefore(op.getComparator(), region, region.end());

    assert(thloSort.getNumDpsInputs() == thloSort.getNumDpsInits());

    // Convert the signature of the comparator.
    TypeConverter::SignatureConversion signatureConverter(
        thloSort.getNumDpsInputs() * 2);
    for (const auto& [idx, val] : llvm::enumerate(operandTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/2 * idx,
          typeConverter->convertType(val.getElementType()));
      signatureConverter.addInputs(
          /*origInputNo=*/2 * idx + 1,
          typeConverter->convertType(val.getElementType()));
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());

    rewriter.replaceOp(op, thloSort.getResults());
    return success();
  }
};

struct ReversePattern : public OpConversionPattern<mhlo::ReverseOp> {
  using OpConversionPattern<mhlo::ReverseOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto reverseDimensions =
        llvm::to_vector(op.getDimensions().getValues<int64_t>());
    Type resultType = typeConverter->convertType(op->getResultTypes()[0]);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    Location loc = op.getLoc();
    auto operandType =
        adaptor.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType)
      return rewriter.notifyMatchFailure(op, "expects known-rank operand");
    auto tensorResultType = resultType.cast<RankedTensorType>();
    SmallVector<Value, 8> dynShape =
        tensor::createDynamicDimValues(rewriter, loc, adaptor.getOperand());
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, tensorResultType.getShape(), tensorResultType.getElementType(),
        dynShape);
    rewriter.replaceOpWithNewOp<thlo::ReverseOp>(
        op, resultType, adaptor.getOperand(), initTensor, reverseDimensions);
    return success();
  }
};

class LegalizeMHLOToTHLOPass
    : public impl::LegalizeMHLOToTHLOPassBase<LegalizeMHLOToTHLOPass> {
 public:
  explicit LegalizeMHLOToTHLOPass(bool enableExperimentalOps) {
    enableExperimental = enableExperimentalOps;
  }

 private:
  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    // clang-format off
    target.addLegalDialect<
        arith::ArithDialect,
        complex::ComplexDialect,
        linalg::LinalgDialect,
        math::MathDialect,
        shape::ShapeDialect,
        tensor::TensorDialect,
        thlo::THLODialect>();
    // clang-format on
    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = std::make_unique<LinalgTypeConverter>();

    populateScalarHloToArithmeticConversionPatterns(
        ctx, *typeConverter, &patterns,
        [](Operation* op) { return isInBodyOfThloOp(op); });

    // List of patterns.
    // clang-format off
    patterns.insert<
        ConcatenateOpPattern,
        ReversePattern,
        ScatterPattern,
        SortPattern,
        ThloRegionReturnOpConversion>(*typeConverter, ctx);
    // clang-format on

    if (enableExperimental) {
      // clang-format off
      patterns.insert<
          DynamicBroadcastInDimOpPattern,
          GatherPattern>(*typeConverter, ctx);
      // clang-format on
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMHLOToTHLOPass(
    bool enableExperimentalOps) {
  return std::make_unique<LegalizeMHLOToTHLOPass>(enableExperimentalOps);
}

}  // namespace mhlo
}  // namespace mlir
